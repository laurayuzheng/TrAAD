import uuid
import argparse
import pathlib

import numpy as np
import torch
import pytorch_lightning as pl
import torchvision
import wandb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image, ImageDraw

from .models import SegmentationModel, RawController, AccelAgentNetwork
from .utils.heatmap import ToHeatmap
from .dataset import get_dataset
from .converter import Converter
from . import common

import gym

# from .traffic.d_car_following_models import IDMStepLayer
from .traffic.rewards import *
from grad import IDMStepLayer

import torch
from torch import det
import numpy as np
# from FLOW_CONFIG import *
from stable_baselines3 import PPO

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)

@torch.no_grad()
def visualize(batch, out, between, out_steer, out_accel, loss_point, loss_cmd, target_heatmap):
    images = list()

    for i in range(out.shape[0]):
        _loss_point = loss_point[i] if loss_point is not None else -1
        _loss_cmd = loss_cmd[i] if loss_cmd is not None else -1
        _out = out[i] if out is not None else [-1]
        # _out_cmd = out_cmd[i]
        _out_steer = out_steer[i] if out_steer is not None else [-1]
        _out_accel = out_accel[i] if out_accel is not None else [-1]
        _between = between[i] if between is not None else [-1]

        rgb, topdown, _, points, target, actions, meta, _, _, _ = [x[i] for x in batch]

        _rgb = np.uint8(rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255)
        _target_heatmap = np.uint8(target_heatmap[i].detach().squeeze().cpu().numpy() * 255)
        _target_heatmap = np.stack(3 * [_target_heatmap], 2)
        _target_heatmap = Image.fromarray(_target_heatmap)
        _topdown = Image.fromarray(common.COLOR[topdown.argmax(0).detach().cpu().numpy()])
        _draw = ImageDraw.Draw(_topdown)

        _draw.ellipse((target[0]-2, target[1]-2, target[0]+2, target[1]+2), (255, 255, 255))

        for x, y in points:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

        for x, y in _out:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        for x, y in _between:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw.ellipse((x-1, y-1, x+1, y+1), (0, 255, 0))

        _draw.text((5, 10), 'Point: %.3f' % _loss_point)
        _draw.text((5, 30), 'Command: %.3f' % _loss_cmd)
        _draw.text((5, 50), 'Meta: %s' % meta)

        _draw.text((5, 90), 'Steer, Accel Label: %.3f %.3f' % tuple(actions))
        _draw.text((5, 110), 'Steer Pred: %.3f' % _out_steer)
        _draw.text((5, 130), 'Accel Pred: %.3f' % _out_accel)

        image = np.array(_topdown).transpose(2, 0, 1)
        images.append((_loss_cmd, torch.ByteTensor(image)))

    images.sort(key=lambda x: x[0], reverse=True)

    result = torchvision.utils.make_grid([x[1] for x in images], nrow=4)
    result = wandb.Image(result.numpy().transpose(1, 2, 0))

    return result


# This module's only responsibility is learn to steer
class TrafficMapModelSteer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.to_heatmap = ToHeatmap(hparams.heatmap_radius)
        self.net = SegmentationModel(10, 4, hack=hparams.hack, temperature=hparams.temperature)
        self.controller = RawController(4, n_classes=2)
        self.converter = Converter()
        self.step_layer = IDMStepLayer.apply

        # self.accel_controller = PPO.load(hparams.accel_model_path)
        # self.accel_controller.to("cpu")

        model = PPO.load(hparams.accel_model_path)
        self.accel_controller = model.policy
        # self.accel_controller = AccelAgentNetwork(model.policy.features_extractor, model.policy.net_arch, model.policy.action_net, model.policy.value_net)

    def forward(self, topdown, target, debug=False):
        target_heatmap = self.to_heatmap(target, topdown)[:, None]
        out = self.net(torch.cat((topdown, target_heatmap), 1))

        if not debug:
            return out

        return out, (target_heatmap,)

    def training_step(self, batch, batch_nb): # img never used in map model
        img, topdown, topdown_raw, points, target, actions, meta, traffic_state, player_ind, num_veh = batch
        steer_actions = actions[:,0:1]
        speeds = actions[:,1].unsqueeze(1)

        observation = {
            'topdown': topdown_raw, 
            'traffic_state': traffic_state
        }

        out, (target_heatmap,) = self.forward(topdown, target, debug=True) # Generate waypoints 
        accel, _, _ = self.accel_controller(observation, deterministic=True)
        target_speeds = torch.add(speeds, accel)
        new_actions = torch.cat((steer_actions, target_speeds), dim=1)
        
        alpha = torch.rand(out.shape).type_as(out)
        between = alpha * out + (1-alpha) * points # Interpolate between predicted waypoints and ground truth waypoints
        out_ctrl = self.controller(between)        

        loss_point = torch.nn.functional.l1_loss(out, points, reduction='none').mean((1, 2))
        loss_cmd_raw = torch.nn.functional.l1_loss(out_ctrl, new_actions, reduction='none')

        loss_cmd = loss_cmd_raw.mean(1)
        loss = (loss_point + self.hparams.command_coefficient * loss_cmd).mean()

        metrics = {
                'loss': loss.item(),
                'point_loss': loss_point.mean().item(),
                'loss_cmd': loss_cmd.mean().item(),
                }

        if batch_nb % 250 == 0:
            metrics['train_image'] = visualize(batch, out, between, out_ctrl[:,0], out_ctrl[:,1], loss_point, loss_cmd, target_heatmap)

        self.logger.log_metrics(metrics, self.global_step)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        img, topdown, topdown_raw, points, target, actions, meta, traffic_state, player_ind, num_veh = batch
        steer_actions = actions[:,0:1]
        speeds = actions[:,1].unsqueeze(1)

        observation = {
            'topdown': topdown_raw, 
            'traffic_state': traffic_state
        }

        out, (target_heatmap,) = self.forward(topdown, target, debug=True)
        accel, _, _ = self.accel_controller(observation, deterministic=True)
        target_speeds = torch.add(speeds, accel)
        new_actions = torch.cat((steer_actions, target_speeds), dim=1)

        alpha = 0.0
        between = alpha * out + (1-alpha) * points

        out_cmd = self.controller(between)
        out_cmd_pred = self.controller(out)

        loss_point = torch.nn.functional.l1_loss(out, points, reduction='none').mean((1, 2))
        loss_cmd_raw = torch.nn.functional.l1_loss(out_cmd, new_actions, reduction='none')
        loss_cmd_pred_raw = torch.nn.functional.l1_loss(out_cmd_pred, new_actions, reduction='none')

        loss_cmd = loss_cmd_raw.mean(1)
        loss = (loss_point + self.hparams.command_coefficient * loss_cmd).mean()

        if batch_nb == 0:
            self.logger.log_metrics({
                'val_image': visualize(batch, out, between, out_cmd_pred[:,0], out_cmd_pred[:,1], loss_point, loss_cmd, target_heatmap)
                }, self.global_step)

        return {
                'val_loss': loss.item(),
                'val_point_loss': loss_point.mean().item(),
                'val_cmd_loss': loss_cmd.mean().item(),
                'val_cmd_pred_loss': loss_cmd_pred_raw[:, 0].mean().item(), 
                }

    def validation_epoch_end(self, batch_metrics):
        results = dict()

        for metrics in batch_metrics:
            for key in metrics:
                if key not in results:
                    results[key] = list()

                results[key].append(metrics[key])

        summary = {key: np.mean(val) for key, val in results.items()}
        self.logger.log_metrics(summary, self.global_step)

        return summary

    def configure_optimizers(self):
        optim = torch.optim.Adam(
                list(self.net.parameters()) + list(self.controller.parameters()),
                lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode='min', factor=0.5, patience=5, min_lr=1e-6,
                verbose=True)

        return [optim], [scheduler]

    def train_dataloader(self):
        return get_dataset(self.hparams.dataset_dir, True, self.hparams.batch_size, sample_by=self.hparams.sample_by, traffic=True)

    def val_dataloader(self):
        return get_dataset(self.hparams.dataset_dir, False, self.hparams.batch_size, sample_by=self.hparams.sample_by, traffic=True)


def main(hparams):
    model = TrafficMapModelSteer(hparams)
    logger = WandbLogger(id=hparams.id, save_dir=str(hparams.save_dir), project='ppo_mapmodel')
    checkpoint_callback = ModelCheckpoint(hparams.save_dir, save_top_k=1)

    try:
        resume_from_checkpoint = sorted(hparams.save_dir.glob('*.ckpt'))[-1]
    except:
        resume_from_checkpoint = None

    gpus = 0 if hparams.cpu else -1 

    trainer = pl.Trainer(
            gpus=gpus, max_epochs=hparams.max_epochs,
            resume_from_checkpoint=resume_from_checkpoint,
            logger=logger, checkpoint_callback=checkpoint_callback)

    trainer.fit(model)

    wandb.save(str(hparams.save_dir / '*.ckpt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--save_dir', type=pathlib.Path, default='results/phase1')
    parser.add_argument('--id', type=str, default=uuid.uuid4().hex)

    parser.add_argument('--heatmap_radius', type=int, default=5)
    parser.add_argument('--sample_by', type=str, choices=['none', 'even', 'speed', 'steer'], default='even')
    parser.add_argument('--command_coefficient', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--hack', action='store_true', default=False)

    # Data args.
    parser.add_argument('--dataset_dir', type=pathlib.Path, required=True)
    parser.add_argument('--accel_model_path', type=pathlib.Path, default="models/accel_agent")
    parser.add_argument('--batch_size', type=int, default=64)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--cpu', action='store_true', default=False)

    parsed = parser.parse_args()
    parsed.save_dir = parsed.save_dir / parsed.id
    parsed.save_dir.mkdir(parents=True, exist_ok=True)

    main(parsed)
