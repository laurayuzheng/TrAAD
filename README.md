# [TrAAD] Traffic-Aware Autonomous Driving 

<!-- ![teaser](https://github.com/dianchen96/LearningByCheating/blob/release-0.9.6/figs/fig1.png "Pipeline") -->
> [**Traffic-Aware Autonomous Driving**](https://arxiv.org/pdf/2210.03772)    
> Laura Zheng, Sanghyun Son, Ming Lin        
> [International Conference on Robotics and Automation](https://www.icra2023.org/) (ICRA 2023)      

# Table of Contents
1. [Installation and Setup](#installation-and-setup)
2. [Dataset](#dataset)
3. [Models](#models)
4. [Training](#training)
5. [Credits](#credits)

## Installation and Setup

The code has only been tested on Ubuntu 20.04. 

### Dependencies 
- Python 3.8
- [[Install]](https://pytorch.org/get-started/previous-versions/) PyTorch 1.11.0 
- [[Install]](https://github.com/eclipse/sumo/releases/tag/v1_10_0) SUMO 1.10.0 (~1.2GB) 
- [[Install]](https://github.com/carla-simulator/carla/releases/tag/0.9.10.1) CARLA 0.9.10.1 (~12.3GB) 

### Environment Setup 

Install SUMO and CARLA from the above, and make sure to know the installation location.
This is important later on for modifying scripts for use on your local computer.

``` 
git clone https://github.com/laurayuzheng/traffic_aware_AD.git --recursive

conda create -f environment.yml 

conda activate traad

pip install --upgrade pip 

pip install -e external/stable-baselines3 
pip install -e external/stable-baselines3-contrib
```

Then, install Pytorch accordingly with your system using the link in the subsection above.

Lastly, modify the paths in [scripts/path_config.sh](./scripts/path_config.sh) and [config.py](./config.py) to your local system. 

## Dataset 

### [[DATASET DOWNLOAD (UMD Box)]](https://umd.box.com/s/02iic1kzb9e4t9c6iytkgttuphd8f7he)

Extracted datasets should go into the ./data folder.

### Collecting Your Own Dataset 

```
bash collect_data.sh [train/test]
```

Collected data will be stored in ./data folder. 
Modify the DISPLAY variable to turn off visualization (0 == OFF, 1 == ON).


## Models 

### [[TRAINED MODELS DOWNLOAD (UMD Box)]](https://umd.box.com/s/gxuw9jod9brineb5httsa752wo75xern)

## Training 

### Phase 1: Learn to Accelerate 
TODO: Work on this section
``` 
bash scripts/train_phase1.sh
```

Note: The repo for DiffPPO and DiffTRPO is a WIP, so only PPO and TRPO are supported right now.

### Phase 2: Learn Everything Else

Training the map (cheating) agent:
```
bash scripts/train_phase2.sh
```

To train the image (student) agent, modify the [train_phase2.sh](./scripts/train_phase2.sh) file. The lower half should have a section (commented out) for running the training of phase 2. You will need to find the checkpoint model produced by phase 1 in the [checkpoints](./checkpoints/) folder.

## Credits 

Much of this repository is built based off of previously released code; thank you to the authors who made them available! 

- [Learning by Cheating repository by Brady Zhou](https://github.com/bradyz/2020_CARLA_challenge)
- [TransFuser by AutonomousVision](https://github.com/autonomousvision/transfuser)