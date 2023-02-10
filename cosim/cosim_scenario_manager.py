#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import signal
import sys
import time

import py_trees
import carla
import datetime
from csv import writer
import numpy as np

sys.path.append("./")
sys.path.append("./external")
sys.path.append("./external/leaderboard")
sys.path.append("./external/scenario_runner")
sys.path.append("./agents")

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider

from sumo_integration.bridge_helper import BridgeHelper  # pylint: disable=wrong-import-position
from sumo_integration.constants import INVALID_ACTOR_ID  # pylint: disable=wrong-import-position


class CosimScenarioManager(ScenarioManager):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """


    def __init__(self, timeout, debug_mode=False, use_watchdog=True):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = float(timeout)

        # Used to detect if the simulation is down
        watchdog_timeout = max(5, self._timeout - 2)

        if use_watchdog:
            self._watchdog = Watchdog(watchdog_timeout)
        else: 
            self._watchdog = None

        # Avoid the agent from freezing the simulation
        agent_timeout = watchdog_timeout - 1
        self._agent_watchdog = Watchdog(agent_timeout)

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

        # Register the scenario tick as callback for the CARLA world
        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)

        self.traffic_info_csv = "traffic_flow_metadata_"
        self.traffic_info_csv_fuel = "traffic_fuel_metadata_"
        now = datetime.datetime.now()
        self.traffic_info_csv += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self.traffic_info_csv += ".csv"
        self.traffic_info_csv_fuel += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self.traffic_info_csv_fuel += ".csv"
        self.scenario_traffic_info = []
        self.scenario_traffic_info_fuel = []

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._running = False


    def load_scenario(self, scenario, agent, rep_number):
        """
        Load a new scenario
        """

        super().load_scenario(scenario, agent, rep_number)
        self.scenario_traffic_info.append(self.scenario.name)
        self.scenario_traffic_info_fuel.append(self.scenario.name)

    def start_scenario_rl(self):
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        # self._watchdog.start()
        self._running = True
        # ticks = 0

    def run_scenario(self, max_ticks=100000):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True
        ticks = 0

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            try:
                if timestamp and ticks < max_ticks:
                    self._tick_scenario(timestamp)
                    ticks += 1
                
                if ticks >= max_ticks:
                    break 

            except:
                break
                

    def run_scenario_cosim(self, sync_obj, max_ticks=1000, record_traffic=False):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True
        ticks = 0

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            
            try:
                if timestamp and ticks < max_ticks:
                    self._tick_cosim_scenario(timestamp, sync_obj, record_traffic=record_traffic)
                    ticks += 1
                
                if ticks >= max_ticks:
                    break 
            except:
                break
                # self._tick_scenario(timestamp)

    def _tick_cosim_scenario(self, timestamp, sync_obj, warmup=False, record_traffic=False):
        """
        Run next tick of scenario and the agent and tick the world.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            try:
                # print("Number of ego vehicles: ", len(self.ego_vehicles))
                ticked_sync = False 
                if not sync_obj.sumo.player_id:
                    print("Adding ego vehicles to SUMO.. ")
                    carla_actor = self.ego_vehicles[0]
                    id = carla_actor.id
                    type_id = BridgeHelper.get_sumo_vtype(carla_actor)
                    # color = self._player.attributes.get('color', None) 
                    color = None
                    if type_id is not None:
                        sumo_actor_id = sync_obj.sumo.spawn_actor(type_id, color)
                        if sumo_actor_id != INVALID_ACTOR_ID:
                            sync_obj.carla2sumo_ids[id] = sumo_actor_id
                            sync_obj.carla_sumo2carla_ids[sumo_actor_id] = id
                            sync_obj.sumo.subscribe(sumo_actor_id)
                    
                        sync_obj._player_sumo_id = sumo_actor_id
                        sync_obj.sumo.player_id = sumo_actor_id
                        sync_obj.carla.player_id = id 
                        ticked_sync = True 

                        carla_actor = sync_obj.carla.get_actor(id)

                        sumo_transform = BridgeHelper.get_sumo_transform(carla_actor.get_transform(),
                                                                        carla_actor.bounding_box.extent)
                        sumo_lights = None

                        # print("Synchronizing vehicle ", sumo_actor_id)
                        sync_obj.sumo.synchronize_vehicle(sumo_actor_id, sumo_transform, sumo_lights)
                        sync_obj.sumo.tick()
                #         sync_obj.tick()
                
                # print("Getting ego action.. ")
                ego_action = self._agent()

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)


            self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario
            self.scenario_tree.tick_once()

            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(
                    self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

            spectator = CarlaDataProvider.get_world().get_spectator()
            ego_trans = self.ego_vehicles[0].get_transform()
            spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
                                                        carla.Rotation(pitch=-90)))

        if self._running and self.get_running_status():
            
            if sync_obj.sumo.player_has_result() == False: 
                print("Adding ego vehicles to SUMO.. ")
                carla_actor = self.ego_vehicles[0]
                id = carla_actor.id
                type_id = BridgeHelper.get_sumo_vtype(carla_actor)
                # color = self._player.attributes.get('color', None) 
                color = None
                if type_id is not None:
                    sumo_actor_id = sync_obj.sumo.spawn_actor(type_id, color)
                    if sumo_actor_id != INVALID_ACTOR_ID:
                        sync_obj.carla2sumo_ids[id] = sumo_actor_id
                        sync_obj.carla_sumo2carla_ids[sumo_actor_id] = id
                        sync_obj.sumo.subscribe(sumo_actor_id)
                
                    sync_obj._player_sumo_id = sumo_actor_id
                    sync_obj.sumo.player_id = sumo_actor_id
                    sync_obj.carla.player_id = id 

                    ticked_sync = True

                    carla_actor = sync_obj.carla.get_actor(id)
                    sumo_transform = BridgeHelper.get_sumo_transform(carla_actor.get_transform(),
                                                                    carla_actor.bounding_box.extent)
                    sumo_lights = None

                    # print("Synchronizing vehicle ", sumo_actor_id)
                    sync_obj.sumo.synchronize_vehicle(sumo_actor_id, sumo_transform, sumo_lights)

                    sync_obj.sumo.tick()

            if ticked_sync:
                sync_obj.tick()
            else:
                CarlaDataProvider.get_world().tick(self._timeout)
            
            if record_traffic and sync_obj.sumo.player_has_result(): 
                state, _ = sync_obj.get_state()
                fuel_consumption = sync_obj.sumo.get_playerlane_fuel_consumption()

                # print(state)
                state = np.array(state)
                fuel_consumption = np.array(fuel_consumption)

                # print(state)
                avg_vel = state[1::2].mean().squeeze()
                avg_fuel_consumption = fuel_consumption.mean().squeeze()
                # print(avg_vel)
                self.scenario_traffic_info.append(avg_vel)
                self.scenario_traffic_info_fuel.append(avg_fuel_consumption)

            # sync_obj.tick()

    def save_traffic_info_scenario(self):
        with open(self.traffic_info_csv, 'a+') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(self.scenario_traffic_info)

        with open(self.traffic_info_csv_fuel, 'a+') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(self.scenario_traffic_info_fuel)
        
        self.scenario_traffic_info = []
        self.scenario_traffic_info_fuel = []
        

    def run_tick_cosim_rl(self, sync_obj, action):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """

        # self._watchdog.start()
        # self._running = True

        if self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            
            try:
                if timestamp:
                    # print("Timestamp exists, using action ", action)
                    result = self.tick_cosim_scenario_rl(timestamp, sync_obj, action)
                    return result

            except:
                print("Exception at cosim tick")
                self._running = False 
                None, None, None, True, None, None, None
        else: 
            return None, None, None, True, None, None, None
            # self._tick_scenario(timestamp)


    def tick_cosim_scenario_rl(self, timestamp, sync_obj, action, warmup=False):
        """
        Run next tick of scenario and the agent and tick the world.
        """ 
        # if self._running:

        # self._watchdog.update()
        # Update game time and actor information
        GameTime.on_carla_tick(timestamp)
        CarlaDataProvider.on_carla_tick()
        state = None 

        try:
            # print("Number of ego vehicles: ", len(self.ego_vehicles))
            ticked_sync = False 
            if not sync_obj.sumo.player_id:
                print("1 Adding ego vehicles to SUMO.. ")
                carla_actor = self.ego_vehicles[0]
                id = carla_actor.id
                type_id = BridgeHelper.get_sumo_vtype(carla_actor)
                # color = self._player.attributes.get('color', None) 
                color = None
                if type_id is not None:
                    sumo_actor_id = sync_obj.sumo.spawn_actor(type_id, color)
                    if sumo_actor_id != INVALID_ACTOR_ID:
                        sync_obj.carla2sumo_ids[id] = sumo_actor_id
                        sync_obj.carla_sumo2carla_ids[sumo_actor_id] = id
                        sync_obj.sumo.subscribe(sumo_actor_id)
                
                    sync_obj._player_sumo_id = sumo_actor_id
                    sync_obj.sumo.player_id = sumo_actor_id
                    sync_obj.carla.player_id = id 
                    ticked_sync = True 

                    carla_actor = sync_obj.carla.get_actor(id)

                    sumo_transform = BridgeHelper.get_sumo_transform(carla_actor.get_transform(),
                                                                    carla_actor.bounding_box.extent)
                    sumo_lights = None

                    # print("Synchronizing vehicle ", sumo_actor_id)
                    sync_obj.sumo.synchronize_vehicle(sumo_actor_id, sumo_transform, sumo_lights)
                    sync_obj.sumo.tick()
            #         sync_obj.tick()
        
            ego_action, img, state, start_player_ind, fuel_consumption = self._agent._agent.forward_step(action)


        # Special exception inside the agent that isn't caused by the agent
        except SensorReceivedNoData as e:
            self._running = False 
            raise RuntimeError(e)

        except Exception as e:
            self._running = False 
            raise AgentError(e) 
        
        except: 
            raise

        self.ego_vehicles[0].apply_control(ego_action)
        
        # Tick scenario
        self.scenario_tree.tick_once()

        if self._debug_mode:
            print("\n")
            py_trees.display.print_ascii_tree(
                self.scenario_tree, show_status=True)
            sys.stdout.flush()

        if self.scenario_tree.status != py_trees.common.Status.RUNNING:
            self._running = False 

        spectator = CarlaDataProvider.get_world().get_spectator()
        ego_trans = self.ego_vehicles[0].get_transform()
        spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
                                                    carla.Rotation(pitch=-90)))

        # if self._running:
            
        if sync_obj.sumo.player_has_result() == False: 
            print("2 Adding ego vehicles to SUMO.. ")
            carla_actor = self.ego_vehicles[0]
            id = carla_actor.id
            type_id = BridgeHelper.get_sumo_vtype(carla_actor)
            # color = self._player.attributes.get('color', None) 
            color = None
            if type_id is not None:
                sumo_actor_id = sync_obj.sumo.spawn_actor(type_id, color)
                if sumo_actor_id != INVALID_ACTOR_ID:
                    sync_obj.carla2sumo_ids[id] = sumo_actor_id
                    sync_obj.carla_sumo2carla_ids[sumo_actor_id] = id
                    sync_obj.sumo.subscribe(sumo_actor_id)
            
                sync_obj._player_sumo_id = sumo_actor_id
                sync_obj.sumo.player_id = sumo_actor_id
                sync_obj.carla.player_id = id 

                ticked_sync = True

                carla_actor = sync_obj.carla.get_actor(id)
                sumo_transform = BridgeHelper.get_sumo_transform(carla_actor.get_transform(),
                                                                carla_actor.bounding_box.extent)
                sumo_lights = None

                # print("Synchronizing vehicle ", sumo_actor_id)
                sync_obj.sumo.synchronize_vehicle(sumo_actor_id, sumo_transform, sumo_lights)

                sync_obj.sumo.tick()

        if self._running:
            if ticked_sync:
                sync_obj.tick()
            else:
                CarlaDataProvider.get_world().tick(self._timeout)
            
            # sync_obj.tick()
        if state is not None:
            next_obs, player_ind = sync_obj.get_state()
        else: 
            next_obs, player_ind, fuel_consumption = None, None, None

        return img, next_obs, player_ind, not self._running, state, start_player_ind, fuel_consumption



    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        if self._watchdog is not None:
            self._watchdog.stop()

        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

        if self.scenario is not None:
            self.scenario.terminate()

        if self._agent is not None:
            self._agent.cleanup()
            self._agent = None

        self.analyze_scenario()
