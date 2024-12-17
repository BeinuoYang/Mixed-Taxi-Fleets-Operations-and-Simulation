from core.controller import Controller, Generator
from core.utils.units import minutes_to_seconds
from ..network.controller import NetworkController
from ..pedestrians.agents import PedestrianAgents
from ..pedestrians.logic import PedestrianLogic
from ..parking_lot.controller import ParkingLotController

import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import random
import csv

random.seed(0)


class PedestrianController(Controller):
    """
    pedestrian agent controller, just creates and destroys pedestrians

    contains no tasks, but has 5 states, all do nothing, but are there for KPI extraction
    """
    def init_controller(self):
        self.destroyed_peds = np.array([])
        self.served_peds = np.array([])
        self.ped_num = 0
        self.wait_time = {}

    def register_implementation(self):
        self.agent_group = PedestrianAgents("Pedestrian", self)
        self.agent_logic: PedestrianLogic = PedestrianLogic(self)
        self.agent_generator: PedGenerator = PedGenerator(self)
        self.network_controller = self.register_dependency(NetworkController)
        self.parkinglot_controller  = self.register_dependency(ParkingLotController)

    @property
    def clock(self):
        return self.world.clock

    def start_controller_post(self):
        import os

        parkinglot_num = self.params["parkinglot"].get("num")
        vehicle_num_per_type = ''.join(
            [str(int(x * 10)).zfill(2) for x in self.params["vehicles"].get("percent_per_type")])
        # output evaluation metrics
        file_path = "/".join([self.params["output_dir"], "{}-{}-pedestrians.csv".format(parkinglot_num, vehicle_num_per_type)])
        # file_path = "/".join([self.params["output_dir"], "pedestrians{}.csv".format(self.clock.real_starttime)])
        header = ["iter", "ped_num", "ped_served", "ped_served_num", "ped_destroyed", "ped_destroyed_num",
                  "percentage_fulfilled", "total_wait_time", "avg_wait_time"]
        percentage_fulfilled = len(self.served_peds)/self.ped_num if self.ped_num > 0 else 0
        record = [self.clock.elapsed_ticks, self.ped_num, self.served_peds, len(self.served_peds),
                  self.destroyed_peds, len(self.destroyed_peds), percentage_fulfilled]

        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"dt={self.clock.dt}"])
            writer.writerow(header)
            writer.writerow(record)

    def step_controller(self):
        elapsed_seconds = self.clock.dt * self.clock.elapsed_ticks
        horizon = self.params["horizon"]
        parkinglot_num = self.params["parkinglot"].get("num")
        vehicle_num_per_type = ''.join(
            [str(int(x * 10)).zfill(2) for x in self.params["vehicles"].get("percent_per_type")])
        if elapsed_seconds > horizon and (elapsed_seconds - self.clock.dt) % horizon == 0 and (self.clock.elapsed_ticks < self.clock.numticks):
            file_path = "/".join([self.params["output_dir"], "{}-{}-pedestrians.csv".format(parkinglot_num, vehicle_num_per_type)])
            # file_path = "/".join([self.params["output_dir"], "pedestrians{}.csv".format(self.clock.real_starttime)])
            percentage_fulfilled = len(self.served_peds) / self.ped_num if self.ped_num > 0 else 0
            total_wait_time = np.sum(np.array(list(self.wait_time.values())))
            avg_wait_time = total_wait_time / len(self.wait_time) if len(self.wait_time) > 0 else 0
            record = [self.clock.elapsed_ticks - 1, self.ped_num, self.served_peds, len(self.served_peds),
                      self.destroyed_peds, len(self.destroyed_peds), percentage_fulfilled,
                      total_wait_time, avg_wait_time]
            with open(file_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(record)


class PedGenerator(Generator):
    @property
    def clock(self):
        return self.owner.world.clock

    def init_generator_custom(self):
        self.gen_prob = 0.5
        # make this more easily accessible (i.e. from load)
        self.data_filename = "/".join([self.owner.params["instance_data_dir"], "customers_HZ.csv"])
        self.raw = None
        self.generate_at = 0
        self.raw_keys = None
        self.map_bounds = [float(i) for i in (self.owner.params['viewmap']).split(',')]
        self.ped_num = 0

    def step(self):
        self.create_agents_from_data()
        # self.create_agents_random()
        self.owner.ped_num = self.ped_num

    def start(self):
        self.raw = pd.read_csv(self.data_filename)
        # self.raw.sort_values(by=['pickup_timewindow_start', ], inplace=True)
        self.raw_keys = self.raw.index.tolist()
        # this is the old version original in the example

    def create_agents_from_data(self):
        current_time = self.clock.elapsed_ticks
        dt = self.clock.dt
        network = self.owner.network_controller.network
        while (self.generate_at < len(self.raw_keys) and dt * current_time >=
               minutes_to_seconds(self.raw.loc[self.raw_keys[self.generate_at], 'pickup_timewindow_start'])):
            # random sampling
            # f = random.random()
            # if f > 0.05:
            #     self.generate_at += 1
            #     continue
            self.ped_num += 1
            next_row = self.raw.iloc[self.generate_at]
            passenger_id = next_row['customer_id']
            nearest_pickup_node = next_row['pickup_node_id']
            nearest_dropoff_node = next_row['dropoff_node_id']
            orig_lat = network.graph.nodes[nearest_pickup_node]['y']
            orig_lon = network.graph.nodes[nearest_pickup_node]['x']
            pickup_tw_start = minutes_to_seconds(next_row['pickup_timewindow_start'])
            # max_wait_time = minutes_to_seconds(10)
            max_wait_time = minutes_to_seconds(np.random.choice([10, 20], p=[0.8, 0.2]))
            pickup_tw_end = pickup_tw_start + max_wait_time
            dest_lat = network.graph.nodes[nearest_dropoff_node]['y']
            dest_lon = network.graph.nodes[nearest_dropoff_node]['x']
            pickup_dropoff_dura = minutes_to_seconds(0)  # tick

            self.owner.agent_group.create_next_pedestrians(passenger_id, orig_lat, orig_lon, nearest_pickup_node,
                                                           pickup_tw_start, pickup_tw_end, max_wait_time, dest_lat,
                                                           dest_lon, nearest_dropoff_node, pickup_dropoff_dura, True)
            self.generate_at += 1

    def create_agents_random(self):
        n = self.ped_num
        if n >= 200:
            return
        current_time = self.clock.elapsed_ticks
        dt = self.clock.dt
        network = self.owner.network_controller.network
        ped_num = max(0, int(random.gauss(mu=0.2, sigma=0.5)))
        self.ped_num += ped_num
        if ped_num == 0:
            return
        lon_max = self.map_bounds[1]
        lon_min = self.map_bounds[3]
        lat_max = self.map_bounds[0]
        lat_min = self.map_bounds[2]
        i = 0
        parking_lots = self.owner.parkinglot_controller.agent_group
        pl_num = len(parking_lots.id)
        while i < ped_num:
            passenger_id = 'c{}'.format(i + n)
            ped_pickup_lat = random.uniform(lat_min, lat_max)
            ped_pickup_lon = random.uniform(lon_min, lon_max)
            ped_dropoff_lat = random.uniform(lat_min, lat_max)
            ped_dropoff_lon = random.uniform(lon_min, lon_max)
            nearest_pickup_node = ox.distance.nearest_nodes(network.graph, ped_pickup_lon, ped_pickup_lat)
            nearest_dropoff_node = ox.distance.nearest_nodes(network.graph, ped_dropoff_lon, ped_dropoff_lat)
            if not nx.has_path(network.graph, nearest_pickup_node, nearest_dropoff_node):
                continue
            j = 0
            while j < pl_num:
                if nx.has_path(network.graph, nearest_dropoff_node, parking_lots.node_id[j]):
                    break
                j += 1
            if j == pl_num:
                continue
            orig_lat = network.graph.nodes[nearest_pickup_node]['y']
            orig_lon = network.graph.nodes[nearest_pickup_node]['x']
            pickup_tw_start = current_time * dt
            max_wait_time = minutes_to_seconds(8)
            pickup_tw_end = pickup_tw_start + max_wait_time
            dest_lat = network.graph.nodes[nearest_dropoff_node]['y']
            dest_lon = network.graph.nodes[nearest_dropoff_node]['x']
            pickup_dropoff_dura = minutes_to_seconds(0)

            self.owner.agent_group.create_next_pedestrians(passenger_id, orig_lat, orig_lon, nearest_pickup_node,
                                                           pickup_tw_start, pickup_tw_end, max_wait_time, dest_lat,
                                                           dest_lon, nearest_dropoff_node, pickup_dropoff_dura, True)
            i += 1
