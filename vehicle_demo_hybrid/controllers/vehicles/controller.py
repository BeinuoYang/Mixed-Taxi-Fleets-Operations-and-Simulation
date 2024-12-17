from core.controller import Controller, ControllerSettings, Generator
from core.tracer import tr
from core.utils.units import seconds_to_hours
from ..network.controller import NetworkController
from ..parking_lot.controller import ParkingLotController
from ..pedestrians.controller import PedestrianController
from .agents import VehicleAgents
from .logic import VehicleLogic
from .states import VehicleStates
from .assignment import Assignment

import numpy as np
import random
import csv


class VehicleController(Controller):
    def init_controller(self):
        self.assignment: Assignment = Assignment()
        self.distance_traveled = None
        self.distance_to_destination = None
        self.charging_time = None
        self.queuing_time = None
        self.assigned_peds_num = None
        self.online_time = None
        self.profits = 0
        self.inserve_time = None

    def register_implementation(self):
        # Initialisation of the settings object
        self.settings: VehicleControllerSettings = VehicleControllerSettings()

        # Declaration of agent group (which will hold all the vehicles)
        self.agent_group: VehicleAgents = VehicleAgents("vehicles", self)

        self.agent_logic: VehicleLogic = VehicleLogic(self)

        self.agent_generator: VehicleGenerator = VehicleGenerator(self)

        self.pedestrian_controller: PedestrianController = self.register_dependency(PedestrianController)
        self.network_controller: NetworkController = self.register_dependency(NetworkController)
        self.parkinglot_controller: ParkingLotController = self.register_dependency(ParkingLotController)

    @property
    def clock(self):
        return self.world.clock

    def start_controller(self):
        pass

    def start_controller_post(self):
        parkinglot_num = self.params["parkinglot"].get("num")
        vehicle_num_per_type = ''.join(
            [str(int(x * 10)).zfill(2) for x in self.params["vehicles"].get("percent_per_type")])
        file_path = "/".join([self.params["output_dir"], "{}-{}-vehicles.csv".format(parkinglot_num, vehicle_num_per_type)])
        header = ["iter", "total_charging_time", "avg_charging_time", "total_queuing_time",
                  "total_distance_traveled_AET", "total_distance_traveled_HGT", "total_distance_traveled_HET",
                  "operational_profits", "avg_online_time_HGT", "avg_online_time_HET", "avg_percentage_time_inservice",
                  "assigned_peds_per_horizon", "served_peds_num_AET", "served_peds_num_HET", "served_peds_num_HGT", ]
        AET_idx = len(self.agent_group.id[(self.agent_group.is_electric > 0) & (self.agent_group.is_autonomous > 0)])
        HET_idx = len(self.agent_group.id[(self.agent_group.is_electric > 0) & (self.agent_group.is_autonomous == 0)])
        HGT_idx = len(self.agent_group.id[(self.agent_group.is_electric == 0) & (self.agent_group.is_autonomous == 0)])
        EV_num = len(self.agent_group.is_electric[self.agent_group.is_electric > 0])
        total_charging_time = np.sum(self.charging_time) if (self.charging_time is not None) else 0
        avg_charging_time = total_charging_time / EV_num if EV_num > 0 else 0
        total_queuing_time = np.sum(self.queuing_time) if (self.queuing_time is not None) else 0
        assigned_peds_num = np.sum(self.assigned_peds_num)
        record = [self.clock.elapsed_ticks, total_charging_time, avg_charging_time, total_queuing_time,
                  0, 0, 0, 0, 0, 0, 0, assigned_peds_num, 0, 0, 0]
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"dt={self.clock.dt}", f"veh_num={len(self.agent_group.id)}",
                             f"AET_num={AET_idx}", f"HET_num={HET_idx}", f"HGT_num={HGT_idx}"])
            writer.writerow(header)
            writer.writerow(record)

    def step_controller(self):
        self.assigned_peds_num = np.zeros(len(self.assigned_peds_num))
        dt = self.clock.dt
        elapsed_ticks = self   .clock.elapsed_ticks
        seconds = elapsed_ticks * dt
        horizon = self.params["horizon"]
        # Match the order to the vehicle at every time horizon
        if seconds > 0 and seconds % horizon == 0:
            if_optimization = self.params["optimization"]
            if if_optimization:
                self.assignment.step_assignment(self.pedestrian_controller.agent_group, self.agent_group,
                                                self.parkinglot_controller.agent_group, self.network_controller.network,
                                                seconds, )
            else:
                # no optimization, just assign the peds to the nearest vehicles
                self.assignment.step_assignment_simple(self.pedestrian_controller.agent_group, self.agent_group,
                                                       self.parkinglot_controller.agent_group,
                                                       self.network_controller.network, seconds,)
        self.update_operation_time()

        # output evaluation metrics
        elapsed_seconds = self.clock.dt * self.clock.elapsed_ticks
        if elapsed_seconds > 0 and elapsed_seconds % horizon == 0 and (self.clock.elapsed_ticks < self.clock.numticks):
            parkinglot_num = self.params["parkinglot"].get("num")
            vehicle_num_per_type = ''.join(
                [str(int(x * 10)).zfill(2) for x in self.params["vehicles"].get("percent_per_type")])
            file_path = "/".join([self.params["output_dir"], "{}-{}-vehicles.csv".format(parkinglot_num, vehicle_num_per_type)])
            # file_path = "/".join([self.params["output_dir"], "vehicles{}.csv".format(self.clock.real_starttime)])
            AET_idx = np.where((self.agent_group.is_electric > 0) & (self.agent_group.is_autonomous > 0))[0]
            HET_idx = np.where((self.agent_group.is_electric > 0) & (self.agent_group.is_autonomous == 0))[0]
            HGT_idx = np.where((self.agent_group.is_electric == 0) & (self.agent_group.is_autonomous == 0))[0]
            total_distance_traveled_AET = np.sum(self.agent_group.total_distance_traveled[AET_idx])
            total_distance_traveled_HET = np.sum(self.agent_group.total_distance_traveled[HET_idx])
            total_distance_traveled_HGT = np.sum(self.agent_group.total_distance_traveled[HGT_idx])

            EV_num = len(AET_idx) + len(HET_idx)
            total_charging_time = np.sum(self.charging_time) if (self.charging_time is not None) else 0
            avg_charging_time = total_charging_time / EV_num if EV_num else 0
            total_queuing_time = np.sum(self.queuing_time) if (self.queuing_time is not None) else 0
            assigned_peds_num_AET = np.sum(self.assigned_peds_num[AET_idx])
            assigned_peds_num_HET = np.sum(self.assigned_peds_num[HET_idx])
            assigned_peds_num_HGT = np.sum(self.assigned_peds_num[HGT_idx])
            avg_online_time_HGT = np.mean(self.online_time[HGT_idx])
            avg_online_time_HET = np.mean(self.online_time[HET_idx])
            worked_vehicles_idx = np.where(self.agent_group.cont_op_time > 0)[0]
            avg_percentage_time_inservice = np.mean(self.inserve_time[worked_vehicles_idx] /
                                                    self.agent_group.cont_op_time[worked_vehicles_idx])
            assigned_peds_num = np.sum(self.assigned_peds_num)
            record = [self.clock.elapsed_ticks, total_charging_time, avg_charging_time, total_queuing_time,
                      total_distance_traveled_AET, total_distance_traveled_HGT, total_distance_traveled_HET,
                      self.profits, avg_online_time_HGT, avg_online_time_HET, avg_percentage_time_inservice,
                      assigned_peds_num, assigned_peds_num_AET, assigned_peds_num_HET, assigned_peds_num_HGT]
            with open(file_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(record)

    def update_operation_time(self):
        dt = self.clock.dt
        operation_veh_idx = [idx for idx, value in enumerate(self.agent_group.is_online) if value]
        self.agent_group.cont_op_time[operation_veh_idx] += dt
        self.online_time[operation_veh_idx] += dt


class VehicleControllerSettings(ControllerSettings):
    def init_settings(self):
        pass


class VehicleGenerator(Generator):

    @property
    def clock(self):
        return self.owner.world.clock

    def start(self):
        self.generate_vehicles_on_network()

    def step(self):
        self.change_vehicles_isonline()

    def generate_vehicles_on_network(self):
        random.seed(5)
        vehicle_num = self.owner.params["vehicles"].get("num")
        # 0:AET 1:HET 2:HGT
        num_per_type = [int(v*vehicle_num) for v in self.owner.params["vehicles"].get("percent_per_type")]
        # The proportion of HVs online during the four time periods: 0-6, 6-12, 12-18, and 18-24.
        HV_online_percent = self.owner.params["vehicles"].get("HV_online_percent")
        network = self.owner.network_controller.network
        vehicle_node_id = [random.choice(list(network.graph.nodes())) for _ in range(vehicle_num)]

        vehicle_lats = np.array([network.get_location(node_id)[1] for node_id in vehicle_node_id])
        vehicle_longs = np.array([network.get_location(node_id)[0] for node_id in vehicle_node_id])
        current_node = np.array(vehicle_node_id)

        self.agents.create_agents(
            np.array(range(vehicle_num)),
            "sid", np.array(range(vehicle_num)),
            "is_autonomous", np.array([True for _ in range(num_per_type[0])] +
                                      [False for _ in range(num_per_type[1] + num_per_type[2])]),
            "is_electric", np.array([True for _ in range(num_per_type[0])] + [True for _ in range(num_per_type[1])] +
                                    [False for _ in range(num_per_type[2])]),
            "lat_deg", vehicle_lats,
            "lon_deg", vehicle_longs,
            "spd_kmh", 22.5,
            "order_sequence", [[] for _ in range(vehicle_num)],
            "current_state", VehicleStates.Idle,
            "battery_capacity", 60,
            "battery_level", np.random.uniform(10, 50, vehicle_num),
            "charge_rate", 50,
            "energy_consumption", 0.15,
            "range_anxiety", np.clip(np.rint(np.random.normal(7, 3, vehicle_num)).astype(int), 1, 10),
            "cont_op_time", 0,
            "max_cont_op_time", 6,
            "min_offline_time", 1 / 3,
            "is_online", np.array([True for _ in range(num_per_type[0] + int(num_per_type[1] * HV_online_percent[0]))] +
                                  [False for _ in range(int(num_per_type[1] * (1 - HV_online_percent[0])))] +
                                  [True for _ in range(int(num_per_type[2] * HV_online_percent[0]))] +
                                  [False for _ in range(int(num_per_type[2] * (1 - HV_online_percent[0])))]),
            "current_node", current_node,
        )

        start1 = num_per_type[0] + int(num_per_type[1] * HV_online_percent[0])
        end1 = sum(num_per_type[:2])
        start2 = sum(num_per_type[:2]) + int(num_per_type[2] * HV_online_percent[0])
        end2 = sum(num_per_type)
        offline_idx = np.concatenate((np.arange(start1, end1), np.arange(start2, end2)))
        offline_id = self.owner.agent_group.id[offline_idx]
        self.owner.agent_group.state_container.change_agent_state(offline_id, VehicleStates.Finished)

        self.owner.charging_time = np.zeros(vehicle_num)
        self.owner.queuing_time = np.zeros(vehicle_num)
        self.owner.assigned_peds_num = np.zeros(vehicle_num)
        self.owner.online_time = np.zeros(vehicle_num)
        self.owner.inserve_time = np.zeros(vehicle_num)

        if self.owner.world.model.sim.settings.debug_mode:
            tr(f"Load vehicles: {self.agents.sid}")

    def change_vehicles_isonline(self):
        vehicle_num = self.owner.params["vehicles"].get("num")
        num_per_type = [int(v * vehicle_num) for v in self.owner.params["vehicles"].get("percent_per_type")]
        HV_online_percent = self.owner.params["vehicles"].get("HV_online_percent")
        current_time = seconds_to_hours(self.clock.elapsed_ticks * self.clock.dt)
        if current_time % 6 == 0:
            p_idx = int(current_time // 6)
        else:
            return
        start1 = num_per_type[0] + int(num_per_type[1] * sum(HV_online_percent[:p_idx]))
        end1 = num_per_type[0] + int(num_per_type[1] * sum(HV_online_percent[:p_idx + 1]))
        start2 = sum(num_per_type[:2]) + int(num_per_type[2] * sum(HV_online_percent[:p_idx]))
        end2 = sum(num_per_type[:2]) + int(num_per_type[2] * sum(HV_online_percent[:p_idx + 1]))
        online_idx = np.concatenate((np.arange(start1, end1), np.arange(start2, end2)))
        online_id = self.owner.agent_group.id[online_idx]
        self.owner.agent_group.is_online[online_idx] = True
        self.owner.agent_group.state_container.change_agent_state(online_id, VehicleStates.Idle)
