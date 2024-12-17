import numpy as np

from core.tasks import AgentTask, TaskContainer
from core.utils.objectopers import index_optional
from core.tracer import tr
from core.utils.units import seconds_to_hours, m_to_km
from .states import VehicleStates
from ..pedestrians.states import PedestrianStates
from ..parking_lot.states import ParkinglotStates


class VehicleTask(AgentTask):
    """
    tasks define specific actions for the vehicles to take.
    """
    mute_debug_message = {"start": False, "step": False, "end": False}

    def register_properties(self):
        self.customer_ids = self.declare_property("customer_ids", int, 99999)
        self.parkinglot_ids = self.declare_property("parkinglot_ids", int, 99999)

    def start_message(self, ids):
        pass

    def step_message(self, ids):
        pass

    def end_message(self, ids):
        pass

    def start(self, ids=None):
        if self.debug_mode() and not self.mute_debug_message["start"] and len(ids) > 0:
            self.start_message(ids)

    def step(self, ids=None):
        if self.debug_mode() and not self.mute_debug_message["step"] and len(ids) > 0:
            self.step_message(ids)

    @index_optional
    def end(self, ids=None):
        if self.debug_mode() and not self.mute_debug_message["end"] and len(ids) > 0:
            self.end_message(ids)

        # change vehicle states given the number of remaining tasks
        idx = self.get_index(ids)
        self.agent_group.current_tasks[idx] = None
        finished = self.agent_group.num_remaining_tasks[idx] <= 0
        unfinish = self.agent_group.num_remaining_tasks[idx] > 0
        # if len(ids[finished]) > 0:
        #     print(f"*********************** Vehicle {ids[finished]} finish its route *****************")
        self.change_vehicle_states(ids[finished], VehicleStates.Idle)
        self.agent_group.task_container.pop_and_start_next_task(ids[unfinish])

    @property
    def clock(self):
        return self.agent_group.world.clock

    @property
    def parkinglots(self):
        return self.controller.parkinglot_controller.agent_group

    @property
    def pedestrians(self):
        return self.controller.pedestrian_controller.agent_group

    @property
    def pedestrian_states(self):
        return self.pedestrians.current_state

    @property
    def change_pedestrian_states(self):
        return self.pedestrians.state_container.change_agent_state

    @property
    def change_vehicle_states(self):
        return self.agent_group.state_container.change_agent_state

    def add_vehicles_to_parkinglot(self, ids):
        agent_idx = self.get_index(ids)
        parkinglot_ids = self.parkinglot_ids[agent_idx]
        self.parkinglots.add_vehicle(ids, parkinglot_ids)

    def remove_vehicles_from_parkinglot(self, idx):
        agent_idx = self.get_index(idx)
        for agent_id in agent_idx:
            if self.agent_group.current_node[agent_id] in self.parkinglots.node_id:
                parkinglot_idx = self.parkinglots.index_by_node_id(self.agent_group.current_node[agent_id])
                self.parkinglots.remove_vehicle([agent_id], parkinglot_idx)

    def set_pedestrian_state(self, ids, state):
        agent_idx = self.get_index(ids)
        customer_ids = self.customer_ids[agent_idx]
        self.change_pedestrian_states(customer_ids, state)


class TravelBaseTask(VehicleTask):
    arrive_criterion = 0.00005  # kilometers

    def register_properties(self):
        super().register_properties()
        self.destination_lat = self.declare_property("destination_lat", float, np.nan)
        self.destination_long = self.declare_property("destination_long", float, np.nan)
        self.destination_node = self.declare_property("destination_node", object, np.nan)

        # used to record moving progress and reconstruct exact coordinates
        self.distance_to_destination = self.declare_property("distance_to_destination", float, np.inf)
        self.distance_traveled = self.declare_property("distance_traveled", float, 0)

    def reset(self, ids):
        idx = self.get_index(ids)
        self.distance_to_destination[idx] = 99999
        self.distance_traveled[idx] = 0
        self.customer_ids[idx] = 99999

    @index_optional
    def start(self, ids):
        super().start(ids)
        # get the appropriate index for each agent based on their position in the agent group
        # agent index in agent group EQUALS agent index in task container
        self.change_vehicle_states(ids, VehicleStates.Moving)
        self.initialize_path(ids)

    def initialize_path(self, ids):
        """(re)-compute the shortest path from current location to destination, distance to destination and distance
        traveled"""
        idx = self.get_index(ids)
        network = self.agent_group.controller.network_controller.network
        vehicles = self.agent_group
        for v in idx:
            vehicles.path[v] = network.shortest_path(vehicles.current_node[v], self.destination_node[v])
            cost = m_to_km(network.shortest_path_cost(vehicles.current_node[v], self.destination_node[v]))
            self.distance_to_destination[v] = cost
            self.distance_traveled[v] = 0

        # used for debug
        self.agent_group.controller.distance_traveled = self.distance_traveled
        self.agent_group.controller.distance_to_destination = self.distance_to_destination

    def execute_move(self, idx):
        """impose vehicles one time-step movement toward destination"""
        dt = self.clock.dt
        self.distance_traveled[idx] += self.agent_group.spd_kmh[idx] * seconds_to_hours(dt)

        # used for debug
        self.agent_group.controller.distance_traveled = self.distance_traveled
        self.agent_group.controller.distance_to_destination = self.distance_to_destination

    def update_location_parameters(self, idx, f=True):
        """update vehicle location information: coordinates, current node, current edge, head degree"""
        from core.utils.geoopers import bearing_wsg84_dg, location_wsg84_given_kms

        network = self.agent_group.controller.network_controller.network
        vehicles = self.agent_group

        for v in idx:
            pre_node = None
            next_node = None
            current_node = None
            current_edge = None
            current_edge_cost = None

            # find the edge or node which currently vehicle is on
            distance_traveled = self.distance_traveled[v]
            if f:
                path = vehicles.path[v]
            else:
                if len(vehicles.path[v]) == 1:
                    vehicles.next_node[v] = vehicles.path[0]
                    continue
                path = vehicles.path[v][:vehicles.path[v].index(vehicles.next_node[v])]
            # No need to update if two orders are mapped to the same node of the map
            if len(path) <= 1:
                continue
            accu_edge_distance = 0
            exceed = True
            for n1, n2 in zip(path[:-1], path[1:]):
                edge_id = (n1, n2, 0)
                edge_cost = m_to_km(network.get_edge_cost(edge_id))
                accu_edge_distance += edge_cost
                if distance_traveled < accu_edge_distance:
                    pre_node, next_node = n1, n2
                    if distance_traveled == accu_edge_distance:
                        current_node = n2
                    else:
                        current_edge = edge_id
                        current_edge_cost = edge_cost
                    exceed = False
                    break
            if exceed:
                pre_node = path[-2]
                next_node = path[-1]
                current_node = path[-1]

            # update pre_node and next_node
            vehicles.pre_node[v] = pre_node
            vehicles.next_node[v] = next_node

            # update vehicle location id - either on a node or an edge
            vehicles.current_node[v] = current_node
            vehicles.current_edge[v] = current_edge

            # update vehicle head degree
            pre_node_long, pre_node_lat = network.get_location(pre_node)
            next_node_long, next_node_lat = network.get_location(next_node)
            bearing = bearing_wsg84_dg(pre_node_lat, pre_node_long, next_node_lat, next_node_long)
            vehicles.hdg_deg[v] = bearing

            # update vehicle current coordinates
            if current_node is not None:
                long, lat = network.get_location(current_node)
            else:
                distance_traveled_on_edge = distance_traveled - (accu_edge_distance - current_edge_cost)
                lat, long = location_wsg84_given_kms(pre_node_lat, pre_node_long, distance_traveled_on_edge, bearing)
            vehicles.lon_deg[v], vehicles.lat_deg[v] = long, lat
        
    @index_optional
    def step(self, ids):
        super().step(ids)
        idx = self.get_index(ids)
        self.execute_move(idx)
        self.update_location_parameters(idx)
        self.rest_distance_to_destination = self.distance_to_destination - self.distance_traveled
        finished = np.where(self.rest_distance_to_destination <= self.arrive_criterion)[0]
        self.signal_end(self.get_agent_id_by_index(finished))

    @index_optional
    def end(self, ids):
        super().end(ids)
        idx = self.get_index(ids)
        self.agent_group.current_node[idx] = self.destination_node[idx]
        # forcefully update vehicle location to the destination
        network = self.agent_group.controller.network_controller.network
        for v in idx:
            long, lat = network.get_location(self.agent_group.current_node[v])
            self.agent_group.lon_deg[v], self.agent_group.lat_deg[v] = long, lat
        self.agent_group.total_distance_traveled[idx] += self.distance_to_destination[idx]


class StopBaseTask(VehicleTask):

    def register_properties(self):
        super().register_properties()
        self.duration = self.declare_property("duration", int, 99999)

    def reset(self, ids):
        idx = self.get_index(ids)
        self.duration[idx] = 99999
        self.customer_ids[idx] = 99999

    @index_optional
    def step(self, ids):
        super().step(ids)
        dt = self.clock.dt
        finished = np.where(self.agent_group.time_in_task > (self.duration/dt))[0]
        # make sure that finished tasks are the ones where agents are involved
        self.signal_end(self.get_agent_id_by_index(finished))


class VehicleTasks(TaskContainer):
    class TravelToParkingLot(TravelBaseTask):
        name = "traveltoparkinglot"

        def start_message(self, ids):
            tr(f"Vehicle {ids}: Travelling to parking lot")

        # TODO: If this task is interrupted, the end message should be different(how to code?)
        def end_message(self, ids):
            tr(f"Vehicle {ids}: Arrived to parking lot")

        @index_optional
        def step(self, ids):
            idx = self.get_index(ids)
            # If an AET that driving towards the parking lot was assigned a new request,
            # its task queue will have 5 tasks (Charging/TravelToPickup/Pickup/TravelToDropoff/Dropoff)
            have_new_task = np.array([i for i, lst in enumerate(self.agent_group.task_queue) if len(lst) == 5])
            terminate = np.array([i for i in idx if (i in have_new_task)])
            if len(terminate) > 0:
                self.update_task(terminate)
                self.execute_move(terminate)
                self.update_location_parameters(terminate, False)
                remain_idx = idx[~np.isin(idx, terminate)]
            else:
                remain_idx = idx
            self.execute_move(remain_idx)
            self.update_location_parameters(remain_idx)
            self.rest_distance_to_destination = self.distance_to_destination - self.distance_traveled
            finished = np.where(self.rest_distance_to_destination <= self.arrive_criterion)[0]
            self.signal_end(self.get_agent_id_by_index(finished))

        @index_optional
        def end(self, ids):
            if self.debug_mode() and not self.mute_debug_message["end"] and len(ids) > 0:
                self.end_message(ids)
            idx = self.get_index(ids)
            self.agent_group.current_node[idx] = self.destination_node[idx]
            network = self.agent_group.controller.network_controller.network
            for v in idx:
                long, lat = network.get_location(self.agent_group.current_node[v])
                self.agent_group.lon_deg[v], self.agent_group.lat_deg[v] = long, lat
            self.agent_group.total_distance_traveled[idx] += self.distance_to_destination[idx]
            for v in idx:
                try:
                    self.parkinglot_ids[v] = self.parkinglots.index_by_node_id(self.agent_group.current_node[v])
                except:
                    # tr(f"Vehicle {ids}: Interrupted")
                    pass
            for veh_id in ids:
                if (self.parkinglot_ids[veh_id] in self.parkinglots.id and
                        self.parkinglots.current_state[self.parkinglot_ids[veh_id]] == ParkinglotStates.Occupied):
                    task = self.agent_group.task_container.generate_task(
                        VehicleTasks.Queuing,
                        veh_id,
                    )
                    # Insert the queuing task into the vehicle
                    self.agent_group.task_queue[veh_id].insert(0, task)
                    self.agent_group.num_remaining_tasks[veh_id] += 1
                elif self.parkinglot_ids[veh_id] in self.parkinglots.id:
                    self.add_vehicles_to_parkinglot(np.array([veh_id]))
            # change vehicle states given the number of remaining tasks
            self.agent_group.current_tasks[idx] = None
            finished = self.agent_group.num_remaining_tasks[idx] <= 0
            unfinish = self.agent_group.num_remaining_tasks[idx] > 0
            self.change_vehicle_states(ids[finished], VehicleStates.Idle)
            self.agent_group.task_container.pop_and_start_next_task(ids[unfinish])

        def update_task(self, idx):
            network = self.agent_group.controller.network_controller.network
            vehicles = self.agent_group
            for v in idx:
                self.destination_node[v] = vehicles.next_node[v]
                cost = m_to_km(network.shortest_path_cost(vehicles.path[v][0], self.destination_node[v]))
                self.distance_to_destination[v] = cost
                self.agent_group.task_queue[v] = self.agent_group.task_queue[v][1:]
                self.agent_group.num_remaining_tasks[v] = 4

    class Queuing(StopBaseTask):
        name = "queuing"

        def start_message(self, ids):
            tr(f"Vehicle {ids}: queuing begins")

        def step_message(self, ids):
            tr(f"Vehicle {ids}: queuing")

        def end_message(self, ids):
            tr(f"Vehicle {ids}: queuing ends")

        @index_optional
        def start(self, ids):
            super().start(ids)
            self.change_vehicle_states(ids, VehicleStates.Queuing)

        @index_optional
        def step(self, ids):
            if self.debug_mode() and not self.mute_debug_message["step"] and len(ids) > 0:
                self.step_message(ids)
            idx = self.get_index(ids)
            self.agent_group.controller.queuing_time[idx] += self.clock.dt
            have_task = np.array([i for i, lst in enumerate(self.agent_group.task_queue) if len(lst) > 1])
            interrupted = [i for i in idx if (i in have_task)]
            self.parkinglot_ids[ids] = self.parkinglots.index_by_node_id(self.agent_group.current_node[ids])
            end_ids = []
            for veh_id in ids:
                if self.get_index(veh_id)[0] in interrupted:
                    continue
                if self.parkinglots.current_state[self.parkinglot_ids[self.get_index(veh_id)[0]]] == ParkinglotStates.Available:
                    end_ids.append(veh_id)
                    self.add_vehicles_to_parkinglot([veh_id])
            finished = list(set(interrupted + end_ids))
            self.signal_end(finished)

    class Charging(StopBaseTask):
        name = "charging"

        def start_message(self, ids):
            tr(f"Vehicle {ids}: charging begins")

        def step_message(self, ids):
            tr(f"Vehicle {ids}: charging")

        def end_message(self, ids):
            tr(f"Vehicle {ids}: charging ends")

        @index_optional
        def start(self, ids):
            super().start(ids)
            self.change_vehicle_states(ids, VehicleStates.Charging)

        @index_optional
        def step(self, ids):
            super().step(ids)
            # If the charging AET is assigned a new task, it will stop charging
            # Only AET may be assigned a new task while it is charging
            idx = self.get_index(ids)
            self.agent_group.controller.charging_time[idx] += self.clock.dt
            have_task = np.array([i for i, lst in enumerate(self.agent_group.task_queue) if lst])
            finished = np.array([i for i in idx if (i in have_task)])
            if len(finished) > 0:
                self.signal_end(self.get_agent_id_by_index(finished))

        @index_optional
        def end(self, ids):
            super().end(ids)
            self.remove_vehicles_from_parkinglot(ids)

    class TravelToPickUp(TravelBaseTask):
        name = "traveltopickup"

        def start_message(self, ids):
            tr(f"Vehicle {ids}: Travelling to pickup point")

        def end_message(self, ids):
            tr(f"Vehicle {ids}: Arrived to the pickup point")

        @index_optional
        def start(self, ids):
            super().start(ids)
            self.set_pedestrian_state(ids, PedestrianStates.Waiting)

        @index_optional
        def step(self, ids):
            super().step(ids)
            idx = self.get_index(ids)
            self.agent_group.controller.inserve_time[idx] += self.clock.dt

    class TravelToDropOff(TravelBaseTask):
        name = "traveltodropoff"

        def start_message(self, ids):
            tr(f"Vehicle {ids}: Travelling to dropoff point")

        def end_message(self, ids):
            tr(f"Vehicle {ids}: Arrived to dropoff point")

        def start(self, ids):
            super().start(ids)
            self.set_pedestrian_state(ids, PedestrianStates.ToDropoff)

        @index_optional
        def step(self, ids):
            super().step(ids)
            idx = self.get_index(ids)
            self.agent_group.controller.inserve_time[idx] += self.clock.dt

    class PickupCustomer(StopBaseTask):
        name = "pickup"

        def start_message(self, ids):
            tr(f"Vehicle {ids}: Pickup starts")

        def end_message(self, ids):
            tr(f"Vehicle {ids}: Pickup ends")

        @index_optional
        def start(self, ids):
            super().start(ids)
            self.change_vehicle_states(ids, VehicleStates.Waiting)
            self.set_pedestrian_state(ids, PedestrianStates.InService)
            self.update_wait_time(ids)

        @index_optional
        def step(self, ids):
            super().step(ids)
            idx = self.get_index(ids)
            self.agent_group.controller.inserve_time[idx] += self.clock.dt

        def update_wait_time(self, ids):
            agent_idx = self.get_index(ids)
            customer_ids = self.customer_ids[agent_idx]
            elapsed_seconds = self.clock.elapsed_ticks * self.clock.dt
            for id in customer_ids:
                idx = self.pedestrians.index_by_id(id)
                self.pedestrians.controller.wait_time[id] = elapsed_seconds - self.pedestrians.pickup_tw_start[idx]

    class DropoffCustomer(StopBaseTask):
        name = "dropoff"

        def start_message(self, ids):
            tr(f"Vehicle {ids}: Dropoff starts")

        def end_message(self, ids):
            tr(f"Vehicle {ids}: Dropoff ends")

        @index_optional
        def start(self, ids):
            super().start(ids)
            self.change_vehicle_states(ids, VehicleStates.Waiting)
            self.set_pedestrian_state(ids, PedestrianStates.InService)

        @index_optional
        def step(self, ids):
            super().step(ids)
            idx = self.get_index(ids)
            self.agent_group.controller.inserve_time[idx] += self.clock.dt

        @index_optional
        def end(self, ids):
            idx = self.get_index(ids)
            self.set_pedestrian_state(ids, PedestrianStates.Served)
            super().end(ids)

    class OfflineToParkingLot(TravelBaseTask):
        name = "offline to parking lot"

        def start_message(self, ids):
            tr(f"Vehicle {ids}: Offline and travel to parking lot")

        def end_message(self, ids):
            tr(f"Vehicle {ids}: Offline and arrived to parking lot")

        @index_optional
        def end(self, ids):
            if self.debug_mode() and not self.mute_debug_message["end"] and len(ids) > 0:
                self.end_message(ids)
            idx = self.get_index(ids)
            self.agent_group.current_node[idx] = self.destination_node[idx]
            network = self.agent_group.controller.network_controller.network
            for v in idx:
                long, lat = network.get_location(self.agent_group.current_node[v])
                self.agent_group.lon_deg[v], self.agent_group.lat_deg[v] = long, lat
            self.agent_group.total_distance_traveled[idx] += self.distance_to_destination[idx]
            self.parkinglot_ids[ids] = self.parkinglots.index_by_node_id(self.agent_group.current_node[ids])
            for veh_id in ids:
                if self.parkinglots.current_state[self.parkinglot_ids[self.get_index(veh_id)[0]]] == ParkinglotStates.Occupied:
                    task = self.agent_group.task_container.generate_task(
                        VehicleTasks.Queuing,
                        veh_id,
                    )
                    # Insert the task of queuing for charging into the vehicle
                    self.agent_group.task_queue[self.get_index(veh_id)[0]].insert(0, task)
                    self.agent_group.num_remaining_tasks[self.get_index(veh_id)[0]] += 1
                else:
                    self.add_vehicles_to_parkinglot(np.array([veh_id]))
            self.agent_group.current_tasks[idx] = None
            finished = self.agent_group.num_remaining_tasks[idx] <= 0
            unfinish = self.agent_group.num_remaining_tasks[idx] > 0
            self.change_vehicle_states(ids[finished], VehicleStates.Idle)
            self.agent_group.task_container.pop_and_start_next_task(ids[unfinish])

    class OfflineCharging(StopBaseTask):
        name = "offline charging"

        def start_message(self, ids):
            tr(f"Vehicle {ids}: Offline and charging begins")

        def step_message(self, ids):
            tr(f"Vehicle {ids}: Offline and charging")
            pass

        def end_message(self, ids):
            tr(f"Vehicle {ids}: Offline and charging ends")

        @index_optional
        def start(self, ids):
            super().start(ids)
            self.change_vehicle_states(ids, VehicleStates.Charging)

        @index_optional
        def step(self, ids):
            super().step(ids)
            idx = self.get_index(ids)
            self.agent_group.controller.charging_time[idx] += self.clock.dt

        @index_optional
        def end(self, ids):
            # If the vehicle's task queue is empty, it doesn't need to rest and will online
            super().end(ids)
            idxs = self.agent_group.index_by_id(ids)
            reonline_idx = [idx for idx in idxs if len(self.agent_group.task_queue[idx]) == 0]
            self.agent_group.is_online[reonline_idx] = True
            self.remove_vehicles_from_parkinglot(ids)
