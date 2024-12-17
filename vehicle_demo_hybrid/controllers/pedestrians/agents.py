from core.agent_group import AgentGroup
from ..pedestrians.states import PedestrianStates

import numpy as np


class PedestrianAgents(AgentGroup):
    """
    Pedestrian agent - created when a vehicle enters the "move" task and is deleted on-end of the same task.
    Does nothing otherwise.
    """

    def init_agents_custom(self):
        self.orig_lat_deg = self.declare_property("orig_lat_deg", float)
        self.orig_lon_deg = self.declare_property("orig_lon_deg", float)
        self.pickup_node_id = self.declare_property("pickup_node_id", float)
        self.pickup_tw_start = self.declare_property("pickup_tw_start", float)
        self.pickup_tw_end = self.declare_property("pickup_tw_end", float)
        self.max_wait_time = self.declare_property("max_wait_time", float)

        self.dest_lat_deg = self.declare_property("dest_lat_deg", float)
        self.dest_lon_deg = self.declare_property("dest_lon_deg", float)
        self.dropoff_node_id = self.declare_property("dropoff_node_id", float)
        self.dropoff_tw_start = self.declare_property("dropoff_tw_start", float)
        self.dropoff_tw_end = self.declare_property("dropoff_tw_end", float)

        self.dropoff_duration = self.declare_property("dropoff_duration", float)
        self.pickup_duration = self.declare_property("pickup_duration", float)

        self.vehicle_id = self.declare_property("vehicle_id", float)
        self.passenger_id = self.declare_property("passenger_id")  # passenger ID based on datapoint
        self.is_pickup = self.declare_property("is_pickup", bool)  # True if pickup order, False if dropoff

        # contains information on the pedestrian states in the state container
        self.state_container = PedestrianStates()

    def create_pedestrian(self, agent_ids, passenger_id, orig_lat, orig_lon, pickup_node_id, pickup_tw_start,
                          pickup_tw_end, max_wait_time, dest_lat, dest_lon, dropoff_node_id, pd_time, is_pickup):
        """
        creates a set of pedestrians based on the input type
        """
        self.create_agents(
            agent_ids,
            "passenger_id", passenger_id,
            "orig_lat_deg", orig_lat,
            "orig_lon_deg", orig_lon,
            "pickup_node_id", pickup_node_id,
            "pickup_tw_start", pickup_tw_start,
            "pickup_tw_end", pickup_tw_end,
            "max_wait_time", max_wait_time,
            "dest_lat_deg", dest_lat,
            "dest_lon_deg", dest_lon,
            "dropoff_node_id", dropoff_node_id,
            # "dropoff_tw_start", dropoff_tw_start,
            # "dropoff_tw_end", dropoff_tw_end,
            "pickup_duration", pd_time,
            "dropoff_duration", pd_time,
            "current_state", PedestrianStates.Unassigned,
            "is_pickup", is_pickup
        )

    def create_next_pedestrians(self, passenger_id, orig_lats, orig_lons, pickup_node_ids, pickup_tw_starts,
                                pickup_tw_ends, max_wait_time, dest_lats, dest_lons, dropoff_node_ids,
                                pickup_dropoff, is_pickup):
        """
        as above but agent ids are created automatically based on the number of created agents in sim
        """
        n = 1 if isinstance(orig_lats, float) else len(orig_lats)
        if n == 1:
            agent_ids = self.agents_created
        else:
            agent_ids = [self.agents_created + i for i in range(n)]
        return self.create_pedestrian(agent_ids, passenger_id, orig_lats, orig_lons, pickup_node_ids, pickup_tw_starts,
                                      pickup_tw_ends, max_wait_time, dest_lats, dest_lons, dropoff_node_ids,
                                      pickup_dropoff, is_pickup)

    @property
    def clock(self):
        return self.world.clock

    def destroy_served_pedestrians(self):
        idxs = self.index_by_state(PedestrianStates.Served)
        ids = self.id_by_index(idxs)
        self.controller.served_peds = np.concatenate((self.controller.served_peds, ids))
        self.destroy_agents(ids)

    def destroy_unassigned_pedestrians(self):
        current_time = self.clock.elapsed_ticks
        dt = self.clock.dt
        idxs = self.index_by_state(PedestrianStates.Unassigned)
        overtime = self.pickup_tw_end[idxs] < current_time*dt
        destroy_idxs = idxs[overtime]
        ids = self.id_by_index(destroy_idxs)
        self.controller.destroyed_peds = np.concatenate((self.controller.destroyed_peds, ids))
        # self.update_wait_time(ids)
        self.destroy_agents(ids)

    def update_wait_time(self, ids):
        # The time a passenger agent waits before being destroyed
        for id in ids:
            idx = self.index_by_id(id)
            self.controller.wait_time[id] = self.clock.elapsed_ticks * self.clock.dt - self.pickup_tw_start[idx]

    def index_by_sid(self, agent_sid):
        return np.where(self.passenger_id == agent_sid)[0]
