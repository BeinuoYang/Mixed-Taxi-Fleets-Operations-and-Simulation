import numpy as np
import warnings

from core.agent_group import AgentGroup
from ..parking_lot.states import ParkinglotStates


class ParkingLotAgents(AgentGroup):
    """
    Main ParkingLot agent class.

    Each parking spot contains a location (latitude and longitude), as well as an occupancy parameter, indicating the
     number of vehicles parked at the lot, a capacity that denotes the maximum number of vehicles that can be parked
     simultaneously, and a list of the vehicles currently parked at a lot.
    """

    def init_agents_custom(self):
        self.sid = self.declare_property('sid', str)
        self.node_id = self.declare_property('node_id', float)
        self.capacity = self.declare_property("capacity", int)
        self.EarliestTime = self.declare_property("EarliestTime", float)
        self.LatestTime = self.declare_property("LatestTime", float)
        self.occupancy = self.declare_property("occupancy", int)
        self.lat_deg = self.declare_property("lat_deg", float)
        self.lon_deg = self.declare_property("lon_deg", float)

        self.vehicle_ids = self.declare_property("vehicle_ids", list, False)
        # list of lists, contains the set of vehicles at a specific parkinglot

        # list of lists.  The electricity prices vary at different times of the day
        # e.g. [[0, 10, 1], [10, 20, 2] means electricity_price=1 if 0<=t<10, =2 if 10<=t<20
        self.electricity_price = self.declare_property("electricity_price", list, False)
        self.charging_power = self.declare_property("charging_power", float)  # constant value

        self.state_container: ParkinglotStates = ParkinglotStates()

    def destroy_parkinglot(self, agent_ids):
        """
        removes parkingLots based on their id
        :param agent_ids: agent ids to destroy, can be a single str or a list of str
        :return:
        """
        super().destroy_agents(agent_ids)
        print(f'Destroyed parkingLots {agent_ids}')

    def add_vehicle(self, vehicle_ids, parking_ids):
        """
        adds vehicles to parking
        """
        parking_idx = self.index_by_id(parking_ids)
        for i, v in enumerate(vehicle_ids):
            pix = parking_idx[i]
            self.vehicle_ids[pix].append(v)
            self.occupancy[pix] += 1

        if np.any(self.occupancy > self.capacity):
            warnings.warn("Parking capacity has been exceeded.")

        self.state_container.change_parking_lot_state(parking_ids)

    def remove_vehicle(self, vehicle_ids, parking_ids):
        """
        removes vehicle from a specific parking spot
        """
        parking_idx = self.index_by_id(parking_ids)
        for i, v in enumerate(vehicle_ids):
            pix = parking_idx[i]
            if v in self.vehicle_ids[pix]:
                self.vehicle_ids[pix].remove(v)
                self.occupancy[pix] -= 1
            else:
                warnings.warn("Vehicle not found at parking lot.")

        self.state_container.change_parking_lot_state(parking_ids)

    def index_by_sid(self, agent_sid):
        return np.where(self.sid == agent_sid)[0]

    def index_by_node_id(self, node_id):
        if isinstance(node_id, np.ndarray):
            result = []
            for target_id in node_id:
                index = np.where(self.node_id == target_id)[0]
                result.extend(index.tolist())
            return np.array(result)
        else:
            return np.where(self.node_id == node_id)[0]
