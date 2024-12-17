from core.agent_group import AgentGroup
from .states import VehicleStates
from .tasks import VehicleTasks
import numpy as np


class VehicleAgents(AgentGroup):
    """
    Main vehicle agent class.
    """

    def init_agents_custom(self):

        # There are four types of vehicles:
        # 1.AET: Autonomous, Electric 2.AGT: Autonomous, Gasoline 3.HET: Human-driven, Electric 4.Human-driven, Gasoline
        self.is_autonomous = self.declare_property("is_autonomous", bool)
        self.is_electric = self.declare_property("is_electric", bool)

        self.sid = self.declare_property("sid", str)
        self.lat_deg = self.declare_property("lat_deg", float)
        self.lon_deg = self.declare_property("lon_deg", float)
        self.hdg_deg = self.declare_property("hdg_deg", float)
        self.spd_kmh = self.declare_property("spd_kmh", float)

        # ride-sharing available for vehicle, ped_ids indicates the id of the passengers being served by the vehicle
        self.ped_id = self.declare_property("ped_id", list, False)
        self.is_assigned = self.declare_property("is_assigned", bool)
        # Generates a list of lists, each agent has given a list of orders to make
        self.order_sequence = self.declare_property("order_sequence", list, False)
        # battery_capacity unit: kWh
        self.battery_capacity = self.declare_property("battery_capacity", float)
        self.battery_level = self.declare_property("battery_level", float)
        # charging rate unit: kW
        self.charge_rate = self.declare_property("charge_rate", float)
        # energy consumption unit: kWh/km
        self.energy_consumption = self.declare_property("energy_consumption", float)
        self.range_anxiety = self.declare_property("range_anxiety", int)
        # continuous operational time unit: second
        self.cont_op_time = self.declare_property("cont_op_time", float)
        # max continuous operational time unit: hour
        self.max_cont_op_time = self.declare_property("max_cont_op_time", float)
        # min offline time unit: hour
        self.min_offline_time = self.declare_property("min_offline_time", float)

        self.is_online = self.declare_property("is_online", bool)

        self.current_node = self.declare_property("current_node", object)
        self.current_edge = self.declare_property("current_edge", object)
        self.pre_node = self.declare_property("pre_node", object)
        self.next_node = self.declare_property("next_node", object)
        self.lat_next = self.declare_property("lat_next", float)
        self.lon_next = self.declare_property("lon_next", float)
        self.time_next = self.declare_property("time_next", float)
        self.total_distance_traveled = self.declare_property("total_distance_traveled", float)

        self.path = self.declare_property("path", list, is_array=False)

        # generate empty state and task containers
        self.state_container = VehicleStates()
        self.task_container = VehicleTasks()

    def index_by_sid(self, agent_sid):
        return np.where(self.sid == agent_sid)[0]


class Order:
    def __init__(self, sid, stype, node_id, lat, lon, duration=0, t_due=999):
        self.id = sid
        self.stype = stype  # type-- Pickup/Dropoff/ParkingLot
        self.node_id = node_id
        self.lat = lat
        self.lon = lon
        self.t_due = t_due
        self.dura = duration
