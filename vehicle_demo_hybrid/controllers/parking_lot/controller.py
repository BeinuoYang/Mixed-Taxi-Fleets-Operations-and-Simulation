from core.controller import Controller, Generator, ControllerSettings
from core.agent_logic import AgentLogic
from core.tracer import tr
from .states import ParkinglotStates
from ..parking_lot.agents import ParkingLotAgents
from ..network.controller import NetworkController


class ParkingLotController(Controller):
    """
    parkinglots agent controller, just creates and destroys parkinglots
    """

    def register_implementation(self):
        self.agent_group: ParkingLotAgents = ParkingLotAgents("parking", self)
        self.settings: ParkingLotSettings = ParkingLotSettings()
        self.agent_logic: AgentLogic = AgentLogic(self)

        self.agent_generator: ParkingLotGenerator = ParkingLotGenerator(self)
        self.network_controller: NetworkController = self.register_dependency(NetworkController)

    def add_vehicle_to_parking(self, vehicle_ids, parking_ids):
        self.agent_group.add_vehicle(vehicle_ids, parking_ids)

    def remove_vehicle_from_parking(self, vehicle_ids, parking_ids):
        self.agent_group.remove_vehicle(vehicle_ids, parking_ids)


class ParkingLotSettings(ControllerSettings):
    pass


class ParkingLotGenerator(Generator):

    def start(self):
        self.load_parkinglots_from_file()

    def load_parkinglots_from_file(self):
        import numpy as np
        import pandas as pd
        parkinglot_data = pd.read_csv("/".join([self.owner.params["instance_data_dir"], "parking_lots_HZ{}.csv".format(
            self.owner.params["parkinglot"].get("num"))]))

        self.owner.agent_group.create_agents(
            np.array(range(len(parkinglot_data["parkinglot_id"]))),
            "sid", np.array(parkinglot_data["parkinglot_id"]),
            "lon_deg", np.array(parkinglot_data["long"]),
            "lat_deg", np.array(parkinglot_data["lat"]),
            "node_id", np.array(parkinglot_data["node_id"]),
            "capacity", np.array(parkinglot_data["capacity"]),
            "vehicle_ids", [[] for _ in range(len(parkinglot_data["parkinglot_id"]))],
            "current_state", ParkinglotStates.Available
        )

        if self.owner.world.model.sim.settings.debug_mode:
            tr(f"Load ParkingLots: {self.owner.agent_group.sid}")
