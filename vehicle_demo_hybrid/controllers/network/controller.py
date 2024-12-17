from core.controller import Controller, ControllerSettings, Generator
from .networks import Network
from core.tracer import tr


class NetworkController(Controller):

    def init_controller(self):
        self.network = Network(self)

    def register_implementation(self):
        self.settings: NetworkControllerSettings = NetworkControllerSettings()
        self.agent_generator = NetworkGenerator(self)


class NetworkControllerSettings(ControllerSettings):
    def init_settings(self):
        self.latlon_box = (30.247176,120.348917,30.16201,120.232031)
        self.weight_label = "length"


class NetworkGenerator(Generator):
    def start(self):
        self.load_network()

    def load_network(self):
        file = self.owner.params["network"]["file"]
        label = self.owner.params["network"].get("weight_label")
        network = Network(self.owner)
        network.load_graph_from_graphml(file)
        if label:
            network.weight_label = label
        self.owner.network = network

        if self.owner.world.model.sim.settings.debug_mode:
            tr("Road network loaded")
