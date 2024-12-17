import numpy as np

from core.tracer import tr
from core.feature import Feature
from core.utils.units import hours_to_seconds
from ..vehicles.controller import VehicleController
from ..vehicles.states import VehicleStates


class EnergyFeature(Feature):
    """
    Feature test, just adds counters at different points in the model - only attached to vehicles
    """

    @property
    def dt(self):
        return self.world.clock.dt

    def init_feature_custom(self):
        self.register_controller_start(VehicleController, self.on_controller_start)
        self.register_controller_step(VehicleController, self.on_controller_step)
        self.register_controller_end(VehicleController, self.on_controller_end)

        self.register_state_start(VehicleStates.Waiting.name, self.on_state_stopping_start)
        self.register_state_start(VehicleStates.Idle.name, self.on_state_stopping_start)

        self.register_state_start(VehicleStates.Charging.name, self.on_state_charging_start)
        self.register_state_step(VehicleStates.Charging.name, self.on_state_charging_step)
        self.register_state_end(VehicleStates.Charging.name, self.on_state_charging_end)

        self.register_state_end(VehicleStates.Moving.name, self.on_state_moving_end)
        self.register_state_step(VehicleStates.Moving.name, self.on_state_moving_step)

    def declare_feature_dependencies(self):
        self.declare_dependency(VehicleController)

    @Feature.agent_group_is_class
    def on_controller_start(self, agent_group):
        """
        at start of controller instantiation, create the relevant properties
        """
        pass

    @Feature.agent_group_is_class
    def on_controller_step(self, agent_group):
        """
        function at end of specific state
        """
        pass

    @Feature.agent_group_is_class
    def on_controller_end(self, agent_group):
        """
        at end of controller run, collect data?
        """
        pass

    @Feature.agent_group_is_class
    def on_state_moving_step(self, agent_group, ids):
        """
        function at each step for specific state
        """
        indices = agent_group.index_by_id(ids)
        indices = indices[np.array(agent_group.is_electric[indices])]
        energy_consumption = (agent_group.spd_kmh[indices]/hours_to_seconds(1) * self.dt *
                              agent_group.energy_consumption[indices])
        agent_group.battery_level[indices] -= energy_consumption

    @Feature.agent_group_is_class
    def on_state_moving_end(self, agent_group, ids):
        """
        function at end of specific state
        """
        indices = agent_group.index_by_id(ids)
        indices = indices[np.array(agent_group.is_electric[indices])]
        battery_level = agent_group.battery_level[indices]

        if len(indices) > 0 and self.world.model.sim.settings.debug_mode:
            tr(f"Vehicle {ids} battery level: {battery_level} kWh")

    @Feature.agent_group_is_class
    def on_state_stopping_start(self, agent_group, ids):
        """
        function at start of specific state
        """
        pass

    @Feature.agent_group_is_class
    def on_state_charging_start(self, agent_group, ids):
        """
        function at start of charging state
        """
        pass

    @Feature.agent_group_is_class
    def on_state_charging_step(self, agent_group, ids):
        """
        function at each step for charging state
        """
        indices = agent_group.index_by_id(ids)
        for index in indices:
            charge_energy = self.dt * agent_group.charge_rate[index]/hours_to_seconds(1)
            agent_group.battery_level[index] = min(agent_group.battery_level[index] + charge_energy,
                                                   agent_group.battery_capacity[index])

    def on_state_charging_end(self, agent_group, ids):
        """
        function at end for charging state
        """
        indices = agent_group.index_by_id(ids)
        battery_level = agent_group.battery_level[indices]

        if len(ids) > 0 and self.world.model.sim.settings.debug_mode:
            tr(f"Vehicle {ids} battery level: {battery_level} kWh")
