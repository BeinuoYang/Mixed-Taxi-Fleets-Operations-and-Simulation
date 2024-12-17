from core.states import AgentState, StateContainer


class ParkinglotState(AgentState):
    # Parking lots can be either occupied or available, based on whether a vehicle agent is parked or not
    pass


class ParkinglotStates(StateContainer):
    # We define the classes to be used in this container. The container will contain a called instance for actual use
    # with methods, but the class itself can and should be used for identity checks.

    def declare_states(self):
        """
        initialisation of vehicle states - must be hard-coded by the user
        """
        self.all_states = self.autoimport_states(ParkinglotStates)

    class Available(ParkinglotState):
        name = "available"

    class Occupied(ParkinglotState):
        name = 'occupied'

    def change_parking_lot_state(self, parking_id):
        parking_idx = self.agent_group.index_by_id(parking_id)
        occupied = self.agent_group.capacity[parking_idx] <= self.agent_group.occupancy[parking_idx]
        unoccupied = self.agent_group.capacity[parking_idx] > self.agent_group.occupancy[parking_idx]
        self.change_agent_state(parking_id[occupied], ParkinglotStates.Occupied)
        self.change_agent_state(parking_id[unoccupied], ParkinglotStates.Available)
