from core.states import AgentState, StateContainer


class PedestrianState(AgentState):
    pass


class PedestrianStates(StateContainer):
    class Unassigned(PedestrianState):
        name = "unassigned"

    class Assigned(PedestrianState):
        name = "assigned"

    class Waiting(PedestrianState):
        name = "waiting"

    class InService(PedestrianState):
        name = "inservice"

    class ToDropoff(PedestrianState):
        name = "todropoff"

    class Served(PedestrianState):
        name = "served"

