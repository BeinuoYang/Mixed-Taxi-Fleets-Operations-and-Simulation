from core.states import AgentState, StateContainer, ToNextTaskState
from core.utils.objectopers import index_optional
from core.utils.geoopers import distance_km


class VehicleStates(StateContainer):
    # We define the classes to be used in this container. The container will contain a called instance for actual use
    # with methods, but the class itself can and should be used for identity checks.

    """
    6 states:   idle - moves the vehicle to the next task in its queue
                moving - moves the vehicle based on a given speed and direction - to be linked to the network
                waiting - vehicle is static, used for pickup and dropoff tasks
                charging - vehicle is charging in a parking lot
                queue - vehicle is queuing to charge in a parking lot
                finished - The vehicle has finished work or has not started work
    """

    class Idle(ToNextTaskState):
        name = "Idle"

    class Moving(AgentState):
        name = "Moving"

    class Waiting(AgentState):
        name = "Waiting"

    class Charging(AgentState):
        name = "Charging"

    class Queuing(AgentState):
        name = "Queuing"

    class Finished(AgentState):
        name = "Finished"
