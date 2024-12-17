from core.agent_logic import AgentLogic


class PedestrianLogic(AgentLogic):
    """
    basic logic, just uses the features
    """

    def step_logic(self):
        self.agents.destroy_served_pedestrians()
        self.agents.destroy_unassigned_pedestrians()
