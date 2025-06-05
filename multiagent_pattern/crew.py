from typing import Any, Callable, Dict, List, Optional

from multiagent_pattern.cognitive_agent import CognitiveAgent


class AgentCrew:
    def __init__(self, agents: List[CognitiveAgent], routing_fn: Optional[Callable[[str], List[str]]] = None):
        """
        :param agents: List of CognitiveAgents
        :param routing_fn: Optional task-routing logic: (task) -> list of agent names
        """
        self.agents = {agent.name: agent for agent in agents}
        self.routing_fn = routing_fn or (lambda task: list(self.agents.keys()))  # broadcast by default

    def run(self, task: str) -> Dict[str, Any]:
        route = self.routing_fn(task)
        results = {}
        for name in route:
            agent = self.agents[name]
            res = agent.run(task)
            results[name] = res
        return results
