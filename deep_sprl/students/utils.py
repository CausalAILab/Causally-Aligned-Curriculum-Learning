from abc import ABC, abstractmethod


class AgentInterface(ABC):
    """Interface of an RL learner used by the experiments."""

    @abstractmethod
    def estimate_value(self, obs_array):
        """Value estimates of a batch of observations (used by PLR)."""
        pass

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def load(self, agent_path):
        pass

    @abstractmethod
    def train(self, optimizer, log_dir):
        pass

    @abstractmethod
    def evaluate(self, mode, ep):
        pass
