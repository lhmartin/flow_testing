from flow_testing.flow.sampleable import Sampleable
import torch

class R3Flow(torch.nn.Module):
    """
    Flows
    """

class R3FlowSampler(Sampleable):

    def __init__(self, flow: R3Flow):
        self.flow = flow

    def sample(self, n_samples: int) -> torch.Tensor:
        return self.flow.sample(n_samples)