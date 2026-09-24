import equinox

from .bronet import BroNet as BroNet
from .cnn import CNN as CNN
from .mlp import MLP as MLP
from .simba import SimBa as SimBa


class Identity(equinox.Module):
    """Identity with `out_features` set up"""

    out_features: int = equinox.field(static=True)

    def __init__(self, in_features: int, *args, **kwargs):
        self.out_features = in_features

    def __call__(self, x, *args, **kwargs):
        return x
