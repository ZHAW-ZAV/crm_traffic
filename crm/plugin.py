from traffic.core import Traffic
from .los import loss_of_spacing


def _onload():
    setattr(Traffic, "loss_of_spacing", loss_of_spacing)
