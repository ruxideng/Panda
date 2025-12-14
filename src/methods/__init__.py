from .CLIP import CLIP
from .TestImageFeat import TestImageFeat
from .BAT import BAT

from .DeYO import DeYO

from .Tent import Tent
from .ETA import ETA
from .SAR import SAR

from .TPS import TPS
from .TPT import TPT
from .Zero import Zero

from .DMN import DMN

def get_method_class(method_name):
    if method_name not in globals():
        raise NotImplementedError("Method not found: {}".format(method_name))
    return globals()[method_name]