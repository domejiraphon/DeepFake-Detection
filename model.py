# standard libraries

# external libraries
# internal libraries
import utils.detection_and_localization.model as dal_m
from utils import Detector


def get_network(config):
    if False:
        model = dal_m.Network(config)
    elif False:
        model = Network(config)
    else:
        model = Detector(config)

    model = model.to(config.parse_args().devices)
    # model.FER.to_device(config.parse_args().devices)
    return model
