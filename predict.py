import torch
import pathlib
import numpy as np

import warnings

warnings.filterwarnings("ignore")

torch.set_num_threads(4)


def prepare_clip(clip):
    clip_image = np.sum(clip, 0) / len(clip)
    clip_image = np.transpose(clip_image, [-1, 0, 1])
    clip_image /= 255.
    return torch.tensor(np.expand_dims(clip_image, 0), dtype=torch.float)


SAVED_MODEL_FILE = pathlib.Path(__file__).parent.joinpath('classifier-h3-30.ckpt')
model = torch.load(SAVED_MODEL_FILE, map_location=torch.device('cpu'))


def predict(clip: np.ndarray):
    
    classes_dict = {
        0: 'no_action',
        1: 'train_in_out',
        2: 'bridge_up',
        3: 'bridge_down'
    }

    x = prepare_clip(clip).to('cpu')
    with torch.no_grad():
        logit = np.argmax(model(x).cpu().numpy())

    return classes_dict[logit]
