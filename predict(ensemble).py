import pathlib
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

torch.set_num_threads(4)


def prepare_clip(clip):
    clip_image = np.sum(clip, 0) / len(clip)
    clip_image = np.transpose(clip_image, [-1, 0, 1])
    clip_image /= 255.
    return torch.tensor(np.expand_dims(clip_image, 0), dtype=torch.float)


class Ensemble(nn.Module):
    def __init__(self, model_d, model_e, model_h):
        super(Ensemble, self).__init__()
        self.model_d = model_d
        self.model_e = model_e
        self.model_h = model_h

    def forward(self, x):
        x1 = self.model_d(x)
        x2 = self.model_e(x)
        x3 = self.model_h(x)
        logits = x1 + x2 + x3
        return logits

    def predict(self, x, device='cpu'):
        logits = self.forward(x.to(device))
        return logits.max(1)[1].data


MODEL_D_FILE = pathlib.Path(__file__).parent.joinpath('classifier-d7-20.ckpt')
MODEL_E_FILE = pathlib.Path(__file__).parent.joinpath('classifier-e3-22.ckpt')
MODEL_H_FILE = pathlib.Path(__file__).parent.joinpath('classifier-h2-30.ckpt')

_model_d = torch.load(MODEL_D_FILE, map_location=torch.device('cpu'))
_model_e = torch.load(MODEL_E_FILE, map_location=torch.device('cpu'))
_model_h = torch.load(MODEL_H_FILE, map_location=torch.device('cpu'))

model = Ensemble(_model_d, _model_e, _model_h)
model.to('cpu')


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
