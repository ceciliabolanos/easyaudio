from pathlib import Path
import torch
import librosa
import numpy as np
import torchaudio
from loguru import logger
from .models import *

available_models = {'BEATs': BEATsWrapped,
          'byola': BYOLAWrapped,
          'encodecmae': EnCodecMAEWrapped}

def get_model(model_name, device='cuda:0'):
    Path('ckpts').mkdir(parents=True, exist_ok=True)
    for k,v in available_models.items():
        if model_name.startswith(k):
            logger.info('Loading {}'.format(model_name))
            model = available_models[k](model_name)

    return model

def list_available_models():
    models_list = []
    for k,v in available_models.items():
        models_list += list(v.list_available_models().keys())
    return models_list