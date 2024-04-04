from .beats.BEATs import BEATs, BEATsConfig
from .byola.byol_a.common import load_yaml_config
from .byola.byol_a.augmentations import PrecomputedNorm
from .byola.byol_a.models import AudioNTT2020
from encodecmae import load_model

from easyaudio.utils import download_blob

import torchaudio
import torch
from pathlib import Path

class BEATsWrapped:
    def __init__(self, model, device='cuda:0'):
        beats_paths = self.list_available_models()
        ckpt_path = Path('ckpts/{}.pt'.format(model))
        if model in beats_paths:
            if not ckpt_path.exists():
                download_blob(beats_paths[model], ckpt_path)
            checkpoint = torch.load(ckpt_path)
            cfg = BEATsConfig(checkpoint['cfg'])
            self.cfg = cfg
            BEATs_model = BEATs(cfg)
            BEATs_model.load_state_dict(checkpoint['model'])
            BEATs_model.eval()
            BEATs_model.to(device)
            self.model = BEATs_model
            self.device = device
        else:
            raise Exception('Unrecognized model. Available BEATs models are: {}'.format(beats_paths.keys()))

    @staticmethod
    def list_available_models():
        beats_paths = {
         'BEATs_iter1': "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter1.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D",
         'BEATs_iter2': "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter2.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D",
         'BEATs_iter3': "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D",
         'BEATs_iter3+_AS20K': "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS20K.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D",
         'BEATs_iter3+_AS2M': "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D",
         }
        return beats_paths

    def extract_activations_from_array(self, x):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x.astype(np.float32))
            if x.ndim == 1:
                x = x.unsqueeze(0)
            x = x.to(self.device)
            rep, pad, feats = self.model.extract_features(x, padding_mask = torch.zeros((1, x.shape[1]), device=self.device), tgt_layer=100)
            feats = [f[0][:,0,:] for f in feats]
        return feats
    
    def extract_activations_from_filename(self, filename):
        x,fs = librosa.core.load(filename, sr=16000)
        acts = self.extract_activations_from_array(x)
        return acts

class BYOLAWrapped:
    def __init__(self, model, device='cuda:0'):
        config_path = Path(Path(__file__).parent,'byola/config.yaml')
        cfg = load_yaml_config(config_path)
        stats = [-5.4919195,  5.0389895]
        self.to_melspec = torchaudio.transforms.MelSpectrogram(
                                sample_rate=cfg.sample_rate,
                                n_fft=cfg.n_fft,
                                win_length=cfg.win_length,
                                hop_length=cfg.hop_length,
                                n_mels=cfg.n_mels,
                                f_min=cfg.f_min,
                                f_max=cfg.f_max,
                            ).to(device)
        self.normalizer = PrecomputedNorm(stats)
        d = int(model.split('_')[-1])
        if d not in [512, 1024, 2048]:
            raise Exception('No weights available for BYOLA with d={}'.format(d))
        self.model = AudioNTT2020(d=d).to(device)
        self.model.load_weight(Path(Path(__file__).parent,'byola/pretrained_weights/AudioNTT2020-BYOLA-64x96d{}.pth'.format(d)), device)
        cfg.d = d
        self.cfg = cfg
        self.device = device

    @staticmethod
    def list_available_models():
        models_path = str(Path(__file__).parent.resolve())
        byola_paths = {'byola_{}'.format(d): '{}/models/byola/pretrained_weights/AudioNTT2020-BYOLA-64x96d{}.pth'.format(models_path,d) for d in [512, 1024, 2048]}
        return byola_paths

    def extract_activations_from_filename(self, filename):
        wav, sr = torchaudio.load(filename)
        assert sr == self.cfg.sample_rate, "Let's convert the audio sampling rate in advance, or do it here online."
        acts = self.extract_activations_from_array(wav)
        return acts

    def extract_activations_from_array(self, x):
        activations = {}
        hooks = {}

        def hook_fn(layer_name):
            #Grab outputs of linear layers and features before downsampling
            def hook(m,i,o):
                if m.__class__.__name__ == 'MaxPool2d':
                    activations[layer_name] = i[0]
                elif m.__class__.__name__ == 'Linear':
                    activations[layer_name] = o
            return hook
        
        for k,v in self.model.features._modules.items():
            hooks['features_{}'.format(k)] = v.register_forward_hook(hook_fn('features_{}'.format(k)))
        for k,v in self.model.fc._modules.items():
            hooks['fc_{}'.format(k)] = v.register_forward_hook(hook_fn('fc_{}'.format(k)))
        
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x.astype(np.float32))
            if x.ndim == 1:
                x = x.unsqueeze(0)
            x = x.to(self.device)
            lms = self.normalizer((self.to_melspec(x) + torch.finfo(torch.float).eps).log())
            features = self.model(lms.unsqueeze(0))
            activations['features'] = features
        
        for k,v in hooks.items():
            v.remove()

        act_keys = ['features_3', 'features_7', 'features_11', 'fc_0', 'fc_3', 'features']
        activations = [activations[k] for k in act_keys]
        return activations

class EnCodecMAEWrapped:
    def __init__(self, model, device='cuda:0'):
        self.model = load_model(model.split('encodecmae_')[-1], device=device)
    
    def extract_activations_from_filename(self, filename):
        return self.model.extract_features_from_file(filename, layer='all')
    
    def extract_activations_from_array(self, x):
        return self.model.extract_features_from_array(x, layer='all')

    @staticmethod
    def list_available_models():
        models = {'encodecmae_{}'.format(k): 'huggingface:lpepino/encodecmae-{}'.format(k) for k in ['base', 'small', 'base-st', 'large', 'large-st']}
        return models