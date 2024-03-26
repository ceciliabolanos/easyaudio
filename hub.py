import requests
from tqdm import tqdm
from pathlib import Path
import torch
from models.beats.BEATs import BEATs, BEATsConfig
import librosa
import numpy as np

beats_paths = {'BEATs_iter1': "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter1.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D",
         'BEATs_iter2': "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter2.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D",
         'BEATs_iter3': "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D",
         'BEATs_iter3+_AS20K': "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS20K.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D",
         'BEATs_iter3+_AS2M': "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D",
         }

def download_blob(url, filename):
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 1024
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

        with open(filename, 'wb') as f:
            for data in response.iter_content(chunk_size=chunk_size):
                progress_bar.update(len(data))
                f.write(data)

        progress_bar.close()
        print("File downloaded successfully as", filename)
    else:
        print("Failed to download file")

class BEATsWrapped:
    def __init__(self, model):
        self.model = model

    def extract_activations_from_array(self, x):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            x = x.to(self.device)
            rep, pad, feats = self.model.extract_features(x, padding_mask = torch.zeros((1, x.shape[1]), device=self.device), tgt_layer=100)
            feats = [f[0] for f in feats]
        return feats
    
    def extract_activations_from_filename(self, filename):
        x,fs = librosa.core.load(filename, sr=16000)
        acts = self.extract_activations_from_array(x)
        return acts

def get_model(model_name, device='cuda:0'):
    Path('ckpts').mkdir(parents=True, exist_ok=True)
    ckpt_path = Path('ckpts/{}.pt'.format(model_name))
    if model_name in beats_paths:
        if not ckpt_path.exists():
            download_blob(beats_paths[model_name], ckpt_path)
        checkpoint = torch.load(ckpt_path)
        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model'])
        BEATs_model.eval()
        BEATs_model.to(device)
        model = BEATsWrapped(BEATs_model)
        model.device = device

    return model
