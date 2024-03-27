Easy interface for extracting features from audio models.

#### Instructions:
- Git clone this repository and its submodules:
  ```
  git clone --recurse-submodules https://github.com/mrpep/easy-audio-embeddings.git
  ```
- Run install.sh to apply patches to BYOL-A models

#### Usage:

```python
from hub import get_model
import numpy as np
import torch

model = get_model('BEATs_iter3')
features = model.extract_activations_from_array(np.random.randn(16000)) #Extract features from numpy array
features = model.extract_activations_from_array(torch.randn((16000,))) #From torch tensor
features = model.extract_activations_from_filename('example.wav') #Given a wav filename

#features is a list of tensors corresponding to the activations from each layer. Each activation has shape (T,D)
```

#### Available models:
##### BEATs:
  Paper: https://arxiv.org/abs/2212.09058
  
  Official code: https://github.com/microsoft/unilm/tree/master/beats
  
  Available models: *BEATs_iter1, BEATs_iter2, BEATs_iter3, BEATs_iter3+_AS20K, BEATs_iter3+_AS2M*
  
  Extraction points: activations are extracted from the transformer input and the output of each transformer layer.
  
##### BYOL-A
  Paper: https://arxiv.org/abs/2204.07402
  
  Official code: https://github.com/nttcslab/byol-a
  
  Available models: *byola_512, byola_1024, byola_2048*
  
  Extraction points: activations are extracted from the input of each MaxPooling2D layer and the output of each Linear layer. The last activation in the list corresponds to the original BYOL-A features (final output) that is pooled over time/freq.
