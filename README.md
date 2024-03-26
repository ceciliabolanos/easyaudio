Easy interface for extracting features from audio models.

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
