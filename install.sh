curl -O https://raw.githubusercontent.com/lucidrains/byol-pytorch/2aa84ee18fafecaf35637da4657f92619e83876d/byol_pytorch/byol_pytorch.py
patch < easyaudio/models/byola/byol_a/byol_pytorch.diff
mv byol_pytorch.py easyaudio/models/byola/byol_a
curl -O https://raw.githubusercontent.com/daisukelab/general-learning/7b31d31637d73e1a74aec3930793bd5175b64126/MLP/torch_mlp_clf.py
mv torch_mlp_clf.py easyaudio/models/byola/utils

pip install -e .
cd easyaudio/models/encodecmae && pip install -e .