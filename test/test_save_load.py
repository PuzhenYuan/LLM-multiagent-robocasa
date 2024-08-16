import torch
import numpy as np
import robomimic.utils.file_utils as FileUtils

paras = dict(
    model="model",
    data = np.array([1,2,3,4,5]),
)
torch.save(paras, "model.pth")

ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path="model.pth", ckpt_dict=None)

print(paras)
print(ckpt_dict)
print(paras == ckpt_dict)