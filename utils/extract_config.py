import torch
import json

model = torch.load("tool_hang_ph_image_epoch_440_succ_74.pth")
with open("optimal_bc_rnn_toolhang.json", "w") as f:
    x = json.loads(model["config"])
    json.dump(x, f, indent=4)