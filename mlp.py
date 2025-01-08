import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)


# Loads in pre-trained mlp with:
# mlp_model = load_mlp_model("models_backup/mlp_elu_512_256_128.pth")
def load_mlp_model(model_path):
    mlp = MLP(input_dim=784)
    mlp.load_state_dict(torch.load(model_path))
    return mlp
