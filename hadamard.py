import torch
import torch.nn as nn


class Hadamard(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.f1 = nn.Sequential(
            nn.Linear(input_dim, input_dim)
        )
        self.f2 = nn.Sequential(
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, x):
        return self.f1(x) * self.f2(x)


# Loads in pre-trained models with:
# hadamard_model = load_hadamard_model(
#     "models_backup/hadamard_f1_1.pth",
#     "models_backup/hadamard_f2_1.pth"
# )
def load_hadamard_model(f1_path, f2_path):
    hadamard = Hadamard(input_dim=784)

    hadamard.f1.load_state_dict(torch.load(f1_path))
    hadamard.f2.load_state_dict(torch.load(f2_path))

    return hadamard