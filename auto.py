import torch
import torch.nn as nn


class Autoencoder64(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder128(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder256(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256)
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ELU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder512(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class FunkyAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder_and_decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ELU()
        )

    def forward(self, x):
        return self.encoder_and_decoder(x)


# Loads in pre-trained models with:
# autoencoder64 = load_autoencoder64_model(
#     "models_backup/encoder_auto_64_elu_1.pth",
#     "models_backup/decoder_auto_64_elu_1.pth"
# )
def load_autoencoder64_model(encoder_path, decoder_path):
    encoder = Autoencoder64(input_dim=794).encoder
    encoder.load_state_dict(torch.load(encoder_path))

    decoder = Autoencoder64(input_dim=794).decoder
    decoder.load_state_dict(torch.load(decoder_path))

    autoencoder = nn.Sequential(encoder, decoder)
    return autoencoder


def load_autoencoder128_model(encoder_path, decoder_path):
    encoder = Autoencoder128(input_dim=794).encoder
    encoder.load_state_dict(torch.load(encoder_path))

    decoder = Autoencoder128(input_dim=794).decoder
    decoder.load_state_dict(torch.load(decoder_path))

    autoencoder = nn.Sequential(encoder, decoder)
    return autoencoder


def load_autoencoder256_model(encoder_path, decoder_path):
    encoder = Autoencoder256(input_dim=794).encoder
    encoder.load_state_dict(torch.load(encoder_path))

    decoder = Autoencoder256(input_dim=794).decoder
    decoder.load_state_dict(torch.load(decoder_path))

    autoencoder = nn.Sequential(encoder, decoder)
    return autoencoder


def load_autoencoder512_model(encoder_path, decoder_path):
    encoder = Autoencoder512(input_dim=794).encoder
    encoder.load_state_dict(torch.load(encoder_path))

    decoder = Autoencoder512(input_dim=794).decoder
    decoder.load_state_dict(torch.load(decoder_path))

    autoencoder = nn.Sequential(encoder, decoder)
    return autoencoder


def load_funky_autoencoder_model(model_path):
    funky_autoencoder = nn.Sequential(
        nn.Linear(794, 794),
        nn.ELU()
    )
    funky_autoencoder.load_state_dict(torch.load(model_path))
    return funky_autoencoder

