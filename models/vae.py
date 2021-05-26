from typing import Dict, List, Sequence

import torch
from torch import Tensor, nn
from torch.nn import Module
from torch.nn import functional as F


class VAE(Module):
    def __init__(
        self,
        *,
        in_channels: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = None,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        out_channels = in_channels

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        elif not isinstance(hidden_dims, list):
            hidden_dims = list(hidden_dims)

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        hidden_dims.reverse()

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 4)

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )
        modules.extend(
            [
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    hidden_dims[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(
                    hidden_dims[-1], out_channels=out_channels, kernel_size=3, padding=1
                ),
                nn.Sigmoid(),
            ]
        )
        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, result.shape[1] // 4, 2, 2)
        result = self.decoder(result)
        return result

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param log_var: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        assert mu.shape == log_var.shape
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor) -> Dict[str, Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        return {'input': input, 'recons': recons, 'mu': mu, 'log_var': log_var}

    def loss_function(
        self,
        *,
        input: Tensor,
        recons: Tensor,
        mu: Tensor,
        log_var: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        assert input.shape == recons.shape, f'{input.shape}, {recons.shape}'
        assert mu.shape == log_var.shape, f'{mu.shape}, {log_var.shape}'
        assert len(input) == len(mu), f'{len(input)}, {len(mu)}'

        batch_size = len(input)

        recons_loss = (
            F.binary_cross_entropy(recons, input, reduction='sum') / batch_size
        )
        kld_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum() / batch_size
        loss = recons_loss + kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

    def sample(self, num_samples: int) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)['recons']


if __name__ == '__main__':
    from torchinfo import summary

    batch_size = 2
    in_channels = 1
    latent_dim = 64
    hidden_dims = [16, 32, 64, 128]
    image_size = 32

    # batch_size = 2
    # in_channels = 1
    # latent_dim = 64
    # hidden_dims = [32, 64, 128, 256, 512]
    # image_size = 64

    model = VAE(in_channels=in_channels, latent_dim=latent_dim, hidden_dims=hidden_dims)

    summary(model)

    x = torch.rand(batch_size, in_channels, image_size, image_size)
    y = model.generate(x)
    print(x.shape, y.shape)
