import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_ResidualBlock, VAE_AttentionBlock

class VAE_Encoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            # (batch_size, 3, height, width) -> (batch_size, 128, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(128, 128),
            
            VAE_ResidualBlock(128, 128),
            
            # (batch_size, 128, height, width) -> (batch_size, 128, height/2, width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            VAE_ResidualBlock(128, 256),
            
            VAE_ResidualBlock(256, 256),
            
            # (batch_size, 256, height/2, width/2) -> (batch_size, 256, height/4, width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            
            VAE_ResidualBlock(256, 512),
            
            VAE_ResidualBlock(512, 512),
            
            # (batch_size, 512, height/4, width/4) -> (batch_size, 512, height/8, width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            
            VAE_ResidualBlock(512, 512),
            
            VAE_ResidualBlock(512, 512),
            
            
            VAE_ResidualBlock(512, 512),
            
            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),
            
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            nn.GroupNorm(32, 512),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 512, height/8, width/8)
            nn.SiLU(),
            
            # (batch_size, 512, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            
            # (batch_size, 8, height/8, width/8) -> (batch_size, 8, height/8, width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )
        
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x (batch_size, 3, height, width)
        # noise (batch_size, out_channels, height/8, width/8)
        
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                F.pad(x, (0, 1, 0, 1))
            x = module(x)
            
        mean, log_variance = torch.chunk(x, 2, dim=1)
        
        log_variance = torch.clamp(log_variance, min=-30, max=20)
        
        variance = log_variance.exp()
        
        std = variance.sqrt()
        
        x = mean + std * noise
        
        x *= 0.18215
        
        return x