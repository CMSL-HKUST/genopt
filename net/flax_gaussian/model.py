"""1D MLP model for diffusion"""

from flax import linen as nn
import jax.numpy as jnp

class DiffusionModel(nn.Module):
    
    dim: int = 64
    
    @nn.compact
    def __call__(self, x, t):
        
        half_dim = self.dim // 2
        emb = jnp.log(10000.0) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
        temb = nn.Dense(self.dim)(emb)
        temb = nn.swish(temb)
        temb = nn.Dense(self.dim)(temb)
        
        x = jnp.expand_dims(x, -1)
        h = nn.Dense(self.dim)(x)
        h = h + temb
        
        for _ in range(3):
            h = nn.Dense(self.dim)(h)
            h = nn.swish(h)
        
        out = nn.Dense(1)(h)
        return out.squeeze(-1)