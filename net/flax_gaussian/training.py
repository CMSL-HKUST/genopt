"""Training"""
import os

import jax
import jax.numpy as jnp
import jax.random as random

from flax.training import train_state
import optax

from tqdm import tqdm

try:
    from model import DiffusionModel
    from utils import save_state_parameters
except ImportError:
    from .model import DiffusionModel
    from .utils import save_state_parameters

class DiffusionScheduler:
    
    """DDIM noise scheduler"""
    
    def __init__(self, timesteps, beta_start, beta_end):
        self.timesteps = timesteps
        
        self.betas = jnp.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, key, x_0, t):
        """Add noise to data according to the noise schedule"""
        noise = random.normal(key, x_0.shape)
        
        alpha_bar_t = self.alphas_cumprod[t]
        
        noisy_data = jnp.sqrt(alpha_bar_t) * x_0 + jnp.sqrt(1 - alpha_bar_t) * noise
        
        return noisy_data, noise


def create_train_state(CONFIG, rng):
    
    batch_size = CONFIG['batch_size']
    learning_rate = CONFIG['learning_rate']
    
    # model
    model = DiffusionModel()
    
    # optimizer
    tx = optax.adam(learning_rate)
    
    # initial state
    dummy_x = jnp.ones((batch_size,))
    dummy_t = jnp.ones((batch_size,), dtype=jnp.int32)
    params = model.init(rng, dummy_x, dummy_t)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    return state
    
def train_model(data, CONFIG):
    
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    
    key = jax.random.PRNGKey(42)

    T = CONFIG['timesteps']
    epochs = CONFIG['num_epochs']
    batch_size = CONFIG['batch_size']
    
    # scheduler
    scheduler = DiffusionScheduler(timesteps=T, 
                                   beta_start=CONFIG['beta_start'], 
                                   beta_end=CONFIG['beta_end'])
    
    key, subkey = jax.random.split(key)
    state = create_train_state(CONFIG, subkey)
    
    @jax.jit
    def step(state, batch, t, noise):
        def loss_fn(params):
            pred_noise = state.apply_fn(params, batch, t)
            return jnp.mean((pred_noise - noise)**2)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss
    
    losses = []
    num_batches = len(data) // batch_size
    
    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, len(data))
        shuffled_data = data[perm]
        
        epoch_loss = 0
        for i in range(num_batches):
            batch = shuffled_data[i * batch_size:(i + 1) * batch_size]
            
            key, subkey = jax.random.split(key)
            t = jax.random.randint(subkey, (batch_size,), 0, T)
            alpha_bar_t = scheduler.alphas_cumprod[t]
            
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, (batch_size,))
            noisy_data = jnp.sqrt(alpha_bar_t) * batch + jnp.sqrt(1 - alpha_bar_t) * noise
            
            state, loss = step(state, noisy_data, t, noise)
            epoch_loss += float(loss)
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    save_state_parameters(state, os.path.join(CONFIG['model_dir'], f"model_{epoch+1:04d}.pt"))
        
    return state, scheduler, losses