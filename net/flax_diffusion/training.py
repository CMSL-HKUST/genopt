"""Training script for DDIM diffusion model"""

import os

import jax
import jax.numpy as jnp
import jax.random as random

from flax.training.train_state import TrainState
import tensorflow_datasets as tfds

import optax

import numpy as np
from tqdm import tqdm

try:
    from .model import UNet
    from .utils import save_state_parameters
except:
    from model import UNet
    from utils import save_state_parameters

class DiffusionScheduler:
    """DDIM noise scheduler"""
    
    def __init__(self, timesteps, beta_start, beta_end):
        self.timesteps = timesteps
        
        # Linear beta schedule
        self.betas = jnp.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        
        # Pre-compute values for forward process (adding noise)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, key, x_0, t):
        """
        Add noise to images according to the noise schedule
        """
        noise = random.normal(key, x_0.shape)
        
        # (batch_size, 1) -> (batch_size,)
        t_indices = t[:, 0]  
        
        # Get coefficients for the specific timesteps
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t_indices][:, None, None, None]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t_indices][:, None, None, None]
        
        # Apply reparameterization trick
        noisy_images = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
        
        return noisy_images, noise
    
    
class EMATrainState(TrainState):
    ema_params: dict  
    ema_decay: float = 0.995 

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        return super().create(
            apply_fn=apply_fn, 
            params=params, 
            tx=tx, 
            ema_params=params, 
            **kwargs
        )

    def update_ema(self):
        # new_ema = decay * old_ema + (1-decay) * current_param
        new_ema = jax.tree_util.tree_map(
            lambda ema, p: self.ema_decay * ema + (1.0 - self.ema_decay) * p,
            self.ema_params,
            self.params
        )
        return self.replace(ema_params=new_ema)


def create_train_state(CONFIG, rng):
    """Create initial training state"""
    
    # Initialize model
    d_model = 48
    model = UNet(dim_init=d_model,
                dim_mults=(1, 2, 2, 2),
                attention_resolutions=(16,),
                attention_num_heads=4,
                num_res_blocks=2,
                sinusoidal_embed_dim=d_model,
                time_embed_dim=4 * d_model,
                kernel_size=3,
                num_groups=4,
                dropout=0.1,
                dtype=jnp.float32)
    
    # Initialize parameters
    dummy_x = jnp.ones([1, CONFIG['image_size'], CONFIG['image_size'], CONFIG['channels']])
    dummy_t = jnp.ones([1, 1], dtype=jnp.int32)  

    variables = model.init(rng, dummy_x, dummy_t, train=True)
    params = variables['params']
    
    # Create optimizer
    total_steps = CONFIG['num_epochs'] * (CONFIG['data_size'] // CONFIG['batch_size'])
        
    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=CONFIG['learning_rate'],
        warmup_steps=500,  
        decay_steps=total_steps,
        end_value=1e-6 
    )
    
    tx = optax.adamw(learning_rate=schedule_fn, weight_decay=1e-4)
    
    return EMATrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            ema_decay=0.999 
        )
    

def make_train_step(scheduler: DiffusionScheduler):
    """Create a JIT-compiled training step function with scheduler captured in closure"""
    
    # Pre-extract the arrays we need from scheduler
    timesteps = scheduler.timesteps
    
    @jax.jit
    def train_step(state: TrainState, batch, key):
        """Single training step"""
        
        def loss_fn(params):
            # Sample random timesteps
            batch_size = batch.shape[0]
            t_key, noise_key, dropout_key = random.split(key,3) 
            
            t = random.randint(
                t_key,
                shape=(batch_size,),
                minval=0,
                maxval=timesteps
            )
            
            # (batch_size,) -> (batch_size, 1)
            t = t[:, None]  
            
            # Add noise to images
            noisy_images, noise = scheduler.add_noise(noise_key, batch, t)
            
            predicted_noise = state.apply_fn(
                {'params': params},
                noisy_images,
                t,
                train=True,
                rngs={'dropout': dropout_key}
            )
            
            # MSE loss between predicted and actual noise
            loss = jnp.mean((noise - predicted_noise) ** 2)
            
            return loss
        
        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        
        # Update parameters
        new_state = state.apply_gradients(grads=grads)
        new_state = new_state.update_ema()

        return new_state, loss
    
    return train_step


def train_epoch(CONFIG, state, train_ds, epoch, rng, train_step_fn):
    """Train for one epoch"""
    
    epoch_losses = []
    
    pbar = tqdm(tfds.as_numpy(train_ds), desc=f'Epoch {epoch}', 
                total=CONFIG['data_size'] // CONFIG['batch_size'])
    
    for step, batch in enumerate(pbar):
        rng, step_key = random.split(rng)
        
        state, loss = train_step_fn(state, batch, step_key)
        epoch_losses.append(loss)
        
        if step % 50 == 0:
            pbar.set_postfix({'loss': f'{loss:.4f}'})
    
    return state, np.mean(epoch_losses), rng


def train(dataset, CONFIG):
    
    """Main training loop"""
    
    print('Initializing training...')
    
    # Directories
    MODEL_DIR = CONFIG['model_dir']
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Create scheduler (not part of TrainState)
    scheduler = DiffusionScheduler(
        timesteps=CONFIG['timesteps'],
        beta_start=CONFIG['beta_start'],
        beta_end=CONFIG['beta_end']
    )
    
    # Initialize training state
    rng = random.PRNGKey(42)
    rng, init_key = random.split(rng)
    
    state = create_train_state(CONFIG, init_key)
    
    print(f'Model parameters: {sum(x.size for x in jax.tree_util.tree_leaves(state.params)):,}')
    
    # Create JIT-compiled train step with scheduler
    train_step_fn = make_train_step(scheduler)
    
    # Training loop
    print('Starting training...')
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        
        rng, epoch_key = random.split(rng)
        
        state, avg_loss, rng = train_epoch(CONFIG, state, dataset, epoch, epoch_key, train_step_fn)
        
        print(f'Epoch {epoch}/{CONFIG["num_epochs"]} - Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == CONFIG['num_epochs']:
            save_state_parameters(state, os.path.join(MODEL_DIR, f"model_{epoch:04d}.pt"))
    
    print('Training completed!')
