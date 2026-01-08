import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.random as random

import numpy as np
import matplotlib.pyplot as plt

from training import DiffusionScheduler, create_train_state
from sampling import DDIMSampler
from utils import load_state_parameters


def generate_samples(
                    CONFIG,
                    epoch,
                    num_samples,
                    num_inference_steps,
                    eta,
                    seed):
    
    """Generate and display samples"""
    
    rng_seed, ini_rng = jax.random.split(jax.random.PRNGKey(seed))
    
    # Load model
    MODEL_DIR = CONFIG['model_dir']
    model_path = os.path.join(MODEL_DIR, f'model_{epoch:04d}.pt')
    print(f'Loading model parameters from {model_path}')
    state_template = create_train_state(CONFIG, ini_rng)
    state = load_state_parameters(state_template, model_path)
    
    # Recreate scheduler
    scheduler = DiffusionScheduler(
        timesteps=CONFIG['timesteps'],
        beta_start=CONFIG['beta_start'],
        beta_end=CONFIG['beta_end']
    )
    
    # Create sampler
    shape = (num_samples, CONFIG['image_size'], CONFIG['image_size'], CONFIG['channels'])
    sampler = DDIMSampler(
            samples_shape=shape,
            state=state,
            scheduler=scheduler,
            num_inference_steps=num_inference_steps
        )
    
    print(f'Generating {num_samples} samples...')
    rng_seed, noise_rng, sample_rng = jax.random.split(rng_seed, 3)
    
    # Generate samples
    x_noise = random.normal(noise_rng, shape)
    samples = sampler.sample(sample_rng, x_noise)
    
    # grads = jax.grad(lambda x: np.sum(sampler.sample(x,params=state.params,
    # rng=rng,
    # shape=shape,
    # num_inference_steps=num_inference_steps,
    # eta=eta)))(x_noise)
    
    samples = np.array(samples)
    
    # Plot samples
    n_rows = int(np.sqrt(num_samples))
    n_cols = (num_samples + n_rows - 1) // n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten() if num_samples > 1 else [axes]
    for idx, ax in enumerate(axes):
        if idx < num_samples:
            ax.imshow(samples[idx, :, :, 0], cmap='gray_r')
            ax.axis('off')
        else:
            ax.remove()
    plt.tight_layout()
    plt.show()
    
    return samples


from train_mnist import CONFIG
from train_micro import CONFIG
# Generate samples
samples = generate_samples(
                        CONFIG,
                        epoch=15,  
                        num_samples=100,
                        num_inference_steps=10,
                        eta=0.0,  
                        seed=3
                        )
