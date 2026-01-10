import os

import jax

from .sampling import DDIMSampler
from .training import DiffusionScheduler, create_train_state
from .utils import load_state_parameters


__all__ = ['prep_sampler']


MODEL_DIR = os.path.join(os.path.dirname(__file__), 'data/model')

def prep_sampler(name, epoch, num_inference_steps, seed):
    
    if name == 'GAUSSIAN':
        from .train_gaussian import CONFIG
        
    rng_seed, ini_rng = jax.random.split(jax.random.PRNGKey(seed))
    
    # Load model
    model_path = os.path.join(MODEL_DIR, name, f'model_{epoch:04d}.pt')
    state_template = create_train_state(CONFIG, ini_rng)
    state = load_state_parameters(state_template, model_path)
    
    # Recreate scheduler
    scheduler = DiffusionScheduler(
        timesteps=CONFIG['timesteps'],
        beta_start=CONFIG['beta_start'],
        beta_end=CONFIG['beta_end']
    )
    
    # Create sampler
    shape = (1, 1)
    sampler = DDIMSampler(
            samples_shape=shape,
            state=state,
            scheduler=scheduler,
            num_inference_steps=num_inference_steps
        )
    
    return sampler, scheduler