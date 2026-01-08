import os

import jax

from .sampling import DDIMSampler
from .training import DiffusionScheduler, create_train_state
from .utils import load_state_parameters

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'data/model')

def prep_sampler(name, epoch, num_inference_steps, seed):
    
    if name == 'MNIST':
        from .train_mnist import CONFIG
    elif name == 'MICRO_GRAY':
        from .train_micro import CONFIG
    elif name == 'MICRO':
        from .train_micro import CONFIG
        CONFIG['model_dir'] = CONFIG['model_dir'][:-5]
        CONFIG['dataset_path'] = CONFIG['dataset_path'][:-9] + CONFIG['dataset_path'][-4:]

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
    shape = (1, CONFIG['image_size'], CONFIG['image_size'], CONFIG['channels'])
    sampler = DDIMSampler(
            samples_shape=shape,
            state=state,
            scheduler=scheduler,
            num_inference_steps=num_inference_steps
        )
    
    return sampler