import jax.numpy as jnp
import jax.random as random
from tqdm import tqdm

class DDIMSampler:
    
    """DDIM sampler for generating images"""
    
    def __init__(self, 
                 samples_shape,
                 state,
                 scheduler,
                 num_inference_steps):
        
        self.samples_shape = samples_shape
        self.state = state
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        
        step_ratio = self.scheduler.timesteps // num_inference_steps
        timesteps = jnp.arange(0, num_inference_steps) * step_ratio + (step_ratio - 1)
        timesteps = jnp.clip(timesteps, 0, self.scheduler.timesteps - 1)
        
        self.timesteps = timesteps[::-1] 

    def sample(
        self,
        rng,
        x,
        eta = 0.0,
        ):
        
        # Denoise iteratively
        for i, t in enumerate(tqdm(self.timesteps, desc='Sampling')):
            
            rng, step_key = random.split(rng)
            
            # Predict noise
            t_batch = jnp.full((x.shape[0],1), t, dtype=jnp.int32)
            
            predicted_noise = self.state.apply_fn(
                {'params': self.state.ema_params},
                x,
                t_batch,
                train=False
            )
            
            # DDIM sampling step
            x = self._ddim_step(step_key, x, predicted_noise, t, i, eta)
        
        return x
    
    def _ddim_step(self, rng, x_t, predicted_noise, t, step_idx, eta):
        """Single DDIM denoising step"""
        
        # Get alpha values
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        
        # Get previous alpha
        if step_idx < len(self.timesteps) - 1:
            t_prev = self.timesteps[step_idx + 1]
            alpha_prod_t_prev = self.scheduler.alphas_cumprod[t_prev]
        else:
            alpha_prod_t_prev = jnp.array(1.0)
        
        # Predict x_0
        pred_x0 = (x_t - jnp.sqrt(1 - alpha_prod_t) * predicted_noise) / jnp.sqrt(alpha_prod_t)
        
        # Compute variance (sigma_t)
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * jnp.sqrt(variance)
        
        # Compute direction pointing to x_t
        pred_dir = jnp.sqrt(1 - alpha_prod_t_prev - std_dev_t ** 2) * predicted_noise
        
        # Compute x_{t-1}
        x_prev = jnp.sqrt(alpha_prod_t_prev) * pred_x0 + pred_dir
        
        # Add noise
        if eta > 0:
            noise = random.normal(rng, x_t.shape)
            x_prev = x_prev + std_dev_t * noise
        
        return x_prev