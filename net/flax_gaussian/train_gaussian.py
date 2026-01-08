import os

import jax

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

try:
    from sampling import DDIMSampler
    from training import train_model
except ImportError:
    from .sampling import DDIMSampler
    from .training import train_model

# Hyperparameters
CONFIG = {
    'data_size': 20000,
    'num_epochs': 100,
    'batch_size': 128,
    'learning_rate': 1e-4,
    'timesteps': 1000,
    'beta_start': 0.0001,
    'beta_end': 0.02,
    'model_dir': os.path.join(os.path.dirname(__file__),
                              'data', 'model', 'GAUSSIAN'),
    'mean': 2.5,      
    'std': 0.5,       
    'num_inference_steps': 50,  
    'eta': 0.0,       
}


def create_data(CONFIG):
    n_samples = CONFIG['data_size']
    np.random.seed(42)
    comp1 = np.random.normal(loc=-CONFIG['mean'], scale=CONFIG['std'], size=n_samples//2)
    comp2 = np.random.normal(loc=CONFIG['mean'], scale=CONFIG['std'], size=n_samples//2)
    data = np.concatenate([comp1, comp2])
    np.random.shuffle(data)
    return data

data = create_data(CONFIG)

if __name__ == '__main__':
    
    #%% TRAIN
    state, scheduler, losses = train_model(data, CONFIG)
    
    #%% SAMPLE
    num_inference_steps=CONFIG['num_inference_steps']
    sampler = DDIMSampler(samples_shape=(1,1),
                          state=state,
                          scheduler=scheduler,
                          num_inference_steps=num_inference_steps)
    
    num_samples = 10000
    key = jax.random.PRNGKey(42)
    # noise
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (num_samples,))
    # sample
    key, subkey = jax.random.split(key)
    samples, trajectory = sampler.sample(subkey, x, eta=0.0)
    
    
    #%% POSTPROCESS
    def diffusion_pdf(current_alpha_bar):
        m_scale = np.sqrt(current_alpha_bar)
        s_val = np.sqrt(CONFIG['std']**2 * current_alpha_bar + (1 - current_alpha_bar))
        m1 = -CONFIG['mean'] * m_scale
        m2 = CONFIG['mean'] * m_scale
        pdf = 0.5 * norm.pdf(x_range, m1, s_val) + 0.5 * norm.pdf(x_range, m2, s_val)
        return pdf
    
    #%% forward - heatmap
    noise_data = data[:CONFIG['batch_size']//10]
    
    plt.figure(figsize=(12, 6))
    
    resolution = 1000
    x_range = np.linspace(-10, 10, resolution)
    num_forward_steps = scheduler.timesteps
    forward_pdf = np.zeros((resolution, num_forward_steps))
    
    for t in range(num_forward_steps):
        current_alpha_bar = scheduler.alphas_cumprod[t]
        forward_pdf[:, t] = diffusion_pdf(current_alpha_bar)
    
    plt.imshow(forward_pdf, extent=[0, num_forward_steps-1, -10, 10], aspect='auto',
              cmap='Reds', alpha=0.8, origin='lower')
    plt.colorbar(label='Theoretical Probability Density')
    
    num_trajectories_to_plot = min(4, noise_data.shape[0])
    colors = plt.cm.rainbow(np.linspace(0, 1, num_trajectories_to_plot))
    
    forward_trajectory = np.zeros((num_forward_steps, num_trajectories_to_plot))
    
    for i in range(num_trajectories_to_plot):
        x0 = noise_data[i:i+1]
        
        for t in range(num_forward_steps):
            key = jax.random.PRNGKey(42 + i * num_forward_steps + t)
            x_t, _ = scheduler.add_noise(key, x0, t)
            forward_trajectory[t, i] = x_t.item()
        
        time_steps = np.arange(num_forward_steps)
        plt.plot(time_steps, forward_trajectory[:, i], 
                color=colors[i], linewidth=1.5, alpha=0.8)
        plt.scatter(0, forward_trajectory[0, i], color=colors[i], s=50, marker='*', edgecolors='black')
        plt.scatter(num_forward_steps-1, forward_trajectory[-1, i], color=colors[i], s=30, marker='o')
    
    plt.xlabel('Forward Diffusion Steps (t)')
    plt.ylabel('Data Value (x)')
    plt.title('Forward Diffusion: From Structure to Noise')
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    plt.text(10, -5.5, 'Start (Data, t=0)', rotation=90, verticalalignment='bottom')
    
    plt.axvline(x=num_forward_steps-1, color='black', linestyle='--', alpha=0.3)
    plt.text(num_forward_steps-20, -5.5, f'End (Noise, t={num_forward_steps-1})', rotation=90, verticalalignment='bottom', horizontalalignment='right')
    
    plt.tight_layout()
    plt.show()
    
    #%% reverse
    trajectory = np.array(trajectory) # (num_inference_steps, num_samples)

    ddim_timesteps = sampler.timesteps  # [999, 979, 959, ..., 19]
    num_steps = len(ddim_timesteps)
    
    resolution = 1000
    x_range = np.linspace(-6, 6, resolution)
    theory_pdf = np.zeros((resolution, num_inference_steps))
    time_list = []
    for traj_idx in range(num_inference_steps):
        
        if traj_idx + 1 < len(ddim_timesteps):
            t_current = int(ddim_timesteps[traj_idx + 1])
            current_alpha_bar = scheduler.alphas_cumprod[t_current]  
        else:
            t_current = 0
            current_alpha_bar = 1.0  
        
        # Theory
        theory_pdf[:, traj_idx] = diffusion_pdf(current_alpha_bar)
        time_list.append(t_current)
        
    # sampling - hist
    plt.figure(figsize=(12, 6))
    steps_to_plot_idx = [0, 
                         num_steps//5-1, 
                         2*num_steps//5-1, 
                         3*num_steps//5-1, 
                         4*num_steps//5-1, 
                         num_steps-1]
    
    for i, traj_idx in enumerate(steps_to_plot_idx):
        plt.subplot(2, 3, i+1)
        traj_data = trajectory[traj_idx,:]
        t_current = time_list[traj_idx]        
        plt.plot(x_range, theory_pdf[:, traj_idx], 'k--', linewidth=2, label='Theory' if i==0 else None)
        plt.hist(traj_data, bins=50, density=True, alpha=0.6, color='skyblue', 
                 label='Generated' if i==0 else None)
        plt.title(f"Step {traj_idx+1} (t={t_current})")
        plt.xlim(-6, 6)
        plt.ylim(0, 0.8)
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.show()
    
    # sampling - heatmap
    plt.figure(figsize=(12, 6))    
    plt.imshow(theory_pdf, extent=[1, num_inference_steps, -6, 6], aspect='auto',
              cmap='Blues', alpha=0.8, origin='lower')
    plt.colorbar(label='Theoretical Probability Density')
    time_steps = np.arange(len(trajectory))+1
    
    num_trajectories_to_plot = min(8, trajectory.shape[1])
    colors = plt.cm.rainbow(np.linspace(0, 1, num_trajectories_to_plot))
    
    for i in range(num_trajectories_to_plot):
        plt.plot(time_steps, trajectory[:, i], 
                color=colors[i], linewidth=1.5, alpha=0.8)
        plt.scatter(1, trajectory[0, i], color=colors[i], s=30, marker='o')
        plt.scatter(len(trajectory), trajectory[-1, i], color=colors[i], s=50, marker='*', edgecolors='black')
    
    plt.xlabel(f'DDIM Reverse Steps (0 = Pure Noise, {num_inference_steps} = Generated Data)')
    plt.ylabel('Data Value (x)')
    plt.title('DDIM Sampling: From Noise to Structure')
    
    plt.axvline(x=1, color='black', linestyle='--', alpha=0.3)
    plt.text(1, -5.5, 'Start (Noise)', rotation=90, verticalalignment='bottom')
    
    plt.axvline(x=num_inference_steps, color='black', linestyle='--', alpha=0.3)
    plt.text(num_inference_steps-1, -5.5, 'End (Data)', rotation=90, verticalalignment='bottom', horizontalalignment='right')
    
    # plt.grid(True, alpha=0.2, color='white')
    plt.tight_layout()
    plt.show()