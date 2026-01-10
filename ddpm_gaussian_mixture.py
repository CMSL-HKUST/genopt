"""

Gaussian mixture problem

"""
import os
import pickle

import jax
import jax.numpy as np

import numpy as onp
import scipy
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "Helvetica",
    "font.family": "Serif",
    "font.size":  20,
    "lines.linewidth": 3,
    'axes.linewidth': 2,})

from net.flax_gaussian import prep_sampler
from net.flax_gaussian.train_gaussian import CONFIG

from helper import logger
from helper.opt import Optimizer

# rng seed
seed = 0

# dir & path
work_dir = '../results/gaussian'
os.makedirs(work_dir, exist_ok=True)

jax.config.update("jax_enable_x64", True)

#%% PREPARATION
# sampler
num_inference_steps = 50
sampler, scheduler = prep_sampler('GAUSSIAN', epoch=100, num_inference_steps=num_inference_steps, seed=seed)

# noise
rng_seed, sample_rng = jax.random.split(jax.random.PRNGKey(seed))
rng_seed, step_rng = jax.random.split(rng_seed)
num_samples = (1,)
sample_shape = (1, 1)
x_noise_ini = jax.random.normal(step_rng, sample_shape)

def loss_fn(x):
    y = sampler.sample(sample_rng, x.reshape(sample_shape))[0]
    return (y[0,0]+2.5)**2

#%% OPTIMIZATION
results_file = os.path.join(work_dir, 'gaussian.pkl')
if not os.path.exists(results_file):
    log_file = os.path.join(work_dir, 'gaussian.log')
    logger.update_config(log_file=log_file, level='INFO', output='a')
    
    optimizer = Optimizer(method='BFGS', backend='scipy', logger=logger)
    
    xiter = onp.array(x_noise_ini.reshape(-1, order='F'))
    scipy_options = {'maxiter': 100, 'maxls':100, 'disp': False, 'ftol': 1e-20, 'gtol': 1e-20}
    result = optimizer.minimize(objective=loss_fn, 
                                x0=xiter,
                                tol = 1e-3,
                                options=scipy_options)
    
    xs = onp.array(result.history['xs'])
    ys = []
    trajs = []
    for x in xs:
        y, traj = sampler.sample(sample_rng, x)
        ys.append(y)
        trajs.append(traj)
    xs = xs.flatten()
    ys = onp.array(ys).flatten()
    trajectory = onp.array(trajs)[:,:,0].T
    
    # save
    with open(results_file, "wb") as f:
        pickle.dump([xs, ys, trajectory], f)
    
else:
    with open(results_file, "rb") as f:  
        xs, ys, trajectory = pickle.load(f)  

#%% POSTPROCESS
save_dir = '../paper/gaussian'
os.makedirs(save_dir, exist_ok=True)

ddim_timesteps = sampler.timesteps
custom_cmap = LinearSegmentedColormap.from_list('light_blues', 
                                                plt.cm.Blues(np.linspace(0, 0.8, 256)), 
                                                N=256)

def diffusion_pdf(current_alpha_bar):
    m_scale = onp.sqrt(current_alpha_bar)
    s_val = onp.sqrt(CONFIG['std']**2 * current_alpha_bar + (1 - current_alpha_bar))
    m1 = -CONFIG['mean'] * m_scale
    m2 = CONFIG['mean'] * m_scale
    pdf = 0.5 * scipy.stats.norm.pdf(x_range, m1, s_val) + 0.5 * scipy.stats.norm.pdf(x_range, m2, s_val)
    return pdf
    
resolution = 1000
x_range = onp.linspace(-6, 6, resolution)
theory_pdf = onp.zeros((resolution, num_inference_steps))
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
    
# pure noise（t=T，alpha_bar=0）
pure_noise_pdf = scipy.stats.norm.pdf(x_range, 0, 1).reshape(-1, 1)
theory_pdf_full = onp.concatenate([pure_noise_pdf, theory_pdf], axis=1)
num_trajectories_to_plot = min(8, trajectory.shape[1])
# colors = plt.cm.rainbow(np.linspace(0, 1, num_trajectories_to_plot))
colors = ['#f8e1b2', '#f5d26d', '#f3b052', '#f07e28', '#e2543a']

# sampling - hist
num_samples = 10000
key = jax.random.PRNGKey(42)
# noise
key, subkey = jax.random.split(key)
x = jax.random.normal(subkey, (num_samples,))
# sample
key, subkey = jax.random.split(key)
trajectory_hist = sampler.sample(subkey, x, eta=0.0)[1]
trajectory_hist = np.array(trajectory_hist)

plt.figure(figsize=(12, 6))
steps_to_plot_idx = [0, 
                     num_inference_steps//5-1, 
                     2*num_inference_steps//5-1, 
                     3*num_inference_steps//5-1, 
                     4*num_inference_steps//5-1, 
                     num_inference_steps-1]

for i, traj_idx in enumerate(steps_to_plot_idx):
    plt.subplot(2, 3, i+1)
    traj_data = trajectory_hist[traj_idx,:]
    t_current = time_list[traj_idx]        
    plt.plot(x_range, theory_pdf[:, traj_idx], 'k--', linewidth=2, label='Theory')
    plt.hist(traj_data, bins=50, density=True, alpha=0.6, color='skyblue', 
             label='Generated')
    plt.title(f"Step {traj_idx+1}",fontsize=22, pad=10)
    plt.xlim(-6, 6)
    plt.ylim(0, 0.9)
    if i>=3:
        plt.xticks([-6,-3,0,3,6])
    else:
        plt.xticks([])
    if i==0 or i==3:
        plt.yticks([0, 0.3, 0.6, 0.9])
    else:
        plt.yticks([])
    # if i == 0:
    plt.legend(fontsize=16, loc='upper left', frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'guassian_hist.pdf'),
                  bbox_inches='tight', format='pdf', dpi=600)
plt.show()


fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 8, 1], wspace=0)
# Middle x in [0, 50]
ax_main = fig.add_subplot(gs[1])
ax_main.spines['left'].set_visible(False)
ax_main.spines['right'].set_visible(False)
ax_main.imshow(theory_pdf_full, extent=[0, num_inference_steps, -6, 6], aspect='auto',
               cmap=custom_cmap, alpha=0.8, origin='lower')

for i in range(num_trajectories_to_plot):
    # xs[i] + trajectory[:, i]
    full_trajectory_x = onp.concatenate([[0], onp.arange(1, len(trajectory)+1)])
    full_trajectory_y = onp.concatenate([[xs[i]], trajectory[:, i]])
    if i==0:
        label='Initial'
    elif i<4:
        label=f'Iter.{i:02d}'
    else:
        label='Optimized'
    ax_main.plot(full_trajectory_x, full_trajectory_y, 
                color=colors[i], linewidth=3, alpha=1.0, label=label)

ax_main.set_xlabel('Denoising steps', fontsize=22)
ax_main.set_xticks([0, 10, 20, 30, 40, 50])
ax_main.set_yticks([])
ax_main.legend(fontsize=16, loc='lower left', frameon=False)

# Left
ax_left = fig.add_subplot(gs[0])
noise_dist = onp.linspace(-6, 6, 1000)
noise_pdf = scipy.stats.norm.pdf(noise_dist, 0, 1)

points = onp.array([noise_pdf, noise_dist]).T.reshape(-1, 1, 2)
segments = onp.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=custom_cmap, linewidth=6)
lc.set_array(noise_pdf)  
ax_left.add_collection(lc)

for i in range(num_trajectories_to_plot):
    ax_left.scatter(0, xs[i], color=colors[i], s=100, marker='o',zorder=5, clip_on=False)

ax_left.set_xlim(0, noise_pdf.max()*1.1)
ax_left.set_ylim(-6, 6)
ax_left.set_yticks([-6,-3,0,3,6])
ax_left.invert_xaxis()
# ax_left.set_ylabel('Value')
ax_left.set_title(r'$\mathcal{N}(0,1)$', y=-0.15, fontsize=22)
ax_left.set_xticks([])

# Right
ax_right = fig.add_subplot(gs[2])
data_dist = onp.linspace(-6, 6, 1000)
data_pdf = diffusion_pdf(1.0)

points = onp.array([data_pdf, data_dist]).T.reshape(-1, 1, 2)
segments = onp.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=custom_cmap, linewidth=6)
lc.set_array(data_pdf)  
ax_right.add_collection(lc)

for i in range(num_trajectories_to_plot):
    ax_right.scatter(0, trajectory[-1, i], color=colors[i], s=200, 
                    marker='*', edgecolors=colors[i], zorder=5, clip_on=False)
ax_right.set_xlim(0, 1.1*data_pdf.max())
ax_right.set_ylim(-6, 6)
ax_right.set_title(r'$q(x_0)$', y=-0.15, fontsize=22)
ax_right.set_xticks([])
ax_right.set_yticks([])

plt.tight_layout()
plt.show()
fig.savefig(os.path.join(save_dir, 'guassian_heatmap.pdf'),
                  bbox_inches='tight', format='pdf', dpi=600)