"""

Hyperelasticity

"""
import os
import pickle

import jax
import jax.numpy as np

import numpy as onp
import matplotlib.pyplot as plt

from jax_fem import logger as jax_fem_logger
jax_fem_logger.setLevel("INFO")

from net.flax_diffusion import prep_sampler

from fem.utils import generate_mesh
from fem.hyperelasticity import prep_fem

from helper import logger
from helper.opt import Optimizer
from helper.post import beige_to_black as custom_cmap

jax.config.update("jax_enable_x64", True)

# rng seed
seed = 3

# dir & path
work_dir = f'../results/hyper/seed_{seed}'
os.makedirs(work_dir, exist_ok=True)

#%% PREPARATION
rng_seed, sample_rng = jax.random.split(jax.random.PRNGKey(seed))
sample_shape = (1, 32, 32, 1)

# forward problem
mesh = generate_mesh(domain_size=(1.,1.), mesh_size=(32, 32), ele_type="QUAD4")
Ly = onp.max(mesh.points[:,1])
disps = onp.linspace(0, 0.5, 11)[1:] * Ly
full_disps = onp.array([0.]+disps.tolist())
bounds = {'Emax': 100., 'Emin': 1.}
problem, fwd_pred_seq = prep_fem(mesh, disps)
problem.set_material_bounds(bounds)

def forward_problem(sample):
    thetas = sample[::-1,:].reshape(-1, order='F')
    energy, sols = fwd_pred_seq(thetas)
    return energy, sols

# sampler
sampler = prep_sampler('MNIST', epoch=100, num_inference_steps=10, seed=seed)
class Sampler:
    def __init__(self, beta):
        self.beta = beta
    
    def __call__(self, params):
        x_noise = params.reshape(sample_shape)
        samples = sampler.sample(sample_rng,
                          x_noise)
        
        samples = (np.tanh(self.beta * samples) + 1) / 2
        
        samples = 1. - samples[0, :, :, 0]
        return samples

betas = [2, 5, 10, 20, 100]
generate_samples = Sampler(beta=betas[-1])

# target data
num_cells = len(mesh.cells)
energy_seq_max, _ = forward_problem(1. * onp.ones(sample_shape))
energy_seq_min, _ = forward_problem(0. * onp.ones(sample_shape))

deg = 3
max_fn_coeffs = onp.polyfit(full_disps, [0.]+energy_seq_max.tolist(), deg=deg)
min_fn_coeffs = onp.polyfit(full_disps, [0.]+energy_seq_min.tolist(), deg=deg)

max_fn = lambda x: -19.880*x**3 + 51.245*x**2 + 0.472 * x
min_fn = lambda x: -0.199*x**3 + 0.512*x**2  + 0.005 * x

# initial guess
rng_seed, step_rng = jax.random.split(rng_seed)
num_samples = (500,)
x_noise_ini_sets = jax.random.normal(step_rng, num_samples + sample_shape)

ids = [(32,9), (386,3), (106,4), (155,5)]

ini_guess_list = []
for i, num in ids:
    x_noise_ini = x_noise_ini_sets[i,...]
    sample_ini = generate_samples(x_noise_ini)
    
    plt.subplot(1, 2, 1)
    plt.imshow(x_noise_ini[0,...,0], cmap=custom_cmap)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(sample_ini,cmap=custom_cmap)
    plt.axis('off')
    plt.show()
    
    energy_ini, sols_ini = forward_problem(sample_ini)
    ini_guess_list.append([x_noise_ini, sample_ini, energy_ini, sols_ini])
    logger.info(f"Initial energy: {energy_ini}")

alpha = 0.3
target_fn = lambda x: (1-alpha) * max_fn(x) + alpha * min_fn(x)
energy_target = target_fn(disps)
logger.info(f"Target energy: {energy_target}")

# objective function
num_steps = len(energy_target)
def loss_fn(x):
    params = x.reshape(sample_shape, order='F')
    sample = generate_samples(params)
    energy_seq, _ = forward_problem(sample)
    loss = np.sum((energy_seq - energy_target)**2.)/num_steps
    return loss

# curve
plt.figure(figsize=(4,4))
plt.plot(full_disps, [0.]+energy_seq_max.tolist(), label='max')
plt.plot(full_disps, max_fn(full_disps), label='max_fn', ls='--')
plt.plot(full_disps, [0.]+energy_seq_min.tolist(), label='min')
plt.plot(full_disps, min_fn(full_disps), label='min_fn', ls='--')
plt.plot(full_disps, target_fn(full_disps), marker='o', label='target')
for i in range(len(ids)):
    energy_ini = ini_guess_list[i][-2]
    plt.plot(full_disps, [0.]+energy_ini.tolist(), label=f'initial_{i}')
plt.legend()
plt.show()

#%% OPTIMIZATION
save_dir = os.path.join(work_dir, f'{alpha}')
os.makedirs(save_dir, exist_ok=True)
for i, (ind, num) in enumerate(ids):
    
    results_file = os.path.join(save_dir, f'hyper_ini_{ind}_obj_{alpha}.pkl')

    if not os.path.exists(results_file):
        
        # logger
        log_file = os.path.join(save_dir, f'hyper_ini_{ind}_obj_{alpha}.log')
        logger.update_config(log_file=log_file, level='INFO', output='a')
        
        # initial
        x_noise_ini, sample_ini, energy_ini, sols_ini = ini_guess_list[i]   
        loss_ini = np.sum((energy_ini - energy_target)**2.)/num_steps
        plt.figure(figsize=(4,4))
        plt.plot([0.]+disps.tolist(), [0.]+energy_target.tolist(), 'r-', marker='o', label='Target')
        plt.plot([0.]+disps.tolist(), [0.]+energy_ini.tolist(), 'g-', marker='o',label='Initial')
        plt.legend()
        plt.show()
        
        # optimize
        method = ['BFGS', 'L-BFGS-B'][0]
        optimizer = Optimizer(method=method, backend='scipy', logger=logger)
        maxits = len(betas) * [30]
        results = []
        x_tol = 1e-3
        xiter = x_noise_ini.reshape(-1, order='F')
        for beta, maxiter in zip(betas, maxits):
            logger.info('=' * 60)
            logger.info(f'Projection function slope: {beta}')
            generate_samples.beta = beta
            scipy_options = {'maxiter': maxiter, 'maxls':100, 'disp': False, 'ftol': 1e-20, 'gtol': 1e-20}
            result = optimizer.minimize(objective=loss_fn, 
                                        x0=xiter,
                                        tol=1e-3,
                                        bounds=None if method=='BFGS' else [(-3,3),]*xiter.size,
                                        options=scipy_options)
            results.append(result)
            xiter = result.x
            
            # result
            x_noise_opt = xiter.reshape(sample_shape, order='F')
            sample_opt = generate_samples(x_noise_opt)
            energy_opt, sols_opt = forward_problem(sample_opt)
            
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(x_noise_opt[0,:,:,0], cmap='gray')
            plt.title("Optimized noise")
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(sample_opt, cmap=custom_cmap)
            plt.title("Optimized sample")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            # gaussian
            logger.info(f'Noise means: {float(onp.mean(x_noise_opt)):.4e} stds: {float(onp.std(x_noise_opt)):.4e}')
            
            # 0-1
            dense_ratio = onp.mean((sample_opt < x_tol) | (sample_opt > 1 - x_tol))
            logger.info(f'0-1 ratio: {dense_ratio}')
            
        # save
        data_ini = [x_noise_ini, sample_ini, energy_ini, sols_ini]
        data_opt = [x_noise_opt, sample_opt, energy_opt, sols_opt]
        with open(results_file, "wb") as f:
            pickle.dump([results, energy_target, data_ini, data_opt], f)
    
    # Postprocessing
    with open(results_file, "rb") as f:  
        results, datat_target, data_ini, data_opt = pickle.load(f)  
    
    energy_target = datat_target
    x_noise_ini, sample_ini, energy_ini, sols_ini = data_ini
    x_noise_opt, sample_opt, energy_opt, sols_opt = data_opt
  
    # loss
    funs = results[-1].history["funs"]
    plt.figure(figsize=(10, 8))
    plt.plot(onp.arange(len(funs)), funs, linestyle='-', linewidth=2, color='black')
    plt.xlabel(r"Optimization step", fontsize=20)
    plt.ylabel(r"Objective value", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.tick_params(labelsize=20)
    plt.show()
    
    # results
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(x_noise_ini[0,:,:,0], cmap='gray')
    plt.title("Initial noise")
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(sample_ini, cmap=custom_cmap)
    plt.title("Initial sample")
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(x_noise_opt[0,:,:,0], cmap='gray')
    plt.title("Optimized noise")
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(sample_opt, cmap=custom_cmap)
    plt.title("Optimized sample")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # curve
    plt.figure(figsize=(6, 6))
    plt.plot([0.]+disps.tolist(), [0.]+energy_target.tolist(), 'r-', marker='o', label='Target')
    plt.plot([0.]+disps.tolist(), [0.]+energy_ini.tolist(), 'g-', marker='o',label='Initial')
    plt.plot([0.]+disps.tolist(), [0.]+energy_opt.tolist(), 'b--', marker='o',label='Optimized')
    plt.title("Strain energy")
    plt.legend(frameon=False)
    plt.show()
    
    # deformation
    plt.figure(figsize=(6, 6))
    X, Y = np.meshgrid(np.linspace(0, 1, 33), np.linspace(0, 1, 33))
    Xd = X + sols_opt[-1][:,0].reshape(33,33,order='F')
    Yd = Y + sols_opt[-1][:,1].reshape(33,33,order='F')
    plt.pcolormesh(Xd, Yd, sample_opt[::-1,:], cmap=custom_cmap)
    plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color='#7f7f7f',
         ls='--', linewidth=2)
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    