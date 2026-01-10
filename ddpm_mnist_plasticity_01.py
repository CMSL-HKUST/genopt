"""

Plasticity (E/sig0)

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
from fem.plasticity import prep_fem

from helper import logger
from helper.opt import Optimizer
from helper.post import beige_to_black as custom_cmap

jax.config.update("jax_enable_x64", True)

# rng seed
seed = 6

# dir & path
work_dir = f'../results/plasticity/seed_{seed}'
os.makedirs(work_dir, exist_ok=True)

#%% PREPARATION
rng_seed, sample_rng = jax.random.split(jax.random.PRNGKey(seed))
sample_shape = (1, 32, 32, 1)

# model
mesh = generate_mesh(domain_size=(1.,1.), mesh_size=(32, 32), ele_type="QUAD4")
num_cells = len(mesh.cells)
Ly = onp.max(mesh.points[:,1])
full_disps = (onp.linspace(0, 0.01, 21) * Ly)
disps = full_disps[1:]
bounds = {'Emax': 100.e3, 'Emin': 1.e3, 'sig0_max': 300, 'sig0_min': 30}
problem, fwd_pred_seq, fwd_pred = prep_fem(mesh, disps)
problem.set_material_bounds(bounds)

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

betas = [5, 10, 20, 100]
generate_samples = Sampler(beta=betas[-1])

# samples
rng_seed, data_rng = jax.random.split(rng_seed)
num_samples = (800,)
x_noise_ini_sets = jax.random.normal(data_rng, num_samples + sample_shape)

# forward problem
ptype = ['E', 'sig0'][0]

if ptype == 'E':
    
    save_dir = os.path.join(work_dir, 'E')
    
    sig0_id = 73
    sample_sig0 = generate_samples(x_noise_ini_sets[sig0_id,...])
    theta_sig04E = sample_sig0[::-1,:].reshape((-1,1), order='F')
    
    plt.imshow(sample_sig0,cmap=custom_cmap)
    plt.axis('off')
    plt.show()
    
    # obj
    obj_id, obj_num = 79, 5
    
    # initial guess
    ids = [(184,2), (313,4), (269,8)]
    
    def forward_problem_E(samples):
        theta_E = samples[::-1,:].reshape((-1,1), order='F')
        thetas = np.hstack((theta_E, theta_sig04E))
        _, stresses = fwd_pred_seq(thetas)
        return stresses
    
    forward_problem = forward_problem_E
    
elif ptype == 'sig0':
    
    save_dir = os.path.join(work_dir, 'sig0')
    
    E_id = 79
    sample_E = generate_samples(x_noise_ini_sets[E_id,...])
    theta_E4sig0 = sample_E[::-1,:].reshape((-1,1), order='F')
    
    plt.imshow(sample_E,cmap=custom_cmap)
    plt.axis('off')
    plt.show()
    
    # obj
    obj_id, obj_num = 73, 0
    
    # initial guess
    ids = [(262,3), (40,5), (618,7)]
    
    def forward_problem_sig0(samples):
        theta_sig0 = samples[::-1,:].reshape((-1,1), order='F')
        thetas = np.hstack((theta_E4sig0, theta_sig0))
        _, stresses = fwd_pred_seq(thetas)
        return stresses
    
    forward_problem = forward_problem_sig0


# target data
sample_obj = generate_samples(x_noise_ini_sets[obj_id,...])
plt.imshow(sample_obj,cmap=custom_cmap)
plt.axis('off')
plt.show()
 
stresses_target = forward_problem(sample_obj)
plt.figure(figsize=(10, 8))
plt.plot(full_disps, [0.]+stresses_target.tolist(), color='red', marker='o', markersize=8, linestyle='-')
plt.xlabel(r'Displacement of top surface [mm]', fontsize=20)
plt.ylabel(r'Volume averaged stress (y-y) [MPa]', fontsize=20)
plt.tick_params(labelsize=18)
plt.show()

# loss
num_steps = len(stresses_target)
def loss_fn(x):
    params = x.reshape(sample_shape, order='F')
    sample = generate_samples(params)
    stresses_seq = forward_problem(sample)
    loss = np.sum((stresses_seq - stresses_target)**2.)/num_steps
    return loss

#%% OPTIMIZATION
os.makedirs(save_dir, exist_ok=True)
for (ini_id, ini_num) in ids:
    
    results_file = os.path.join(save_dir, f'plast_ini_{ini_id}_obj_{obj_id}.pkl')

    if not os.path.exists(results_file):
        
        # logger
        log_file = os.path.join(save_dir, f'plast_ini_{ini_id}_obj_{obj_id}.log')
        logger.update_config(log_file=log_file, level='INFO', output='a')
        
        # initial 
        x_noise_ini = x_noise_ini_sets[ini_id,...]
        sample_ini = generate_samples(x_noise_ini)
        stresses_ini = forward_problem(sample_ini)
        
        plt.imshow(sample_ini,cmap=custom_cmap)
        plt.axis('off')
        plt.show()
        
        # curve
        plt.figure(figsize=(4,4))
        plt.plot(full_disps, [0.]+stresses_target.tolist(), label='target')
        plt.plot(full_disps, [0.]+stresses_ini.tolist(), label='initial')
        plt.legend()
        plt.show()
        
        # optimize
        optimizer = optimizer = Optimizer(method='BFGS', backend='scipy', logger=logger)
        maxits = len(betas) * [50]
        results = []
        x_tol = 1e-3
        xiter = x_noise_ini.reshape(-1, order='F')
        for beta, maxiter in zip(betas, maxits):
            logger.info('=' * 60)
            logger.info(f'Projection function slope: {beta}')
            generate_samples.beta = beta
            scipy_options = {'maxiter': maxiter, 'disp': False, 'ftol': 1e-20, 'gtol': 1e-20}
            result = optimizer.minimize(objective=loss_fn, 
                                        x0=xiter,
                                        tol = 0.2,
                                        options=scipy_options)
            results.append(result)
            xiter = result.x
            
            # result
            x_noise_opt = xiter.reshape(sample_shape, order='F')
            sample_opt = generate_samples(x_noise_opt)
            
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
        
        # optimal
        stresses_opt = forward_problem(sample_opt)
        
        # save
        data_ini = [x_noise_ini, sample_ini, stresses_ini]
        data_opt = [x_noise_opt, sample_opt, stresses_opt]
        with open(results_file, "wb") as f:
            pickle.dump([results, stresses_target, data_ini, data_opt], f)
    
    
    # Postprocessing
    with open(results_file, "rb") as f:  
        results, stresses_target, data_ini, data_opt = pickle.load(f)  
        
    x_noise_ini, sample_ini, stresses_ini = data_ini
    x_noise_opt, sample_opt, stresses_opt = data_opt
        
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
    plt.plot(full_disps, [0.]+stresses_target.tolist(), 'r-', marker='o', label='Target')
    plt.plot(full_disps, [0.]+stresses_ini.tolist(), 'g-', marker='o',label='Initial')
    plt.plot(full_disps, [0.]+stresses_opt.tolist(), 'b--', marker='o',label='Optimized')
    plt.title("Stress")
    plt.legend(frameon=False)
    plt.show()