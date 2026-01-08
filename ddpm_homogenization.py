"""

Homogenization

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import pickle

import jax
import jax.numpy as np

import numpy as onp
import matplotlib.pyplot as plt

from jax_fem import logger as jax_fem_logger
jax_fem_logger.setLevel("INFO")

from net.flax_diffusion import prep_sampler

from fem.utils import generate_mesh
from fem.homogenization import prep_fem, compute_E_thetas, compute_nu_thetas

from helper import logger
from helper.opt import Optimizer
from helper.post import beige_to_black as custom_cmap

# dataset
dataset = ['MNIST', 'MICRO_GRAY'][1]

# rng seed
seed = 9

# dir & path
work_dir = f'../results/homogenization/{dataset}/seed_{seed}'

jax.config.update("jax_enable_x64", True)

#%% PREPARATION
rng, sample_rng = jax.random.split(jax.random.PRNGKey(seed))
sample_shape = (1, 32, 32, 1)

# forward problem
mesh = generate_mesh(domain_size=(1.,1.), mesh_size=(32, 32), ele_type="QUAD4")
problem, fwd_pred = prep_fem(mesh)
mat_params = {'E_max':100., 'nu_max':0.3, 'E_min':1., 'nu_min':0.3}
problem.set_material_params(mat_params)
 
def forward_problem(sample):
    thetas = sample[::-1,:].reshape(-1, order='F')
    Chom = fwd_pred(thetas)
    return Chom

# sampler
sampler = prep_sampler(dataset, epoch=100, num_inference_steps=10, seed=seed)
class Sampler:
    def __init__(self, beta):
        self.beta = beta
    
    def __call__(self, params):
        x_noise = params.reshape(sample_shape)
        samples = sampler.sample(sample_rng,
                          x_noise)
        
        samples = (np.tanh(self.beta * samples) + 1) / 2
        
        # samples = 1. - samples[0, :, :, 0]
        samples = samples[0, :, :, 0]
        return samples

betas = [5, 10, 20, 80] if dataset=='MNIST' else [5, 10, 20, 100]
generate_samples = Sampler(beta=betas[-1])

# initial guess
rng_seed, step_rng = jax.random.split(rng)
num_samples = (2000,)
x_noise_ini_sets = jax.random.normal(step_rng, num_samples + sample_shape)

# target data
if dataset == 'MNIST':
    task_id = [0, 1, 2, 3][1]
    if task_id == 0:
        # anisotropic / triclinic in 2D sense
        ids = [(210,9), (272,8)]
        Chom_target = onp.array([[50, 12, -3],
                                 [12, 60, -3],
                                 [-3, -3, 15]])
    elif task_id == 1:
        # monoclinic
        ids = [(72,3), (898,6), (1930,6)]
        Chom_target = onp.array([[50, 12,  0],
                                 [12, 60, -3],
                                 [0,  -3,  15]])
    elif task_id == 2:
        # orthotropic
        ids = [(78,4), (51, 7)]
        Chom_target = onp.array([[50, 12, 0],
                                 [12, 60, 0],
                                 [0,  0,  15]])
    elif task_id == 3:
        # tetragornal
        ids = [(246,5), (164, 0)]
        Chom_target = onp.array([[60, 12, 0],
                                 [12, 60, 0],
                                 [0,  0,  15]])

if dataset == 'MICRO_GRAY':
    task_id = [0, 1][0]
    if task_id == 0:
        # orthotropic
        ids = [(28,0), (48,0), (71,0), (185,0), (188,0)]
        Chom_target = onp.array([[50, 15, 0],
                                 [15, 70, 0],
                                 [0,  0,  15]])
    elif task_id == 1:
        # tetragornal
        ids = [(133,0), (230,0), (26,0), (205, 0), (96,0)]
        Chom_target = onp.array([[50, 15, 0],
                                 [15, 50, 0],
                                 [0,  0,  15]])

datas_ini = []
for i, num in ids:
    # data
    x_noise_ini = x_noise_ini_sets[i,...]
    sample_ini = generate_samples(x_noise_ini)
    Chom_ini = forward_problem(sample_ini)
    datas_ini.append([x_noise_ini, sample_ini, Chom_ini])
    print(f'Chom for No,{i} initial guess:\n{Chom_ini}')
    # fig
    plt.imshow(sample_ini, cmap=custom_cmap)
    plt.axis('off')
    plt.show()
    plt.close()

def compute_loss(Chom):
    return np.sum((Chom-Chom_target)**2)

# loss
def loss_fn(x):
    params = x.reshape(sample_shape,order='F')
    sample = generate_samples(params)
    Chom = forward_problem(sample)
    loss = compute_loss(Chom)
    return loss

# polar
thetas = onp.linspace(0, 2*onp.pi, num=360)
plt.subplot(1,2,1, projection='polar')
for ind, (i, num) in enumerate(ids):
    E_vec = compute_E_thetas(datas_ini[ind][2], thetas)
    plt.polar(thetas, E_vec, ls='-', label=f'{i}-{num}')
plt.polar(thetas, compute_nu_thetas(Chom_target, thetas), ls='--', label='target')   
plt.legend()
plt.subplot(1,2,2, projection='polar')
for ind, (i, num) in enumerate(ids):
    nu_vec = compute_nu_thetas(datas_ini[ind][2], thetas)
    plt.polar(thetas, nu_vec, ls='-', label=f'{i}-{num}')
plt.polar(thetas, compute_nu_thetas(Chom_target, thetas), ls='--', label='target')  
plt.ylim([0,0.5])  
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

#%% OPTIMIZATION
for i, (ind, num) in enumerate(ids):
    
    save_dir = os.path.join(work_dir, f'task_{task_id}')
    os.makedirs(save_dir, exist_ok=True)
    
    result_file = os.path.join(save_dir, f'homo_obj_{task_id}_ini_{ind}.pkl')
    
    if not os.path.exists(result_file):
        # logger
        log_file = os.path.join(save_dir, f'homo_obj_{task_id}_ini_{ind}.log')
        logger.update_config(log_file=log_file, level='INFO', output='a', mode='w')
        
        # initial
        x_noise_ini, sample_ini, Chom_ini = datas_ini[i]
        
        plt.polar(thetas, compute_E_thetas(Chom_ini,thetas), label='initial')
        plt.polar(thetas, compute_E_thetas(Chom_target,thetas), label='target')
        plt.legend()
        plt.show()

        # optimize
        optimizer = Optimizer(method='BFGS', backend='scipy', logger=logger)
        maxits = len(betas) * [100]
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
                                        tol= 1e-2,
                                        options=scipy_options)
            results.append(result)
            xiter = result.x
            
            # result
            x_noise_opt = xiter.reshape(sample_shape, order='F')
            sample_opt = generate_samples(x_noise_opt)
            Chom_opt = forward_problem(sample_opt)
            Etheta_opt = compute_E_thetas(Chom_opt, thetas)
            
            print(Chom_opt)
            
            plt.figure(figsize=(8, 8))
            plt.subplot(2, 2, 1)
            plt.imshow(x_noise_ini[0,:,:,0], cmap='gray')
            plt.title("Initial noise",fontsize=20)
            plt.axis('off')
            plt.subplot(2, 2, 2)
            plt.imshow(sample_ini, cmap=custom_cmap)
            plt.title("Initial sample",fontsize=20)
            plt.axis('off')
            plt.subplot(2, 2, 3)
            plt.imshow(x_noise_opt[0,:,:,0], cmap='gray')
            plt.title("Optimized noise",fontsize=20)
            plt.axis('off')
            plt.subplot(2, 2, 4)
            plt.imshow(sample_opt, cmap=custom_cmap)
            plt.title("Optimized sample",fontsize=20)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            # gaussian
            logger.info(f'Noise means: {float(onp.mean(x_noise_opt)):.4e} stds: {float(onp.std(x_noise_opt)):.4e}')
    
            # 0-1
            dense_ratio = onp.mean((sample_opt < x_tol) | (sample_opt > 1 - x_tol))
            logger.info(f'0-1 ratio: {dense_ratio}')
            
        # save
        data_ini = [x_noise_ini, sample_ini, Chom_ini]
        data_opt = [x_noise_opt, sample_opt, Chom_opt]
        with open(result_file, "wb") as f:
            pickle.dump([results, data_ini, data_opt], f)
    
    # Postprocessing
    with open(result_file, "rb") as f:  
        results, data_ini, data_opt = pickle.load(f)  
        
    x_noise_ini, sample_ini, Chom_ini = data_ini
    x_noise_opt, sample_opt, Chom_opt = data_opt
        
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
    plt.title("Initial noise",fontsize=20)
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(sample_ini, cmap=custom_cmap)
    plt.title("Initial sample",fontsize=20)
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(x_noise_opt[0,:,:,0], cmap='gray')
    plt.title("Optimized noise",fontsize=20)
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(sample_opt, cmap=custom_cmap)
    plt.title("Optimized sample",fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # elastic surface
    plt.figure(figsize=(8, 8))
    plt.subplot(projection='polar')
    plt.plot(thetas, compute_E_thetas(Chom_ini, thetas), label='initial')
    plt.plot(thetas, compute_E_thetas(Chom_target, thetas), label='Target')
    plt.plot(thetas, compute_E_thetas(Chom_opt, thetas), ls='--', label='optimized')
    plt.legend(fontsize=20)
    plt.show()
    
    print(Chom_opt)
    