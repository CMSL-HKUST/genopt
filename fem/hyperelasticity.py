"""Neo-Hookean hyperelasticity"""

import jax
import jax.numpy as np

import numpy as onp

from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper


class Hyperelasticity(Problem):
    
    def custom_init(self):
        self.fe = self.fes[0]
        
    def set_material_bounds(self, bounds):
        self.Emax, self.Emin = bounds['Emax'], bounds['Emin']
        
    def psi(self, F2d, theta):
        
        penal = 1
        E = (self.Emax-self.Emin)*theta**penal + self.Emin
        
        nu = 0.3
        mu = E / (2. * (1. + nu))
        kappa = E / (3. * (1. - 2. * nu))
        
        # transform F from 2D to 3D
        F = np.array([[F2d[0,0], F2d[0,1], 0.],
                      [F2d[1,0], F2d[1,1], 0.],
                      [0.,       0.,       1.]
                     ])
        
        J = np.linalg.det(F)
        Jinv = J**(-2. / 3.)
        I1 = np.trace(F.T @ F)
        energy = (mu / 2.) * (Jinv * I1 - 3.) + (kappa / 2.) * (J - 1.)**2.
        return energy
    
    def get_tensor_map(self):

        P_fn = jax.grad(self.psi)

        def first_PK_stress(u_grad, theta):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F, theta)
            return P

        return first_PK_stress

    def set_params(self, params):
        thetas = params
        self.internal_vars = [thetas]
        
    def compute_strain_energy(self, sol_list, thetas):
        sol = sol_list[0]
        u_grads = self.fe.sol_to_grad(sol)
        F = u_grads + np.eye(self.fe.dim)
        energy_densities = jax.vmap(jax.vmap(self.psi))(F, thetas)
        energy = np.sum(energy_densities * self.fe.JxW)
        return energy


def prep_fem(mesh, disps):
    
    Ly = onp.max(mesh.points[:,1])
    
    # BCs
    def bottom(point):
        return np.isclose(point[1], 0., atol=1e-5)
    
    def top(point):
        return np.isclose(point[1], Ly, atol=1e-5)
    
    def dirichlet_val(point):
        return 0.
    
    def get_dirichlet_fn(disp):
        def dirichlet_fn(point):
            return disp
        return dirichlet_fn
    
    dirichlet_bc_info = [[bottom, bottom, top], [0, 1, 1], [dirichlet_val, dirichlet_val, get_dirichlet_fn(disps[0])]]
    
    # problem
    problem = Hyperelasticity(mesh, vec=2, dim=2, ele_type='QUAD4', dirichlet_bc_info=dirichlet_bc_info)
    fwd_pred = ad_wrapper(problem, solver_options={'umfpack_solver': {}, 'line_search_flag':True}, adjoint_solver_options={'umfpack_solver': {}})
    
    # fwd
    def fwd_pred_seq(thetas):
        thetas = np.repeat(thetas[:, None], problem.fe.num_quads, axis=1)
        energy_seq = []
        sol_lists = []
        for disp in disps:
            dirichlet_bc_info[-1][-1] = get_dirichlet_fn(disp)
            problem.fe.update_Dirichlet_boundary_conditions(dirichlet_bc_info)
            sol_list = fwd_pred(thetas)
            sol_lists.append(sol_list[0])
            energy = problem.compute_strain_energy(sol_list, thetas)
            energy_seq.append(energy)
        return np.array(energy_seq), sol_lists
    
    return problem, fwd_pred_seq
