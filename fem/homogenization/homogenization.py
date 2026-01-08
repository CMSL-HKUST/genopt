"""

FEM-based homogenization

"""

import jax
import jax.numpy as np

import numpy as onp

from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper

from .helper import setup_boundary_conditions, periodic_boundary_conditions

jax.config.update("jax_enable_x64", True)
    
# weak form
class Composite_Elasticity(Problem):
    
    def custom_init(self, P_mat):
        self.fe = self.fes[0]
        self.P_mat = P_mat
        
    def set_material_params(self, params):
        self.E_max, self.E_min = params['E_max'], params['E_min']
        self.nu_max, self.nu_min = params['nu_max'], params['nu_min']
        

    def get_tensor_map(self):
        def stress(u_grad, macro_strain, theta):
            
            penal = 1
            
            E = (self.E_max-self.E_min)*theta**penal + self.E_min
            nu = (self.nu_max-self.nu_min)*theta**penal + self.nu_min
            
            mu = E/(2.*(1.+nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            # plane stress
            lmbda = 2*mu*lmbda/(lmbda+2*mu)
            
            eps = 0.5 * (u_grad + u_grad.T)
            Eps = macro_strain + eps
            sigma = lmbda * np.trace(Eps) * np.eye(2) + 2 * mu * Eps
            return sigma
        return stress

    def set_params(self, params):
        # Set params
        self.internal_vars = params
        
    def compute_avg_stress(self, sol, params):
        # Compute volume averaged stress.
        u_grads = self.fe.sol_to_grad(sol)  # (num_cells, num_quads, 2, 2)
        stress_fn = self.get_tensor_map()
        vmap_stress = jax.vmap(jax.vmap(stress_fn))  
        sigmas = vmap_stress(u_grads, *params)
        
        # (num_cells*num_quads, vec, dim) * (num_cells*num_quads, 1, 1) -> (vec, dim)
        sigma = np.sum(sigmas.reshape(-1, self.fe.vec, self.dim) * self.fe.JxW.reshape(-1)[:, None, None], 0)
        vol = np.sum(self.fe.JxW)
        avg_sigma = sigma/vol
        
        return avg_sigma
    

def prep_fem(mesh, ele_type='QUAD4'):   
    
    # preparation
    x_max, y_max = onp.max(mesh.points, axis=0)
    x_min, y_min = onp.min(mesh.points, axis=0)
    vertices = onp.array([[x_min,y_min],
                          [x_max,y_min],
                          [x_max,y_max],
                          [x_min,y_max]])
    
    dirichlet_bc_info, periodic_bc_info = setup_boundary_conditions(vertices)
    P_mat = periodic_boundary_conditions(periodic_bc_info, mesh, vec=2, dirichlet_bc_info=dirichlet_bc_info)
    problem = Composite_Elasticity(mesh, 
                                   vec=2, 
                                   dim=2,
                                   ele_type=ele_type,
                                   dirichlet_bc_info=dirichlet_bc_info,
                                   additional_info=(P_mat,))
    
    fwd_pred = ad_wrapper(problem, solver_options={'umfpack_solver': {}}, 
                                   adjoint_solver_options={'umfpack_solver': {}})
    
    macro_strains = [onp.array([[1., 0], [0, 0]]),
                     onp.array([[0, 0], [0, 1.]]),
                     onp.array([[0, 0.5], [0.5, 0]])]
    
    # compute effective C matrix
    def compute_C_hom(thetas):
        thetas = np.repeat(thetas[:, None], problem.fe.num_quads, axis=1)
        C_hom = []
        for macro_strain in macro_strains:
            macro_strain_field = np.repeat(np.repeat(macro_strain[None,None,:,:],
                                            repeats=problem.fe.num_cells,
                                            axis=0),
                                            repeats=problem.fe.num_quads,
                                            axis=1)
            params = [macro_strain_field, thetas]
            sol_list = fwd_pred(params)
            avg_sigma = problem.compute_avg_stress(sol_list[0], params)
            sigma_voigt = np.array([avg_sigma[0,0], avg_sigma[1,1], avg_sigma[0,1]])
            C_hom.append(sigma_voigt.reshape(-1,1))
        return np.hstack(C_hom)
    
    return problem, compute_C_hom

def compute_E_thetas(C, thetas=np.linspace(0, 2*onp.pi, num=100)):
    S = np.linalg.inv(C)
    c, s = np.cos(thetas), np.sin(thetas)
    c2, s2, c3, s3, c4, s4 = c*c, s*s, c*c*c, s*s*s, c*c*c*c, s*s*s*s
    S_11_rot = (c4*S[0,0] + s4*S[1,1] + 2*c2*s2*S[0,1] + 
                c2*s2*S[2,2] + 2*c3*s*S[0,2] + 2*c*s3*S[1,2])
    return 1.0 / S_11_rot

def compute_nu_thetas(C, thetas=np.linspace(0, 2*np.pi, num=100)):
    S = np.linalg.inv(C)
    c, s = np.cos(thetas), np.sin(thetas)
    c2, s2, c3, s3, c4, s4 = c*c, s*s, c*c*c, s*s*s, c*c*c*c, s*s*s*s
    
    S_11_rot = (c4*S[0,0] + s4*S[1,1] + 2*c2*s2*S[0,1] + 
                c2*s2*S[2,2] + 2*c3*s*S[0,2] + 2*c*s3*S[1,2])
                
    S_12_rot = (c2*s2*(S[0,0] + S[1,1] - S[2,2]) + 
                (c4 + s4)*S[0,1] + 
                (c*s3 - c3*s)*S[0,2] + (c3*s - c*s3)*S[1,2])
                
    return -S_12_rot / S_11_rot