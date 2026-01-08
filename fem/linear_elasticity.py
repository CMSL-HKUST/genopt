"""Linear elasticity (plane stress)"""

import jax
import jax.numpy as np

from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper


class Elasticity(Problem):
    
    def custom_init(self):
        self.fe = self.fes[0]

    def get_tensor_map(self):
        def stress(u_grad, theta):
            # Plane stress assumption
            # Reference: https://en.wikipedia.org/wiki/Hooke%27s_law
            
            Emax = 5.
            Emin = 1.
            penal = 1
            E = (Emax-Emin)*theta**penal + Emin
            nu = 0.3
            
            epsilon = 0.5*(u_grad + u_grad.T)
            eps11 = epsilon[0, 0]
            eps22 = epsilon[1, 1]
            eps12 = epsilon[0, 1]
            sig11 = E/(1 + nu)/(1 - nu)*(eps11 + nu*eps22) 
            sig22 = E/(1 + nu)/(1 - nu)*(nu*eps11 + eps22)
            sig12 = E/(1 + nu)*eps12
            sigma = np.array([[sig11, sig12], [sig12, sig22]])
            
            return sigma
        return stress

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([1., 0.])
        return [surface_map]

    def set_params(self, thetas):
        thetas = np.repeat(thetas[:, None], self.fe.num_quads, axis=1)
        self.internal_vars = [thetas]

    def compute_compliance(self, sol):
        # Surface integral
        boundary_inds = self.boundary_inds_list[0]
        _, nanson_scale = self.fe.get_face_shape_grads(boundary_inds)
        # (num_selected_faces, 1, num_nodes, vec) * # (num_selected_faces, num_face_quads, num_nodes, 1)    
        u_face = sol[self.fe.cells][boundary_inds[:, 0]][:, None, :, :] * self.fe.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
        u_face = np.sum(u_face, axis=2) # (num_selected_faces, num_face_quads, vec)
        # (num_selected_faces, num_face_quads, dim)
        subset_quad_points = self.physical_surface_quad_points[0]
        neumann_fn = self.get_surface_maps()[0]
        traction = -jax.vmap(jax.vmap(neumann_fn))(u_face, subset_quad_points) # (num_selected_faces, num_face_quads, vec)
        val = np.sum(traction * u_face * nanson_scale[:, :, None])
        return val


def prep_fem(mesh):
    
    Lx = np.max(mesh.points[:,0])
    
    # BCs
    def fixed_location(point):
        return np.isclose(point[0], 0., atol=1e-5)
        
    def load_location(point):
        return np.isclose(point[0], Lx, atol=1e-5)

    def dirichlet_val(point):
        return 0.

    dirichlet_bc_info = [[fixed_location]*2, [0, 1], [dirichlet_val]*2]
    location_fns = [load_location]
    
    # problem
    problem = Elasticity(mesh, vec=2, dim=2, ele_type='QUAD4', dirichlet_bc_info=dirichlet_bc_info, location_fns=location_fns)
    fwd_pred = ad_wrapper(problem, solver_options={'umfpack_solver': {}}, adjoint_solver_options={'umfpack_solver': {}})
    
    return problem, fwd_pred
