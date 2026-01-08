"""useful utils for jax_fem"""

import jax.numpy as np

import numpy as onp
import meshio

from jax_fem.generate_mesh import rectangle_mesh, box_mesh_gmsh, Mesh
from jax_fem.generate_mesh import get_meshio_cell_type


def generate_mesh(domain_size, mesh_size, ele_type):
    
    """generate mesh, supporting QUAD4, QUAD8, and HEX8"""
    
    cell_type = get_meshio_cell_type(ele_type)
    dim = len(domain_size)
    
    if dim == 2:
        # 2D
        Lx, Ly = domain_size
        Nx, Ny = mesh_size
        if ele_type == 'QUAD4':
            meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
        elif ele_type == 'QUAD8':
            meshio_mesh = rectangle_mesh_quad8(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    elif dim == 3:
        # 3D
        Lx, Ly, Lz = domain_size
        Nx, Ny, Nz = mesh_size
        if ele_type == 'HEX8':
            meshio_mesh = box_mesh_gmsh(Nx=Nx,
                                        Ny=Ny,
                                        Nz=Nz,
                                        domain_x=Lx,
                                        domain_y=Ly,
                                        domain_z=Lz,
                                        data_dir='./',
                                        ele_type=ele_type) # no need to save?
    
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type], ele_type=ele_type)
    
    return mesh


def rectangle_mesh_quad8(Nx, Ny, domain_x, domain_y):
    """Generate QUAD8 mesh.

    Parameters
    ----------
    Nx : int
        Number of elements along x-axis.
    Ny : int
        Number of elements along y-axis.
    domain_x : float
        Length of side along x-axis.
    domain_y : float
        Length of side along y-axis.
    """
    dim = 2
    
    # Total nodes: corners + edge midpoints
    # Corners: (Nx+1) * (Ny+1)
    # Horizontal edges: Nx * (Ny+1)
    # Vertical edges: (Nx+1) * Ny
    
    x_nodes = onp.linspace(0, domain_x, 2*Nx + 1)
    y_nodes = onp.linspace(0, domain_y, 2*Ny + 1)
    
    xv, yv = onp.meshgrid(x_nodes, y_nodes, indexing='ij')
    points_xy = onp.stack((xv, yv), axis=dim)
    points = points_xy.reshape(-1, dim)
    
    points_inds = onp.arange(len(points))
    points_inds_xy = points_inds.reshape(2*Nx + 1, 2*Ny + 1)
    
    # QUAD8 standard node numbering:
    # 3---6---2
    # |       |
    # 7       5
    # |       |
    # 0---4---1
    
    cells = []
    for ex in range(Nx):
        for ey in range(Ny):
            i = 2 * ex
            j = 2 * ey
            
            n0 = points_inds_xy[i,     j]
            n1 = points_inds_xy[i + 2, j]
            n2 = points_inds_xy[i + 2, j + 2]
            n3 = points_inds_xy[i,     j + 2]
            n4 = points_inds_xy[i + 1, j]
            n5 = points_inds_xy[i + 2, j + 1]
            n6 = points_inds_xy[i + 1, j + 2]
            n7 = points_inds_xy[i,     j + 1]
            
            cell = [n0, n1, n2, n3, n4, n5, n6, n7]
            cells.append(cell)
    
    cells = onp.array(cells, dtype=onp.int32)
    
    out_mesh = meshio.Mesh(points=points, cells={'quad8': cells})
    return out_mesh

