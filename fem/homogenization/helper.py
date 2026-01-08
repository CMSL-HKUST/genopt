"""

Customized utils for PBC

"""

import jax
import jax.numpy as np
import numpy as onp
import scipy.sparse


def setup_boundary_conditions(vertices, EPS=1e-5):
    
    # ========== Dirichlet ========== 
    bc = create_boundary_conditions(vertices, EPS)
    
    def dirichlet_val(point):
        return 0.
    
    dirichlet_bc_info = [
        [bc['corners']['A'], bc['corners']['A']],  # fix x & y of A
        [0, 1],                                    # x & y
        [dirichlet_val, dirichlet_val]]

    # ========== Periodic ========== 
    # edge
    edge_bc = setup_periodic_edges(bc)
    
    # corner
    corner_bc = setup_periodic_corners(bc)
    
    # combine
    periodic_bc_info = [
        edge_bc[0] + corner_bc[0],      # location_fns_A
        edge_bc[1] + corner_bc[1],      # location_fns_B
        edge_bc[2] + corner_bc[2],      # mappings
        edge_bc[3] + corner_bc[3]       # vecs
    ]
    
    return dirichlet_bc_info, periodic_bc_info


def create_boundary_conditions(vertices, EPS=1e-5):
    """
    Geometry definition for 2D PBC with arbitrary quadrilateral
    
    Parameters:
    -----------
    vertices : np.array of shape (4, 2)
        Coordinates of the four vertices in counter-clockwise order:
        [0] - bottom-left (vertex A)  - MASTER
        [1] - bottom-right (vertex B) - slave to A
        [2] - top-right (vertex C)    - slave to A
        [3] - top-left (vertex D)     - slave to A
    EPS : float, optional
        Tolerance for numerical comparisons (default: 1e-5)
    
    Returns:
    --------
    dict : Dictionary containing corner, edge, and mapping functions
    """
    
    # Extract vertices for clarity
    v0, v1, v2, v3 = vertices[0], vertices[1], vertices[2], vertices[3]
    
    # Compute edge vectors
    edge_vec_bottom = v1 - v0  # bottom edge vector (v0 -> v1)
    edge_vec_top = v2 - v3     # top edge vector (v3 -> v2)
    edge_vec_left = v3 - v0    # left edge vector (v0 -> v3)
    edge_vec_right = v2 - v1   # right edge vector (v1 -> v2)
    
    # ========== Helper Functions ==========
    def corner_fn_helper(vertex):
        def is_on_corner(point):
            """Check if point is at a specific vertex"""
            return (np.isclose(point[0], vertex[0], atol=EPS) & 
                    np.isclose(point[1], vertex[1], atol=EPS))
        return is_on_corner
    
    def edge_fn_helper(start_vertex, edge_vector):
        """
        Check if point is on an edge (excluding endpoints)

        start_vertex : array
            Starting vertex of the edge
            
        edge_vector : array
            Vector from start to end of the edge
        """
        def is_on_edge(point):
            # Check collinearity using cross product: (point - start) × edge_vector = 0
            diff = point - start_vertex
            cross = diff[0] * edge_vector[1] - diff[1] * edge_vector[0]
            flag_on_line = np.isclose(cross, 0., atol=EPS)
            
            # Check if point is strictly between start and end vertices
            # Parameter t: point = start + t * edge_vector, t ∈ (0, 1)
            # Use the larger component to avoid division by near-zero
            if np.abs(edge_vector[0]) > np.abs(edge_vector[1]):
                t = diff[0] / edge_vector[0]
            else:
                t = diff[1] / edge_vector[1]
            flag_interior = np.logical_and(t > EPS, t < 1. - EPS)
            
            return np.logical_and(flag_on_line, flag_interior)
        return is_on_edge
    
    # ========== Corner ==========
    # A - vertex 0 - MASTER (can be free or fixed via Dirichlet BC)
    # B - vertex 1 - slave to A
    # C - vertex 2 - slave to A
    # D - vertex 3 - slave to A
    corner_A, corner_B, corner_C, corner_D = [corner_fn_helper(v) for v in [v0, v1, v2, v3]]
    
    
    # ========== Edge ==========

    # Left edge: from v0 to v3
    edge_left = edge_fn_helper(v0, edge_vec_left)
    
    # Right edge: from v1 to v2
    edge_right = edge_fn_helper(v1, edge_vec_right)

    # Bottom edge: from v0 to v1
    edge_bottom = edge_fn_helper(v0, edge_vec_bottom)
    
    # Top edge: from v3 to v2
    edge_top = edge_fn_helper(v3, edge_vec_top)
    
    
    # ========== Mapping ==========
    def mapping_x(point_A):
        # Map from left edge to right edge
        # Translation vector from v0 to v1
        return point_A + edge_vec_bottom
    
    def mapping_y(point_A):
        # Map from bottom edge to top edge
        # Translation vector from v0 to v3
        return point_A + edge_vec_left
    
    def mapping_A_to_B(point_A):
        """
        Map from point A (vertex 0) as master to point B (vertex 1) as slave
        Translation: v0 -> v1
        """
        return point_A + edge_vec_bottom
    
    def mapping_A_to_C(point_A):
        """
        Map from point A (vertex 0) as master to point C (vertex 2) as slave
        Translation: v0 -> v2
        """
        return point_A + (v2 - v0)
    
    def mapping_A_to_D(point_A):
        """
        Map from point A (vertex 0) as master to point D (vertex 3) as slave
        Translation: v0 -> v3
        """
        return point_A + edge_vec_left
    
    return {
        'corners': {
            'A': corner_A, 'B': corner_B, 'C': corner_C, 'D': corner_D
        },
        'edges': {
            'left': edge_left, 'right': edge_right, 
            'bottom': edge_bottom, 'top': edge_top
        },
        'mappings': {
            'x': mapping_x, 'y': mapping_y, 
            'A_to_B': mapping_A_to_B,
            'A_to_C': mapping_A_to_C,
            'A_to_D': mapping_A_to_D
        }
    }


def setup_periodic_edges(bc):
    """
    PBC for edges (except for corners)
    """
    edges = bc['edges']
    mappings = bc['mappings']
    
    location_fns_A = []
    location_fns_B = []
    mappings_list = []
    vecs = []
    
    # PBC for edges：left ↔ right, bottom ↔ top
    edge_pairs = [
        ('left', 'right', 'x'),   # Left → Right (x)
        ('left', 'right', 'x'),   # Left → Right (x)  
        ('bottom', 'top', 'y'),   # Bottom → Top (y)
        ('bottom', 'top', 'y'),   # Bottom → Top (y)
    ]
    
    dofs = [0, 1, 0, 1]  # x,y,x,y
    
    for i, (edge_A, edge_B, direction) in enumerate(edge_pairs):
        location_fns_A.append(edges[edge_A])
        location_fns_B.append(edges[edge_B])
        mappings_list.append(mappings[direction])
        vecs.append(dofs[i])
    
    return [location_fns_A, location_fns_B, mappings_list, vecs]


def setup_periodic_corners(bc):
    """
    PBC for corners
    Constraints: A is MASTER, B/C/D are all slaves to A
    """
    corners = bc['corners']
    mappings = bc['mappings']
    
    location_fns_A = []
    location_fns_B = []
    mappings_list = []
    vecs = []
    
    # All corners (B, C, D) are slaves to corner A
    corner_constraints = [
        ('A', 'B', 'A_to_B', 0),  # A_x is master, B_x is slave
        ('A', 'B', 'A_to_B', 1),  # A_y is master, B_y is slave
        ('A', 'C', 'A_to_C', 0),  # A_x is master, C_x is slave
        ('A', 'C', 'A_to_C', 1),  # A_y is master, C_y is slave
        ('A', 'D', 'A_to_D', 0),  # A_x is master, D_x is slave
        ('A', 'D', 'A_to_D', 1),  # A_y is master, D_y is slave
    ]
    
    for corner_master, corner_slave, mapping_name, dof in corner_constraints:
        location_fns_A.append(corners[corner_master])
        location_fns_B.append(corners[corner_slave])
        mappings_list.append(mappings[mapping_name])
        vecs.append(dof)
    
    return [location_fns_A, location_fns_B, mappings_list, vecs]


def periodic_boundary_conditions(periodic_bc_info, mesh, vec, dirichlet_bc_info=None):
    """
    See https://github.com/deepmodeling/jax-fem/blob/main/applications/periodic_bc/example.py
    
    Modified by xinyu
    """
    p_node_inds_list_A = []
    p_node_inds_list_B = []
    p_vec_inds_list = []

    location_fns_A, location_fns_B, mappings, vecs = periodic_bc_info
    for i in range(len(location_fns_A)):
        node_inds_A = onp.argwhere(jax.vmap(location_fns_A[i])(mesh.points)).reshape(-1)
        node_inds_B = onp.argwhere(jax.vmap(location_fns_B[i])(mesh.points)).reshape(-1)
        points_set_A = mesh.points[node_inds_A]
        points_set_B = mesh.points[node_inds_B]

        EPS = 1e-5
        node_inds_B_ordered = []
        for node_ind in node_inds_A:
            point_A = mesh.points[node_ind]
            dist = onp.linalg.norm(mappings[i](point_A)[None, :] - points_set_B, axis=-1)
            node_ind_B_ordered = node_inds_B[onp.argwhere(dist < EPS)].reshape(-1)
            node_inds_B_ordered.append(node_ind_B_ordered)

        node_inds_B_ordered = onp.array(node_inds_B_ordered).reshape(-1)
        vec_inds = onp.ones_like(node_inds_A, dtype=onp.int32) * vecs[i]

        p_node_inds_list_A.append(node_inds_A)
        p_node_inds_list_B.append(node_inds_B_ordered)
        p_vec_inds_list.append(vec_inds)
        assert len(node_inds_A) == len(node_inds_B_ordered)

    # For multiple variables (e.g, stokes flow, u-p coupling), offset will be nonzero.
    offset = 0
    inds_A_list = []
    inds_B_list = []
    for i in range(len(p_node_inds_list_A)):
        inds_A_list.append(onp.array(p_node_inds_list_A[i] * vec + p_vec_inds_list[i] + offset, dtype=onp.int32))
        inds_B_list.append(onp.array(p_node_inds_list_B[i] * vec + p_vec_inds_list[i] + offset, dtype=onp.int32))

    inds_A = onp.hstack(inds_A_list)
    inds_B = onp.hstack(inds_B_list)

    # Remove duplicate constraints: keep first occurrence
    unique_pairs = {}
    for i in range(len(inds_B)):
            if inds_B[i] not in unique_pairs:
                unique_pairs[inds_B[i]] = inds_A[i]
    
    inds_B = onp.array(list(unique_pairs.keys()), dtype=onp.int32)
    inds_A = onp.array(list(unique_pairs.values()), dtype=onp.int32)

    num_total_nodes = len(mesh.points)
    num_total_dofs = num_total_nodes * vec
    N = num_total_dofs
    M = num_total_dofs - len(inds_B)

    # The use of 'reduced_inds_map' seems to be a smart way to construct P_mat
    reduced_inds_map = onp.ones(num_total_dofs, dtype=onp.int32)
    reduced_inds_map[inds_B] = -(inds_A + 1)
    reduced_inds_map[reduced_inds_map == 1] = onp.arange(M)

    I = []
    J = []
    V = []
    for i in range(num_total_dofs):
        I.append(i)
        V.append(1.)
        if reduced_inds_map[i] < 0:
            J.append(reduced_inds_map[-reduced_inds_map[i] - 1])
        else:
            J.append(reduced_inds_map[i])
 
    P_mat = scipy.sparse.csr_array((onp.array(V), (onp.array(I), onp.array(J))), shape=(N, M))

    return P_mat