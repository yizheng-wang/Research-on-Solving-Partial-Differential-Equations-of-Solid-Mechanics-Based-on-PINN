#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subroutines for multi-patch geometries
"""
import numpy as np

def get_corner_node2D(deg, position):
    """
    Gets the corner node position for an element of given degree

    Parameters
    ----------
    deg : (array like of length 2) 
        degree in the u and v direction (p and q)
    position : (string)
        position of the vertex can be either 
            "down_left"  (u=0, v=0), 
            "down_right" (u=1, v=0), 
            "up_right"   (u=1, v=1), 
            "up_left"    (u=0, v=1)

    Returns
    -------
    index : (int) index of the node in the elem_node array of length (p+1)*(q+1)

    """
    if position=="down_left":
        return 0
    elif position=="down_right":
        return deg[0]
    elif position=="up_right":
        return (deg[0]+1)*(deg[1]+1)-1
    elif position=="up_left":
        return (deg[0]+1)*deg[1]
    else:
        raise Exception("Wrong position given")
        
def find_vertex(vertices, vertex, tol_eq=1e-5):
    """
    Find if a vertex exist in the vertices list up to tol_eq tolerance

    Parameters
    ----------
    vertices : (2D array-like with 3 columns)
        list of vertices
    vertex : (1D array-like of length 3)
        vertex to check 
    tol_eq : float, optional
        Tolerance for equality. The default is 1e-10.

    Returns
    -------
    index : (int) index of vertex if it exists in vertices, None otherwise

    """
    for i in range(len(vertices)):
        if all([abs(vertices[i][j]-vertex[j])<tol_eq for j in range(3)]):
            return i
    

def gen_vertex2patch2D(patch_list):
    """
    Generates a list of vertex corners and the connectivity between the corner
    vertices and the patches

    Parameters
    ----------
    patch_list : list of IGAMesh2D objects

    Returns
    -------
    vertices : (array of Nx3) list of corner vertices containing the x, y and z 
               coordinate
    vertex2patch : (list of arrays of length N) list where each entry is an
                    array, with the first column the patch indices that contain
                    each vertex and in the second column the location of the
                    corner of each vertex in a patch with the encoding 
                         1 - down-left  (u=0, v=0), 
                         2 - down-right (u=1, v=0), 
                         3 - up-right   (u=1, v=1), 
                         4 - up-left    (u=0, v=1)
                         
    patch2vertex : (list of arrays of length 4) list which gives the patch to 
                    vertex connectivity, where the vertices are in counter-
                    clockwise order starting from the origin of the parameter 
                    space (u=0, v=0)

    """
    num_patches = len(patch_list)
    vertices = []
    vertex2patch = []
    patch2vertex = []
    tol_eq = 1e-10
    # loop over all the patches and elements and add the corner vertices
    for i in range(num_patches):
        cur_patch = patch_list[i]
        patch_entry = np.zeros(4, dtype=int)
        # helper function that gets called to update the vertices and vertex2patch
        def update_indices(encoding, node_index):
            cpt_index = cur_patch.elem_node[j][node_index]
            # check if the vertex exists in the list of vertices
            vertex = cur_patch.cpts[:, cpt_index]
            vert_index = find_vertex(vertices, vertex)
            if vert_index == None:
                # create new entry in the vertices and vertex2patch list
                vertices.append(vertex)
                row_entry = [[i, encoding]]
                vertex2patch.append(row_entry)
                vert_index = len(vertices)-1
            else:
                # update the vertex2patch list with the current patch
                row_entry = [i, encoding]
                vertex2patch[vert_index].append(row_entry)
            patch_entry[encoding-1] = vert_index
            return vertices, vertex2patch
        
        # find a corner element and the corner vertex
        num_elem = len(cur_patch.elem_node)
        for j in range(num_elem):
            elem_vertex = cur_patch.elem_vertex[j]
            if abs(elem_vertex[0])<tol_eq and abs(elem_vertex[1])<tol_eq:
                # get the vertex coordinates for the down_left element 
                encoding = 1
                node_index = get_corner_node2D(cur_patch.deg, "down_left")
                vertices, vertex2patch = update_indices(encoding, node_index)
                
            if abs(elem_vertex[2]-1)<tol_eq and abs(elem_vertex[1])<tol_eq:
                # get the vertex coordinates for the down_right element 
                encoding = 2
                node_index = get_corner_node2D(cur_patch.deg, "down_right")
                vertices, vertex2patch = update_indices(encoding, node_index)

            if abs(elem_vertex[2]-1)<tol_eq and abs(elem_vertex[3]-1)<tol_eq:
                # get the vertex coordinates for the up_right element 
                encoding = 3
                node_index = get_corner_node2D(cur_patch.deg, "up_right")
                vertices, vertex2patch = update_indices(encoding, node_index)

            if abs(elem_vertex[0])<tol_eq and abs(elem_vertex[3]-1)<tol_eq:
                # get the vertex coordinates for the up_left element 
                encoding = 4
                node_index = get_corner_node2D(cur_patch.deg, "up_left")
                vertices, vertex2patch = update_indices(encoding, node_index)
        patch2vertex.append(patch_entry)
                                                
    return vertices, vertex2patch, patch2vertex

def gen_edge_list(patch2vertex):
    """
    Generates a list of patch boundaries (edges) 

    Parameters
    ----------    
    patch2vertex: (list of arrays) patch to vertex connectivity

    Returns
    -------
    edge_list : dict where each entry contains either an array
                of length 5 or an array of length 2. If it is an array of length
                5, it is of the form [patchA, sideA, patchB, sideB, direction_flag]
                where patchA and patchB are the patch indices sharing the edge, 
                direction_flag = 1 if the parameteric direction matches along the
                edge and direction_flag = -1 if the parametric direction is
                reversed, sideA, sideB give the orientation of the side on patchA
                and patchB using the encoding 
                  1 - down (v=0), 
                  2 - right (u=1),
                  3 - up (v=1),
                  4 - left (u=0). 
                If it is an array of length 2, it is of
                the form [patch, side], which shows the patch index (for a boundary
                patch) and the side orientation as before. The key name is of
                the form "n_index1_n_index2" where index1 and index2 are the
                indices of the endpoint vertices, and index1<index2. 
    """
    num_patches = len(patch2vertex)
    edge_list = dict()
    for i in range(num_patches):
        num_verts = len(patch2vertex[i])
        for j in range(num_verts):
            if j<num_verts-1:
                pt_a = min(patch2vertex[i][j], patch2vertex[i][j+1])
                pt_b = max(patch2vertex[i][j], patch2vertex[i][j+1])
            else:
                pt_a = min(patch2vertex[i][j], patch2vertex[i][0])
                pt_b = max(patch2vertex[i][j], patch2vertex[i][0])
                
            edge_key = "n_" + str(pt_a) + "_n_" + str(pt_b)
            if edge_key in edge_list:
                edge_list[edge_key].append(i)
                edge_list[edge_key].append(j+1)
                # rearrange the order of the quad vertices so that the order
                # matches that of the parametric direction on each edge
                temp_patch_a = patch2vertex[edge_list[edge_key][0]][[0,1,3,2]]
                temp_patch_b = patch2vertex[i][[0,1,3,2]]
                indx_pt_a_patch_a = np.where(temp_patch_a==pt_a)[0][0]
                indx_pt_a_patch_b = np.where(temp_patch_b==pt_a)[0][0]
                indx_pt_b_patch_a = np.where(temp_patch_a==pt_b)[0][0]
                indx_pt_b_patch_b = np.where(temp_patch_b==pt_b)[0][0]
                # if the point locations on each quad are both in increasing
                # order or both in decreasing order, then the parameteric
                # direction is matching, otherwise it is reversed
                if ((indx_pt_a_patch_a > indx_pt_b_patch_a and 
                     indx_pt_a_patch_b > indx_pt_b_patch_b) or
                    (indx_pt_a_patch_a < indx_pt_b_patch_a and 
                     indx_pt_a_patch_b < indx_pt_b_patch_b)):
                    edge_list[edge_key].append(1)
                else:
                    edge_list[edge_key].append(-1)
            else:
                edge_list[edge_key]=[i,j+1]
    return edge_list

            
def get_element_from_corner(mesh, corner_index):
    """
    Outputs the element corresponding to the corner index in the given PHTelem patch
    Note: this depends on the entries created by mesh.classify_boundaries()
    Parameters
    ----------
    mesh : (IGAMesh2D) mesh object
    corner_index : (int) the given corner index using the encoding 
                    1 - down-left  (u=0, v=0), 
                    2 - down-right (u=1, v=0), 
                    3 - up-right   (u=1, v=1), 
                    4 - up-left    (u=0, v=1)
    Returns
    -------
    element_indx : (int) element index corresponding to the given corner index
    """
    if corner_index==1:
        return mesh.elem["down"][0]
    elif corner_index==2:
        return mesh.elem["right"][0]
    elif corner_index==3:
        return mesh.elem["up"][-1]
    elif corner_index==4:
        return mesh.elem["left"][-1]

def sort_edge_nodes_element(mesh, edge, flag_dir):
    """
    Outputs the nodes and element along the given edge, sorted in the order in
    which they appear in the parameter space
    Note: this depends on the entries created by mesh.classify_boundaries()
    
    Parameters
    ----------
    mesh : (IGAMesh2D) mesh object
    edge : (int) edge direction with the encoding 1- down (v=0), 2-right (u=1)
                 3-up (v=1), 4-left (u=0)
    flag_dir : (int) 1 means the parametric direction matches along the patch
                interface, -1 means that the parametric direction is reversed

    Returns
    -------
    nodes : (list) control point indices in the order that they appear along
              the patch interface
    elements : (list) element indicies in the order that they appear along the 
                patch interface
    """
    
    # initialize array to store the corners of the edge elements and the edge
    # element indices, for sorting purposes
    
    if edge == 1:
        elements =  mesh.elem["down"]
        nodes = mesh.bcdof["down"]
    elif edge==2:
        elements = mesh.elem["right"]
        nodes = mesh.bcdof["right"]
    elif edge==3:
        elements = mesh.elem["up"]
        nodes = mesh.bcdof["up"]
    elif edge==4:
        elements = mesh.elem["left"]
        nodes = mesh.bcdof["left"]
        
    if flag_dir == -1:
        elements = np.flip(elements)
        nodes = np.flip(nodes)
        
    return nodes, elements
    

def zip_conforming(mesh_list, vertex2patch, edge_list):
    """
    Connects conforming patches by creating the elem_nodes_global fields 
    from the vertex2patch and edge_list connectivities

    Parameters
    ----------
    mesh_list : (list of IGAMesh2D) list of meshes
    vertex2patch : list of arrays generated by gen_vertex2patch2D
    edge_list : (dict) generated by gen_edge_list

    Returns
    -------
    size_basis : (int) the dimension of the global basis 

    """
    num_patches = len(mesh_list)
    # we assume that all patches have the same polynomial degree and take the
    # corner indices based on the degree on the 0th patch
    corners = [0, mesh_list[0].deg[0], (mesh_list[0].deg[0]+1)*(mesh_list[0].deg[1]+1)-1, 
                      (mesh_list[0].deg[0]+1)*mesh_list[0].deg[1]]
    assigned_nodes = [None]*num_patches
    nodes_pattern = [None]*num_patches
    
    # initialize the nodes_pattern in all patches to zero 
    # and the elem_node_global field
    for i in range(num_patches):
        nodes_pattern[i] = np.zeros(mesh_list[i].num_basis, dtype=int)
        mesh_list[i].elem_node_global = [None]*mesh_list[i].num_elem
        
    
    # Step 1: Loop over the vertices in vertex2patch and assign the global indices 
    # as the patch corners
    size_basis = 0
    for vert_index in range(len(vertex2patch)):
        for i in range(len(vertex2patch[vert_index])):
            patch_index = vertex2patch[vert_index][i][0]
            corner_index = vertex2patch[vert_index][i][1]
            element_index = get_element_from_corner(mesh_list[patch_index], corner_index)
            local_nodes = mesh_list[patch_index].elem_node[element_index][corners[corner_index-1]]
            if assigned_nodes[patch_index]==None:
                assigned_nodes[patch_index] = [local_nodes]
            else:
                assigned_nodes[patch_index].append(local_nodes)
            nodes_pattern[patch_index][local_nodes] = vert_index
        
    # Step 2: Loop over the edges in edge_list and assign the global indices in
    # the interior of each edge
    
    size_basis += len(vertex2patch)
    for edge_field in edge_list:
        patchA = edge_list[edge_field][0]
        edgeA = edge_list[edge_field][1]
        nodesA, _ = sort_edge_nodes_element(mesh_list[patchA], edgeA, 1)
        nodesA = nodesA[1:-1]
        assigned_nodes[patchA] += nodesA
        num_new_nodes = len(nodesA)
        new_node_set = list(range(size_basis, size_basis+num_new_nodes))
        nodes_pattern[patchA][nodesA] = new_node_set
        # if the edge is in the interior (has length=5)
        if len(edge_list[edge_field])==5:
            # get the nodes on the boundary edge in patchB
            patchB = edge_list[edge_field][2]
            edgeB = edge_list[edge_field][3]
            flag_dir = edge_list[edge_field][4]
            nodesB, _ = sort_edge_nodes_element(mesh_list[patchB], edgeB, flag_dir)
            nodesB = nodesB[1:-1]
            assigned_nodes[patchB] += nodesB
            assert len(nodesA)==len(nodesB), "Non-conforming patches encountered"
            nodes_pattern[patchB][nodesB] = new_node_set
        size_basis += len(nodesA)
    
    # Step 3: Loop over the patches and assign the interior global indices
    for patch_index in range(num_patches):
        unassigned_nodes = np.setdiff1d(range(mesh_list[patch_index].num_basis), 
                                                   assigned_nodes[patch_index])
        num_new_nodes = len(unassigned_nodes)
        new_node_set = range(size_basis, size_basis+num_new_nodes)
        nodes_pattern[patch_index][unassigned_nodes] = new_node_set
        # loop over the active elements and assign the global node indices
        for i_elem in range(mesh_list[patch_index].num_elem):
            mesh_list[patch_index].elem_node_global[i_elem] = \
                nodes_pattern[patch_index][mesh_list[patch_index].elem_node[i_elem]]
        # update the global boundary node indices
        mesh_list[patch_index].bcdof_global = {}
        for side in mesh_list[patch_index].bcdof:
            mesh_list[patch_index].bcdof_global[side] = \
                nodes_pattern[patch_index][mesh_list[patch_index].bcdof[side]].tolist()
        size_basis += num_new_nodes
    
    return size_basis
    
    
                
                    
                    
                    
                
        