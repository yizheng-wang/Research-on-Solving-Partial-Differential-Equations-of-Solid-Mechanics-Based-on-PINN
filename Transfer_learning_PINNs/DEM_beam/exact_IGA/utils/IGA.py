#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File for IGA mesh classes
"""
import numpy as np
from utils.bernstein import bezier_extraction

class IGAMesh2D:
    """
    Class for a single-patch IGA mesh
    Input
    ------
    patch : Geometry2D object
    """
    def __init__(self, patch):
         # Compute the Bezier extraction operator in each space direction
        knot_u = patch.surf.knotvector_u
        knot_v = patch.surf.knotvector_v
        self.deg = np.array(patch.surf.degree)
        C_u, num_elem_u = bezier_extraction(knot_u, self.deg[0])
        C_v, num_elem_v = bezier_extraction(knot_v, self.deg[1])
        self.num_elem = num_elem_u * num_elem_v
        
        # Compute the Bezier extraction operator in 2D
        self.C = [None]*self.num_elem
        temp = np.arange(0, self.num_elem)
        index_matrix = np.reshape(temp, (num_elem_v, num_elem_u))
        #index_matrix = np.transpose(temp)
        for j in range(num_elem_v):
            for i in range(num_elem_u):
                elem_index = index_matrix[j,i]
                
                self.C[elem_index] = np.kron(C_v[j], C_u[i])
        # Compute the IEN array
        IEN, self.elem_vertex = self.makeIEN(patch)
        
        # Compute the number of basis functions
        len_u = len(knot_u)-self.deg[0]-1
        len_v = len(knot_v)-self.deg[1]-1
        self.num_basis = len_u * len_v
        
        # Set the (unweighted) control points and weights as arrays
        cpts_temp = np.reshape(patch.surf.ctrlpts, (len_u, len_v, 3))
        self.cpts = np.reshape(np.transpose(cpts_temp, (1,0,2)), (len_u*len_v, 3))
        self.cpts = np.transpose(self.cpts)
        #self.cpts = np.array(patch.surf.ctrlpts).transpose()
        wgts_temp = np.reshape(patch.surf.weights, (len_u, len_v))
        self.wgts = np.reshape(np.transpose(wgts_temp), len_u*len_v)
        #self.wgts = np.array(patch.surf.weights)
        
        # Set the IEN as list of arrays
        self.elem_node = [IEN[i,:] for i in range(self.num_elem)]

    
    def makeIEN(self, patch):
        """
        Create the IEN (node index to element) array for a given knot vector and p and
        elementVertex array
        
        Input:
            patch - (Geometry2D) geometry patch
            
        Output:
            IEN - array where each row corresponds to an element (non-empty
                  knot-span) and it contains the basis function indices with
                  support on the element
                  
            element_vertex - array where each row corresponds to an element
                             and it contains the coordinates of the element
                             corners in the parameters space in the format
                             [u_min, v_min, u_max, v_max]
        """
        
        knot_u = patch.surf.knotvector_u
        knot_v = patch.surf.knotvector_v
        num_elem_u = len(np.unique(knot_u))-1
        num_elem_v = len(np.unique(knot_v))-1
        num_elem = num_elem_u*num_elem_v
        deg = np.array(patch.surf.degree)
        num_entries = np.prod(deg+1)
        len_u = len(knot_u)-deg[0]-1
        IEN = np.zeros((num_elem, num_entries), dtype=int)
        element_vertex = np.zeros((num_elem, 4))
        element_counter = 0
        for j in range(len(knot_v)-1):
            for i in range(len(knot_u)-1):
                if (knot_u[i+1]>knot_u[i]) and (knot_v[j+1]>knot_v[j]):
                    element_vertex[element_counter, :] = [knot_u[i], knot_v[j],
                                                    knot_u[i+1], knot_v[j+1]]
                    # now we add the nodes from i-p,..., i in the u direction
                    # j-q,..., j in the v direction
                    tcount = 0
                    for t2 in range(j-deg[1], j+1):
                        for t1 in range(i-deg[0], i+1):
                            IEN[element_counter, tcount] = t1+t2*len_u
                            tcount += 1
                    element_counter += 1
        assert element_counter == num_elem
        return IEN, element_vertex
    
    
    def _get_boundary_indices(self):
        """
        Returns the boundary (down, right, up, left) indices for a matrix of
        dimension (p+1)*(q+1)
        
        Returns
        -------
        bnd_index : dict containing the "down", "up", "left", "right" boundary
                    indices
        """
        bnd_index = {}
        bnd_index["down"] = list(range(self.deg[0]+1))
        bnd_index["right"] = list(range(self.deg[0], np.prod(self.deg+1), 
                                        self.deg[0]+1))
        bnd_index["up"] = list(range((self.deg[0]+1)*self.deg[1], 
                               np.prod(self.deg+1)))
        bnd_index["left"] = list(range(0, 1+(self.deg[0]+1)*self.deg[1], 
                                       self.deg[0]+1))
        return bnd_index
        
    
    def classify_boundary(self):
        """   
        Classifies the boundary nodes and boundary elements according to the 
        side in the parameter space (i.e. bcdof_down and elem_down for v=0, 
        bcdof_up and elem_up for v=1, bcdof_left and elem_left for u=0, 
        bcdof_right and elem_right for u=1)        
        """
        bcdof_down = []
        elem_down = []
        bcdof_up = []
        elem_up = []
        bcdof_left = []
        elem_left = []
        bcdof_right = []
        elem_right = []
        bnd_index = self._get_boundary_indices()
        tol_eq = 1e-10
        self.bcdof = {}
        self.elem = {}
        
        for i_elem in range(self.num_elem):
            if abs(self.elem_vertex[i_elem, 0])<tol_eq:
                # u_min = 0
                bcdof_left.append(self.elem_node[i_elem][bnd_index["left"]])
                elem_left.append(i_elem)
            if abs(self.elem_vertex[i_elem, 1])<tol_eq:
                # v_min = 0
                bcdof_down.append(self.elem_node[i_elem][bnd_index["down"]])
                elem_down.append(i_elem)
            if abs(self.elem_vertex[i_elem, 2]-1)<tol_eq:
                # u_max = 1
                bcdof_right.append(self.elem_node[i_elem][bnd_index["right"]])
                elem_right.append(i_elem)
            if abs(self.elem_vertex[i_elem, 3]-1)<tol_eq:
                # v_max = 1
                bcdof_up.append(self.elem_node[i_elem][bnd_index["up"]])
                elem_up.append(i_elem)
        self.bcdof["down"] = np.unique(bcdof_down).tolist()
        self.bcdof["up"] = np.unique(bcdof_up).tolist()
        self.bcdof["left"] = np.unique(bcdof_left).tolist()
        self.bcdof["right"] = np.unique(bcdof_right).tolist()
        self.bcdof["down_left"] = np.intersect1d(bcdof_down, bcdof_left).tolist()
        self.bcdof["down_right"] = np.intersect1d(bcdof_down, bcdof_right).tolist()
        self.bcdof["up_left"] = np.intersect1d(bcdof_up, bcdof_left).tolist()
        self.bcdof["up_right"] = np.intersect1d(bcdof_up, bcdof_right).tolist()
        self.elem["down"] = elem_down
        self.elem["up"] = elem_up
        self.elem["left"] = elem_left
        self.elem["right"] = elem_right
        
    
        
