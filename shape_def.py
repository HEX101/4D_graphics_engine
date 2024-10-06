import itertools

import numpy as np
from numpy.typing import NDArray


class Shape:
    def __init__(self, vertices: NDArray[np.float64], edges: NDArray[np.int_]) -> None:
        self.vertices: NDArray[np.float64] = vertices
        self.edges: NDArray[np.int_] = edges
            
    @staticmethod
    def define_n_dimensional_cube(n: int) -> 'Shape':        
        vertices = np.array(list(itertools.product([-1, 1], repeat=n)), dtype=np.float64)

        edges = []
        dimension = vertices.shape[1]
        
        vertex_to_index = {tuple(vertex): idx for idx, vertex in enumerate(vertices)}
        
        for idx, vertex in enumerate(vertices):
            for dim in range(dimension):
                neighbor = vertex.copy()
                neighbor[dim] *= -1
                neighbor_tuple = tuple(neighbor)
                neighbor_idx = vertex_to_index.get(neighbor_tuple)
                if neighbor_idx is not None and neighbor_idx > idx:
                    edges.append((idx, neighbor_idx))
        
        edges_array = np.array(edges, dtype=np.int_)

        return Shape(vertices, edges_array)