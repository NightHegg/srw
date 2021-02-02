import numpy as np
import os

def read_task(cur_task):
    dirichlet_conditions = []
    neumann_conditions = []

    with open(f"modules/operations_input_files/input_files/task_{cur_task}.dat") as f:
        dimTask = int(f.readline())
        E, nyu = list(map(float, f.readline().split()))
        for _ in range(int(f.readline())):
            dirichlet_conditions.append([int(val) if idx in [0, 1] else float(val) for idx, val in enumerate(f.readline().split())])
        for _ in range(int(f.readline())):
            neumann_conditions.append([int(val) if idx == 0 else float(val) for idx, val in enumerate(f.readline().split())])
        coef_u, coef_sigma = list(map(float, f.readline().split()))
        coef_overlap = float(f.readline())

    return dimTask, E, nyu, dirichlet_conditions, neumann_conditions, coef_u, coef_sigma, coef_overlap


def read_mesh(cur_mesh):
    bounds = []
    area_points_coords = []
    area_elements = []

    with open(f"modules/operations_input_files/input_files/mesh/mesh_{cur_mesh}.dat") as f:
        for _ in range(int(f.readline())):
            bounds.append([float(x) for x in f.readline().split()])
        for _ in range(int(f.readline())):
            area_points_coords.append([float(val) for val in f.readline().split()])
        for _ in range(int(f.readline())):
            area_elements.append([int(val) for val in f.readline().split()])
    
    return np.array(bounds), np.array(area_points_coords), area_elements