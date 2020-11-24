import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
from itertools import combinations
from itertools import groupby
import matplotlib.pyplot as plt
import sys
import os
import math
import time

def CalculateMatrixKInfo(idv, subdElements_Local, subdPoints_Coords, dimTask, D):
    row, col, data = [], [], []
    for element in subdElements_Local:
                B, A = CalculateStiffnessMatrix(element, subdPoints_Coords[idv], dimTask)
                K_element = np.dot(np.dot(B.transpose(), D), B) * A
                for i in range(3):
                    for j in range(3):
                        for k in range(dimTask):
                            row.append(element[i] * dimTask + k)
                            col.append(element[j] * dimTask + k)
                            data.append(K_element[i * dimTask + k, j * dimTask + k])
    return row, col, data

def CalculateCritConvergence(u_Current, u_Previous, areaPoints_Coords, dimTask, relation_PointsElements):
    first_sum, second_sum = 0, 0
    for idx, val in enumerate(u_Current):
        if val[1]:
            s = sum([CalculateStiffnessMatrix(i, areaPoints_Coords + u_Current, dimTask)[1] for i in relation_PointsElements[idx]]) / 3
            first_sum += s * np.linalg.norm(np.copy(val[1]) - np.copy(u_Previous[idx, 1]))**2 / np.linalg.norm(np.copy(val[1]))**2
            second_sum += s
        
    return math.sqrt(first_sum / second_sum)

def PlotBacklog(areaPoints_Coords, areaElements, coef_u):
    def temp(u):
        fig, ax = plt.subplots(1, len(u))
        for idx, val in enumerate(u):
            ax[idx].triplot(areaPoints_Coords[:, 0], areaPoints_Coords[:, 1], areaElements.copy())
            ax[idx].triplot(val[:, 0] * coef_u + areaPoints_Coords[:, 0], val[:, 1] * coef_u + areaPoints_Coords[:, 1], areaElements.copy())
            ax[idx].plot(areaPoints_Coords[:,0] + val[:, 0] * coef_u, areaPoints_Coords[:,1] + val[:, 1] * coef_u, 'o')
        
        fig.set_figwidth(12)
        fig.set_figheight(7)
        fig.set_facecolor('mintcream')
        plt.show()
    return temp

def getMessage():
    def temp(func, name = "Func"):
        message = []
        message = (f"{name}_xx (min: {min(func.transpose()[:, 0]):g}, max: {max(func.transpose()[:, 0]):g})\n"
                   f"{name}_yy (min: {min(func.transpose()[:, 1]):g}, max: {max(func.transpose()[:, 1]):g})\n"
                   f"{name}_xy (min: {min(func.transpose()[:, 2]):g}, max: {max(func.transpose()[:, 2]):g})\n")
        return message
    return temp

def PlotDisplacements(areaPoints_Coords, areaElements, coef_u):
    def temp(u):
        fig, ax = plt.subplots()
        ax.triplot(areaPoints_Coords[:,0], areaPoints_Coords[:,1], areaElements.copy())
        ax.triplot(u[:, 0] * coef_u + areaPoints_Coords[:, 0], u[:,1] * coef_u + areaPoints_Coords[:, 1], areaElements.copy())
        ax.plot(areaPoints_Coords[:,0] + u[:, 0] * coef_u, areaPoints_Coords[:,1] + u[:, 1] * coef_u, 'o')

        fig.set_figwidth(12)
        fig.set_figheight(9)
        fig.set_facecolor('mintcream')

        plt.show()
    return temp

def CalculateStiffnessMatrix(element, points, dimTask):
    '''
    element - элемент, в котором идёт расчёт \n
    points - массив координат \n
    dimTask - размерность \n
    '''
    B = np.zeros((3, 6))
    x_points = np.array([points[element[0], 0], points[element[1], 0], points[element[2], 0]])
    y_points = np.array([points[element[0], 1], points[element[1], 1], points[element[2], 1]])
    a = np.array([x_points[1] * y_points[2] - x_points[2] * y_points[1], 
                  x_points[2] * y_points[0] - x_points[0] * y_points[2],
                  x_points[0] * y_points[1] - x_points[1] * y_points[0]] )
    b = np.array([y_points[1]-y_points[2], y_points[2]-y_points[0], y_points[0]-y_points[1]])
    c = np.array([x_points[2]-x_points[1], x_points[0]-x_points[2], x_points[1]-x_points[0]])
    A = 0.5*np.linalg.det(np.array([[1, 1, 1], [x_points[0], x_points[1], x_points[2]], 
                                  [y_points[0], y_points[1], y_points[2]]]).transpose())
    for i in range(len(element)):
        B[:, i*dimTask:i*dimTask + 2] += np.array([[b[i], 0], [0, c[i]], [c[i]/2, b[i]/2]])/2/A
    return B, A

def boundCondition_Neumann(F, element, neumannPoints_Local, dimTask, value, points, dim):
    L = list(set(element) & set(neumannPoints_Local))
    len = np.linalg.norm(np.array(points[L[0]]) - np.array(points[L[1]]))
    for node in L:
        F[node * dimTask + dim] += value * len / 2
    return F

def boundCondition_Dirichlet(K, F, dimTask, node, value, dim):
    '''
    K - матрица жёсткости \n
    F - матрица правой части \n
    dimTask - размерность задачи \n
    node - номер узла \n
    value - значение \n
    dim - по какой координате \n
    '''
        
    F -= K.getcol(node * dimTask + dim) * value
    K[node * dimTask + dim, :] = 0
    K[:, node * dimTask + dim] = 0

    K[node * dimTask + dim, node * dimTask + dim] = 1
    F[node * dimTask + dim] = value

    return K, F

def readInputFile(nameFile):
    bounds = []
    dirichlet_conditions = []
    neumann_conditions = []
    areaPoints_Coords = []
    areaElements = []
    with open("tests/"+nameFile) as f:
        dimTask = int(f.readline())
        for _ in range(int(f.readline())):
            bounds.append([float(x) for x in f.readline().split()])
        splitCoef = float(f.readline())
        coefOverlap = float(f.readline())
        amntSubdomains = int(f.readline())
        E, nyu = list(map(float, f.readline().split()))
        coef_u, coef_sigma = list(map(float, f.readline().split()))
        for _ in range(int(f.readline())):
            dirichlet_conditions.append([int(val) if idx in [0, 1] else float(val) for idx, val in enumerate(f.readline().split())])
        for _ in range(int(f.readline())):
            neumann_conditions.append([int(val) if idx == 0 else float(val) for idx, val in enumerate(f.readline().split())])
        for _ in range(int(f.readline())):
            areaPoints_Coords.append([float(val) for val in f.readline().split()])
        for _ in range(int(f.readline())):
            areaElements.append([int(val) for val in f.readline().split()])
    return bounds, areaPoints_Coords, areaElements, dirichlet_conditions, neumann_conditions, dimTask, bounds, splitCoef, coefOverlap, amntSubdomains, E, nyu, coef_u, coef_sigma

def showInputFile(nameFile):
    bounds = []
    dirichlet_conditions = []
    neumann_conditions = []
    areaPoints_Coords = []
    areaElements = []
    with open("tests/"+nameFile) as f:
        dimTask = int(f.readline())
        for _ in range(int(f.readline())):
            bounds.append([float(x) for x in f.readline().split()])
        splitCoef = float(f.readline())
        coefOverlap = float(f.readline())
        amntSubdomains = list(map(int, f.readline().split()))
        E, nyu = list(map(float, f.readline().split()))
        coef_u, coef_sigma = list(map(float, f.readline().split()))
        for _ in range(int(f.readline())):
            dirichlet_conditions.append([int(val) if idx in [0, 1] else float(val) for idx, val in enumerate(f.readline().split())])
        for _ in range(int(f.readline())):
            neumann_conditions.append([int(val) if idx == 0 else float(val) for idx, val in enumerate(f.readline().split())])
        for _ in range(int(f.readline())):
            areaPoints_Coords.append([float(val) for val in f.readline().split()])
        for _ in range(int(f.readline())):
            areaElements.append([int(val) for val in f.readline().split()])
    message = (f"Размерность задачи - {dimTask} \n"
               f"Коэффициент разделения - {splitCoef} \n"
               f"Коэффициент захлёста - {coefOverlap} \n"
               f"Количество подобластей - [{amntSubdomains}] \n"
               f"Модуль Юнга - {E} Па, коэффициент Пуассона - {nyu} \n"
               f"Коэффициент масштаба для перемещения - {coef_u}, для напряжений - {coef_sigma} \n")
    return message

def SchwarzMultiplicativeProgram(nameFile, coefConvergence):
    """
    Программа возвращает:
    u_Current, Eps, Sigma, graph, it
    """
    bounds, areaPoints_Coords, areaElements, dirichlet_conditions, neumann_conditions, dimTask, bounds, _, coefOverlap, amntSubdomains, E, nyu, coef_u, coef_sigma = readInputFile(nameFile)
    
    bounds = np.array(bounds)
    areaPoints_Coords = np.array(areaPoints_Coords)

    #areaPoints = [x for x in range(len(areaPoints_Coords))]
    #areaBoundaryPoints = [idx for idx, val in enumerate(areaPoints_Coords) if val[0] in [bounds[0, 0], bounds[1, 0]] or val[1] in [bounds[0, 1], bounds[2, 1]]]
    #dirichletPointsAll = sorted(list(set(sum([side[0] for side in dirichletPoints], []))))
    #neumannPointsAll = sorted(list(set(sum([side[0] for side in neumannPoints], []))))

    dirichletPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                     if min([bounds[cond[0], 0], bounds[cond[0] + 1, 0]]) <= val[0] <= max([bounds[cond[0], 0], bounds[cond[0] + 1, 0]]) and
                        min([bounds[cond[0], 1], bounds[cond[0] + 1, 1]]) <= val[1] <= max([bounds[cond[0], 1], bounds[cond[0] + 1, 1]])], cond[1:]] for cond in dirichlet_conditions]

    neumannPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                     if min([bounds[cond[0], 0], bounds[cond[0] + 1, 0]]) <= val[0] <= max([bounds[cond[0], 0], bounds[cond[0] + 1, 0]])
                     and min([bounds[cond[0], 1], bounds[cond[0] + 1, 1]]) <= val[1] <= max([bounds[cond[0], 1], bounds[cond[0] + 1, 1]])], cond[1], cond[2]] for cond in neumann_conditions]

    relation_PointsElements = {}
    subdElements = []
    subdPoints = []
    subdPoints_Coords = []
    subdInternalPoints = []
    subdBoundary = []

    subdBounds = [bounds[0][0]] + [bounds[0][0] + i*(bounds[1][0] - bounds[0][0])/amntSubdomains for i in range(1, amntSubdomains)] + [bounds[1][0]]
    overlapBounds = [subdBounds[0]] + sum([[subdBounds[x] - (subdBounds[x] - subdBounds[x - 1]) * coefOverlap, subdBounds[x] + (subdBounds[x] - subdBounds[x - 1]) * coefOverlap]   
               for x in range(1, len(subdBounds)-1)], []) + [subdBounds[-1]]

    subdElements.append([element for element in areaElements 
                         if overlapBounds[0] < sum([areaPoints_Coords[element[i], 0] for i in range(3)])/3 < overlapBounds[2]])
    for i in range(1, len(overlapBounds) - 3, 2):
        subdElements.append([element for element in areaElements
                         if overlapBounds[i] < sum([areaPoints_Coords[element[i], 0] for i in range(3)])/3 < overlapBounds[i+3]])
    subdElements.append([element for element in areaElements 
                         if overlapBounds[-3] < sum([areaPoints_Coords[element[i], 0] for i in range(3)])/3 < overlapBounds[-1]])

    for subd in subdElements:
        subdPoints.append([el for el, _ in groupby(sorted(sum([i for i in subd], [])))])

    for subd in subdPoints:
        subdPoints_Coords.append(np.array([areaPoints_Coords[i].tolist() for i in subd]))

    for idx, val in enumerate(areaPoints_Coords):
        relation_PointsElements[idx] = [element for element in areaElements if idx in element]
    
    for idv, subd in enumerate(subdElements):
        tempList = []
        relation_PointsElements_subd = {}
        for point in subdPoints[idv]:
            relation_PointsElements_subd[point] = [element for element in subd if point in element]
        for idx, val in relation_PointsElements_subd.items():
            if val != relation_PointsElements[idx]:
                tempList.append(idx)
        #for point in areaBoundaryPoints:
        #    if point in subdPoints[idv]:
        #        tempList.append(point)
        subdBoundary.append(list(set(tempList)))

    for idx, subd in enumerate(subdElements):   
        subdInternalPoints.append(list(set(subdPoints[idx]) - set(subdBoundary[idx])))

    it = 0
    u_Current = np.zeros((areaPoints_Coords.shape[0], 2))
    D = np.array([[1, nyu/(1-nyu), 0], [nyu/(1-nyu), 1, 0], [0, 0, (1-2 * nyu) / 2 / (1-nyu)]]) * E * (1-nyu) / (1-2 * nyu) / (1+nyu)
    while True:
        u_Previous = np.copy(u_Current)
        for idv, subd in enumerate(subdElements):
            ratioPoints_LocalGlobal = dict(zip(range(len(subdPoints[idv])), subdPoints[idv]))
            ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}

            F = np.zeros((len(ratioPoints_GlobalLocal) * 2, 1))
            subdElements_Local = np.array([ratioPoints_GlobalLocal[x] for x in np.array(subd).ravel()]).reshape(len(subd), 3)
            
            row, col, data = CalculateMatrixKInfo(idv, subdElements_Local, subdPoints_Coords, dimTask, D)

            K = lil_matrix(coo_matrix((data, (row, col)), shape = (len(ratioPoints_GlobalLocal) * dimTask, len(ratioPoints_GlobalLocal) * dimTask)))

            for condition in dirichletPoints:
                listPoints = list(set(condition[0]) & set(subdPoints[idv]))
                for node in listPoints:
                    if condition[1][0] == 2:
                        K, F = boundCondition_Dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][1], 0)
                        K, F = boundCondition_Dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][2], 1)
                    else:
                        K, F = boundCondition_Dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][1], condition[1][0])

            for condition in neumannPoints:
                listPoints = list(set(condition[0]) & set(subdPoints[idv]))
                segmentPoints = list(combinations(listPoints, 2))
                for element in [element for element in subd for x in segmentPoints if x[0] in element and x[1] in element]:                
                    F = boundCondition_Neumann(F, [ratioPoints_GlobalLocal[x] for x in element], [ratioPoints_GlobalLocal[x] for x in listPoints], dimTask, condition[1], subdPoints_Coords[idv], 0)
                    F = boundCondition_Neumann(F, [ratioPoints_GlobalLocal[x] for x in element], [ratioPoints_GlobalLocal[x] for x in listPoints], dimTask, condition[2], subdPoints_Coords[idv], 1)

            listPoints_Schwarz = sum([list(set(subdBoundary[idv]) & set(subd)) for idx, subd in enumerate(subdPoints) if idx != idv], [])
            for node in listPoints_Schwarz:
                K, F = boundCondition_Dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], u_Current[node, 0], dim = 0)
                K, F = boundCondition_Dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], u_Current[node, 1], dim = 1)

            u_subd = spsolve(K.tocsc(), F).reshape((-1, 2))
  
            for x in list(ratioPoints_LocalGlobal.keys()):
                u_Current[ratioPoints_LocalGlobal[x], :] = np.copy(u_subd[x, :])

        it += 1

        critConvergence = CalculateCritConvergence(u_Current, u_Previous, areaPoints_Coords, dimTask, relation_PointsElements)
        print(f"Multiplicative CritConvergence = {critConvergence}", end = "\r")
        if critConvergence < coefConvergence:
            break
    
    Eps = []
    for element in areaElements:
        B, _ = CalculateStiffnessMatrix(element, areaPoints_Coords + u_Current, dimTask)
        Eps.append(np.dot(B, np.ravel(np.array([u_Current[element[0]], u_Current[element[1]], u_Current[element[2]]])).transpose()))
    Eps = np.array(Eps)
    Sigma = np.dot(D, Eps.transpose())
    graph = PlotDisplacements(areaPoints_Coords, areaElements, coef_u)

    return u_Current, Eps, Sigma, graph, it

def SchwarzAdditiveProgram(nameFile, coefConvergence):
    """
    Программа возвращает:
    u_Current, Eps, Sigma, graph, it
    """
    bounds, areaPoints_Coords, areaElements, dirichlet_conditions, neumann_conditions, dimTask, bounds, splitCoef, coefOverlap, amntSubdomains, E, nyu, coef_u, coef_sigma = readInputFile(nameFile)
    
    bounds = np.array(bounds)
    areaPoints_Coords = np.array(areaPoints_Coords)

    dirichletPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                     if min([bounds[cond[0], 0], bounds[cond[0] + 1, 0]]) <= val[0] <= max([bounds[cond[0], 0], bounds[cond[0] + 1, 0]]) and
                        min([bounds[cond[0], 1], bounds[cond[0] + 1, 1]]) <= val[1] <= max([bounds[cond[0], 1], bounds[cond[0] + 1, 1]])], cond[1:]] for cond in dirichlet_conditions]

    neumannPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                     if min([bounds[cond[0], 0], bounds[cond[0] + 1, 0]]) <= val[0] <= max([bounds[cond[0], 0], bounds[cond[0] + 1, 0]])
                     and min([bounds[cond[0], 1], bounds[cond[0] + 1, 1]]) <= val[1] <= max([bounds[cond[0], 1], bounds[cond[0] + 1, 1]])], cond[1], cond[2]] for cond in neumann_conditions]


    #areaPoints = [x for x in range(len(areaPoints_Coords))]
    #areaBoundaryPoints = [idx for idx, val in enumerate(areaPoints_Coords) if val[0] in [bounds[0, 0], bounds[1, 0]] or val[1] in [bounds[0, 1], bounds[2, 1]]]
    #dirichletPointsAll = sorted(list(set(sum([side[0] for side in dirichletPoints], []))))
    #neumannPointsAll = sorted(list(set(sum([side[0] for side in neumannPoints], []))))

    relation_PointsElements = {}
    subdElements = []
    subdPoints = []
    subdPoints_Coords = []
    subdInternalPoints = []
    subdBoundary = []

    subdBounds = [bounds[0][0]] + [bounds[0][0] + i*(bounds[1][0] - bounds[0][0])/amntSubdomains for i in range(1, amntSubdomains)] + [bounds[1][0]]
    overlapBounds = [subdBounds[0]] + sum([[subdBounds[x] - (subdBounds[x] - subdBounds[x - 1]) * coefOverlap, subdBounds[x] + (subdBounds[x] - subdBounds[x - 1]) * coefOverlap]   
               for x in range(1, len(subdBounds)-1)], []) + [subdBounds[-1]]

    subdElements.append([element for element in areaElements 
                         if overlapBounds[0] < sum([areaPoints_Coords[element[i], 0] for i in range(3)])/3 < overlapBounds[2]])
    for i in range(1, len(overlapBounds) - 3, 2):
        subdElements.append([element for element in areaElements
                         if overlapBounds[i] < sum([areaPoints_Coords[element[i], 0] for i in range(3)])/3 < overlapBounds[i+3]])
    subdElements.append([element for element in areaElements 
                         if overlapBounds[-3] < sum([areaPoints_Coords[element[i], 0] for i in range(3)])/3 < overlapBounds[-1]])

    for subd in subdElements:
        subdPoints.append([el for el, _ in groupby(sorted(sum([i for i in subd], [])))])

    for subd in subdPoints:
        subdPoints_Coords.append(np.array([areaPoints_Coords[i].tolist() for i in subd]))

    for idx, val in enumerate(areaPoints_Coords):
        relation_PointsElements[idx] = [element for element in areaElements if idx in element]
    
    for idv, subd in enumerate(subdElements):
        tempList = []
        relation_PointsElements_subd = {}
        for point in subdPoints[idv]:
            relation_PointsElements_subd[point] = [element for element in subd if point in element]
        for idx, val in relation_PointsElements_subd.items():
            if val != relation_PointsElements[idx]:
                tempList.append(idx)
        #for point in areaBoundaryPoints:
        #    if point in subdPoints[idv]:
        #        tempList.append(point)
        subdBoundary.append(list(set(tempList)))

    for idx, subd in enumerate(subdElements):   
        subdInternalPoints.append(list(set(subdPoints[idx]) - set(subdBoundary[idx])))

    graph = PlotDisplacements(areaPoints_Coords, areaElements, coef_u)

    alpha = 0.5
    it = 0
    u_Current = np.zeros((areaPoints_Coords.shape[0], 2))
    D = np.array([[1, nyu/(1-nyu), 0], [nyu/(1-nyu), 1, 0], [0, 0, (1-2 * nyu) / 2 / (1-nyu)]]) * E * (1-nyu) / (1-2 * nyu) / (1+nyu)
    while True:
        u_Previous = np.copy(u_Current)
        u_Current = np.zeros_like(u_Previous)
        u_Current_temp = np.zeros_like(u_Previous)
        u_Sum = np.zeros_like(u_Current)
        for idv, subd in enumerate(subdElements):
            u_Current_temp = np.copy(u_Previous)

            ratioPoints_LocalGlobal = dict(zip(range(len(subdPoints[idv])), subdPoints[idv]))
            ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}
        
            F = np.zeros((len(ratioPoints_GlobalLocal) * 2, 1))
            subdElements_Local = np.array([ratioPoints_GlobalLocal[x] for x in np.array(subd).ravel()]).reshape(len(subd), 3)

            row, col, data = CalculateMatrixKInfo(idv, subdElements_Local, subdPoints_Coords, dimTask, D)

            K = lil_matrix(coo_matrix((data, (row, col)), shape = (len(ratioPoints_GlobalLocal) * dimTask, len(ratioPoints_GlobalLocal) * dimTask)))

            for condition in dirichletPoints:
                listPoints = list(set(condition[0]) & set(subdPoints[idv]))
                for node in listPoints:
                    if condition[1][0] == 2:
                        K, F = boundCondition_Dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][1], 0)
                        K, F = boundCondition_Dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][2], 1)
                    else:
                        K, F = boundCondition_Dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][1], condition[1][0])

            for condition in neumannPoints:
                listPoints = list(set(condition[0]) & set(subdPoints[idv]))
                segmentPoints = list(combinations(listPoints, 2))
                for element in [element for element in subd for x in segmentPoints if x[0] in element and x[1] in element]:                
                    F = boundCondition_Neumann(F, [ratioPoints_GlobalLocal[x] for x in element], [ratioPoints_GlobalLocal[x] for x in listPoints], dimTask, condition[1], subdPoints_Coords[idv], 0)
                    F = boundCondition_Neumann(F, [ratioPoints_GlobalLocal[x] for x in element], [ratioPoints_GlobalLocal[x] for x in listPoints], dimTask, condition[2], subdPoints_Coords[idv], 1)
        
            listPoints_Schwarz = sum([list(set(subdBoundary[idv]) & set(subd)) for idx, subd in enumerate(subdPoints) if idx != idv], [])
            for node in listPoints_Schwarz:
                K, F = boundCondition_Dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], u_Previous[node, 0], dim = 0)
                K, F = boundCondition_Dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], u_Previous[node, 1], dim = 1)
            
            u_subd = spsolve(K.tocsc(), F).reshape((-1, 2))

            for x in list(ratioPoints_LocalGlobal.keys()):
                u_Current_temp[ratioPoints_LocalGlobal[x], :] = np.copy(u_subd[x, :])

            u_Sum += (u_Current_temp - u_Previous)
        it += 1
        u_Current = np.copy(u_Previous) + (alpha * u_Sum)

        critConvergence = CalculateCritConvergence(u_Current, u_Previous, areaPoints_Coords, dimTask, relation_PointsElements)
        print(f"Additive CritConvergence = {critConvergence}", end = "\r")
        if critConvergence < coefConvergence:
            break
    
    Eps=[]
    for element in areaElements:
        B, _ = CalculateStiffnessMatrix(element, areaPoints_Coords, dimTask)
        Eps.append(np.dot(B, np.ravel(np.array([u_Current[element[0]], u_Current[element[1]], u_Current[element[2]]])).transpose()))
    Eps = np.array(Eps)
    Sigma = np.dot(D, Eps.transpose())

    return u_Current, Eps, Sigma, graph, it

def mainProgram(nameFile):
    """
    Программа возвращает:
    u_Current, Eps, Sigma, graph
    """
    bounds, areaPoints_Coords, areaElements, dirichlet_conditions, neumann_conditions, dimTask, bounds, splitCoef, coefOverlap, amntSubdomains, E, nyu, coef_u, coef_sigma = readInputFile(nameFile)
    
    bounds = np.array(bounds)
    areaPoints_Coords = np.array(areaPoints_Coords)
    
    areaBoundaryPoints = [idx for idx, val in enumerate(areaPoints_Coords) if val[0] in [bounds[0, 0], bounds[1, 0]] or val[1] in [bounds[0, 1], bounds[2, 1]]]
    
    dirichletPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                     if min([bounds[cond[0], 0], bounds[cond[0] + 1, 0]]) <= val[0] <= max([bounds[cond[0], 0], bounds[cond[0] + 1, 0]]) and
                        min([bounds[cond[0], 1], bounds[cond[0] + 1, 1]]) <= val[1] <= max([bounds[cond[0], 1], bounds[cond[0] + 1, 1]])], cond[1:]] for cond in dirichlet_conditions]

    neumannPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                     if min([bounds[cond[0], 0], bounds[cond[0] + 1, 0]]) <= val[0] <= max([bounds[cond[0], 0], bounds[cond[0] + 1, 0]])
                     and min([bounds[cond[0], 1], bounds[cond[0] + 1, 1]]) <= val[1] <= max([bounds[cond[0], 1], bounds[cond[0] + 1, 1]])], cond[1], cond[2]] for cond in neumann_conditions]

    areaPoints = [x for x in range(len(areaPoints_Coords))]

    #dirichletPointsAll = sorted(list(set(sum([side[0] for side in dirichletPoints], []))))
    #neumannPointsAll = sorted(list(set(sum([side[0] for side in neumannPoints], []))))
    #areaBoundaryPoints_Coords = [areaPoints_Coords[x] for x in areaBoundaryPoints]

    K_global = np.zeros((len(areaPoints_Coords) * 2, len(areaPoints_Coords) * 2))
    F = np.zeros((len(areaPoints_Coords) * 2, 1))
    D = np.array([[1, nyu/(1-nyu), 0], [nyu/(1-nyu), 1, 0], [0, 0, (1-2 * nyu) / 2 / (1-nyu)]]) * E * (1-nyu) / (1-2 * nyu) / (1+nyu)
    
    for element in areaElements:
        B, A = CalculateStiffnessMatrix(element, areaPoints_Coords, dimTask)
        K = np.dot(np.dot(B.transpose(), D), B) * A
        for i in range(3):
            for j in range(3):
                K_global[element[i]*dimTask:element[i]*dimTask+2, element[j]*dimTask:element[j]*dimTask+2] += K[i*dimTask:i*dimTask+2,j*dimTask:j*dimTask+2]      

    for condition in dirichletPoints:
        for node in condition[0]:
            if condition[1][0] == 2:
                K_global, F = boundCondition_Dirichlet(K_global, F, dimTask, node, condition[1][1], 0)
                K_global, F = boundCondition_Dirichlet(K_global, F, dimTask, node, condition[1][2], 1)
            else:
                K_global, F = boundCondition_Dirichlet(K_global, F, dimTask, node, condition[1][1], condition[1][0])

    #for idx, val in enumerate(areaBoundaryPoints_Coords):
    #    K_global, F = boundCondition_Dirichlet(K_global, F, dimTask, idx, val[0]**2 + val[1]**2, 0)
    #    K_global, F = boundCondition_Dirichlet(K_global, F, dimTask, idx, val[0]**2 + val[1]**2, 1)
    
    for condition in neumannPoints:
        for element in [element for element in areaElements for x in list(combinations(condition[0], 2)) if x[0] in element and x[1] in element]:
            F = boundCondition_Neumann(F, element, condition[0], dimTask, condition[1], areaPoints_Coords, 0)
            F = boundCondition_Neumann(F, element, condition[0], dimTask, condition[2], areaPoints_Coords, 1)
    
    u = np.linalg.solve(K_global, F).reshape((areaPoints_Coords.shape[0],2))

    Eps = []
    for element in areaElements:
        B, A = CalculateStiffnessMatrix(element, areaPoints_Coords + u, dimTask)
        Eps.append(np.dot(B, np.ravel(np.array([u[element[0]], u[element[1]], u[element[2]]])).transpose()))
    Eps = np.array(Eps).transpose()
    Sigma = np.dot(D, Eps)

    graph = PlotDisplacements(areaPoints_Coords, areaElements, coef_u)

    return u, Eps, Sigma, graph

if __name__ == "__main__":
    nameFile = "test_1_20_4.dat"
    coefConvergence = 1e-3

    start_time_1 = time.time()
    u, *arg, it_1 = SchwarzMultiplicativeProgram(nameFile, coefConvergence)
    end_time_1 = time.time()
    print("\n")

    start_time_2 = time.time()
    u_2, *arg, graph, it_2 = SchwarzAdditiveProgram(nameFile, coefConvergence)
    end_time_2 = time.time()
    print("\n")

    print(f"Время выполнения первой программы: {end_time_1 - start_time_1} \n"
          f"Количество итераций первой программы: {it_1} \n"
          f"Время выполнения второй программы: {end_time_2 - start_time_2} \n"
          f"Количество итераций второй программы: {it_2} \n"
         )