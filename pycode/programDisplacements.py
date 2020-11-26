import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg, cgs
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
from itertools import combinations
from itertools import groupby
import matplotlib.pyplot as plt
import sys
import os
import math
import time

def benchmark(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        u, Eps, Sigma, graph = func(*args, **kwargs)
        end_time = time.time()
        print(f"Время выполнения обычной задачи: {end_time - start_time}\n")
        return u, Eps, Sigma, graph
    return wrapper

def benchmarkSchwarz(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        u, Eps, Sigma, graph, it = func(*args, **kwargs)
        end_time = time.time()
        print(f"Время выполнения методом Шварца: {end_time - start_time}\n"
              f"Количество выполненных итераций: {it}")
        return u, Eps, Sigma, graph, it
    return wrapper

def CalculateEps(areaElements, areaPoints_Coords, dimTask, u_Current, coef_u):
    Eps=[]
    for element in areaElements:
        B, _ = CalculateStiffnessMatrix(element, areaPoints_Coords, dimTask)
        Eps.append(np.dot(B, np.ravel(np.array([u_Current[element[0]], u_Current[element[1]], u_Current[element[2]]])).transpose()))
    return np.array(Eps)

def calculateK(areaElements, areaPoints_Coords, D, dimTask):
    K = np.zeros((len(areaPoints_Coords) * 2, len(areaPoints_Coords) * 2))
    for element in areaElements:
                B, A = CalculateStiffnessMatrix(element, areaPoints_Coords, dimTask)
                K_element = B.T @ D @ B * A
                for i in range(3):
                    for j in range(3):
                        K[element[i]*dimTask:element[i]*dimTask+2, element[j]*dimTask:element[j]*dimTask+2] += K_element[i*dimTask:i*dimTask+2,j*dimTask:j*dimTask+2]
    return K

def CalculateMatrixKInfo(areaElements, areaPoints_Coords, D, dimTask):
    row, col, data = [], [], []
    for element in areaElements:
                B, A = CalculateStiffnessMatrix(element, areaPoints_Coords, dimTask)
                K_element = B.T @ D @ B * A
                for i in range(3):
                    for j in range(3):
                        for k in range(dimTask):
                            for z in range(dimTask):
                                row.append(element[i] * dimTask + k)
                                col.append(element[j] * dimTask + z)
                                data.append(K_element[i * dimTask + k, j * dimTask + z])                 
    return row, col, data

def CalculateCritConvergence(u_Current, u_Previous, areaPoints_Coords, dimTask, relation_PointsElements):
    first_sum, second_sum = 0, 0
    for idx, val in enumerate(u_Current):
        if val[1]:
            s = sum([CalculateStiffnessMatrix(i, areaPoints_Coords + u_Current, dimTask)[1] for i in relation_PointsElements[idx]]) / 3
            first_sum += s * np.linalg.norm(np.copy(val[1]) - np.copy(u_Previous[idx, 1]))**2 / np.linalg.norm(np.copy(val[1]))**2
            second_sum += s
        
    return math.sqrt(first_sum / second_sum)

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
    if sparse.issparse(K):
        K_col = K.getcol(node * dimTask + dim).toarray()
    else:
        K_col = K[:, node * dimTask + dim]
    F -= np.ravel(K_col) * value
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
    return bounds, areaPoints_Coords, areaElements, dirichlet_conditions, neumann_conditions, dimTask, splitCoef, coefOverlap, amntSubdomains, E, nyu, coef_u, coef_sigma

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

def SchwarzMultiplicativeProgram(nameFile, coefConvergence, func = sparse.linalg.spsolve):
    """
    Программа возвращает:
    u_Current, Eps, Sigma, graph, it
    """
    *arg, = readInputFile(nameFile)
    
    bounds               = np.array(arg[0])
    areaPoints_Coords    = np.array(arg[1])
    areaElements         = arg[2]
    dirichlet_conditions = arg[3]
    neumann_conditions   = arg[4]
    dimTask              = arg[5]
    splitCoef            = arg[6]
    coefOverlap          = arg[7]
    amntSubdomains       = arg[8]
    E                    = arg[9]
    nyu                  = arg[10]
    coef_u               = arg[11]
    coef_sigma           = arg[12]

    #areaPoints = [x for x in range(len(areaPoints_Coords))]
    #areaBoundaryPoints = [idx for idx, val in enumerate(areaPoints_Coords) if val[0] in [bounds[0, 0], bounds[1, 0]] or val[1] in [bounds[0, 1], bounds[2, 1]]]
    #dirichletPointsAll = sorted(list(set(sum([side[0] for side in dirichletPoints], []))))
    #neumannPointsAll = sorted(list(set(sum([side[0] for side in neumannPoints], []))))
    boundLimits = lambda cond, type: [bounds[cond[0], type], bounds[cond[0] + 1, type]]

    dirichletPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                    if min(boundLimits(cond, 0)) <= val[0] <= max(boundLimits(cond, 0)) and
                       min(boundLimits(cond, 1)) <= val[1] <= max(boundLimits(cond, 1))], cond[1:]] for cond in dirichlet_conditions]

    neumannPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                    if min(boundLimits(cond, 0)) <= val[0] <= max(boundLimits(cond, 0)) and
                       min(boundLimits(cond, 1)) <= val[1] <= max(boundLimits(cond, 1))], cond[1], cond[2]] for cond in neumann_conditions]

    relation_PointsElements = {}
    subdElements = []
    subdPoints = []
    subdPoints_Coords = []
    subdInternalPoints = []
    subdBoundary = []

    subdBounds = [bounds[0][0]] + [bounds[0][0] + i*(bounds[1][0] - bounds[0][0])/amntSubdomains for i in range(1, amntSubdomains)] + [bounds[1][0]]

    subdLimit = lambda x, type: subdBounds[x] + type * (subdBounds[x] - subdBounds[x - 1]) * coefOverlap
    overlapBounds = [subdBounds[0]] + sum([[subdLimit(x, -1), subdLimit(x, 1)] for x in range(1, len(subdBounds)-1)], []) + [subdBounds[-1]]

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
    it = 0
    u_Current = np.zeros((areaPoints_Coords.shape[0], 2))
    D = np.array([[1, nyu/(1-nyu), 0], [nyu/(1-nyu), 1, 0], [0, 0, (1-2 * nyu) / 2 / (1-nyu)]]) * E * (1-nyu) / (1-2 * nyu) / (1+nyu)
    while True:
        u_Previous = np.copy(u_Current)
        for idv, subd in enumerate(subdElements):
            ratioPoints_LocalGlobal = dict(zip(range(len(subdPoints[idv])), subdPoints[idv]))
            ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}

            F = np.zeros(subdPoints_Coords[idv].size)
            subdElements_Local = np.array([ratioPoints_GlobalLocal[x] for x in np.array(subd).ravel()]).reshape(len(subd), 3)
            
            row, col, data = CalculateMatrixKInfo(subdElements_Local, subdPoints_Coords[idv], D, dimTask)

            K = lil_matrix(coo_matrix((data, (row, col)), shape = (F.size, F.size)))

            for condition in dirichletPoints:
                listPoints = list(set(condition[0]) & set(subdPoints[idv]))
                for node in listPoints:
                    if condition[1][0] == 2:
                        K, F = boundCondition_Dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][1], 0)
                        K, F = boundCondition_Dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][2], 1)
                    else:
                        K, F = boundCondition_Dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][1], condition[1][0])

            rpGL_Change = lambda L: [ratioPoints_GlobalLocal[x] for x in L]
            for condition in neumannPoints:
                listPoints = list(set(condition[0]) & set(subdPoints[idv]))
                segmentPoints = list(combinations(listPoints, 2))
                for element in [element for element in subd for x in segmentPoints if x[0] in element and x[1] in element]:                
                    F = boundCondition_Neumann(F, rpGL_Change(element), rpGL_Change(listPoints), dimTask, condition[1], subdPoints_Coords[idv], 0)
                    F = boundCondition_Neumann(F, rpGL_Change(element), rpGL_Change(listPoints), dimTask, condition[2], subdPoints_Coords[idv], 1)

            listPoints_Schwarz = sum([list(set(subdBoundary[idv]) & set(subd)) for idx, subd in enumerate(subdPoints) if idx != idv], [])

            for node in listPoints_Schwarz:
                K, F = boundCondition_Dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], u_Current[node, 0], dim = 0)
                K, F = boundCondition_Dirichlet(K, F, dimTask, ratioPoints_GlobalLocal[node], u_Current[node, 1], dim = 1)

            [*arg,] = func(K.tocsr(), F)
            u_subd = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))

            for x in list(ratioPoints_LocalGlobal.keys()):
                u_Current[ratioPoints_LocalGlobal[x], :] = np.copy(u_subd[x, :])

        it += 1

        critConvergence = CalculateCritConvergence(u_Current, u_Previous, areaPoints_Coords, dimTask, relation_PointsElements)
        #print(f"Multiplicative CritConvergence = {critConvergence}", end = "\r")
        if critConvergence < coefConvergence:
            break
    
    Eps = CalculateEps(areaElements, areaPoints_Coords, dimTask, u_Current, coef_u)
    Sigma = np.dot(D, Eps.transpose())

    return u_Current, Eps, Sigma, graph, it

def SchwarzAdditiveProgram(nameFile, coefConvergence, func = sparse.linalg.spsolve):
    """
    Программа возвращает:
    u_Current, Eps, Sigma, graph, it
    """
    *arg, = readInputFile(nameFile)
    
    bounds               = np.array(arg[0])
    areaPoints_Coords    = np.array(arg[1])
    areaElements         = arg[2]
    dirichlet_conditions = arg[3]
    neumann_conditions   = arg[4]
    dimTask              = arg[5]
    splitCoef            = arg[6]
    coefOverlap          = arg[7]
    amntSubdomains       = arg[8]
    E                    = arg[9]
    nyu                  = arg[10]
    coef_u               = arg[11]
    coef_sigma           = arg[12]

    #areaPoints = [x for x in range(len(areaPoints_Coords))]
    #areaBoundaryPoints = [idx for idx, val in enumerate(areaPoints_Coords) if val[0] in [bounds[0, 0], bounds[1, 0]] or val[1] in [bounds[0, 1], bounds[2, 1]]]
    #dirichletPointsAll = sorted(list(set(sum([side[0] for side in dirichletPoints], []))))
    #neumannPointsAll = sorted(list(set(sum([side[0] for side in neumannPoints], []))))

    boundLimits = lambda cond, type: [bounds[cond[0], type], bounds[cond[0] + 1, type]]

    dirichletPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                    if min(boundLimits(cond, 0)) <= val[0] <= max(boundLimits(cond, 0)) and
                       min(boundLimits(cond, 1)) <= val[1] <= max(boundLimits(cond, 1))], cond[1:]] for cond in dirichlet_conditions]

    neumannPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                    if min(boundLimits(cond, 0)) <= val[0] <= max(boundLimits(cond, 0)) and
                       min(boundLimits(cond, 1)) <= val[1] <= max(boundLimits(cond, 1))], cond[1], cond[2]] for cond in neumann_conditions]

    relation_PointsElements = {}
    subdElements = []
    subdPoints = []
    subdPoints_Coords = []
    subdInternalPoints = []
    subdBoundary = []

    subdBounds = [bounds[0][0]] + [bounds[0][0] + i*(bounds[1][0] - bounds[0][0])/amntSubdomains for i in range(1, amntSubdomains)] + [bounds[1][0]]
    
    subdLimit = lambda x, type: subdBounds[x] + type * (subdBounds[x] - subdBounds[x - 1]) * coefOverlap
    overlapBounds = [subdBounds[0]] + sum([[subdLimit(x, -1), subdLimit(x, 1)] for x in range(1, len(subdBounds)-1)], []) + [subdBounds[-1]]

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
        
            F = np.zeros(subdPoints_Coords[idv].size)
            subdElements_Local = np.array([ratioPoints_GlobalLocal[x] for x in np.array(subd).ravel()]).reshape(len(subd), 3)

            row, col, data = CalculateMatrixKInfo(subdElements_Local, subdPoints_Coords[idv], D, dimTask)

            K = lil_matrix(coo_matrix((data, (row, col)), shape = (F.size, F.size)))

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
            
            *arg, = func(K.tocsr(), F)

            u_subd = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))

            for x in list(ratioPoints_LocalGlobal.keys()):
                u_Current_temp[ratioPoints_LocalGlobal[x], :] = np.copy(u_subd[x, :])

            u_Sum += (u_Current_temp - u_Previous)
        it += 1
        u_Current = np.copy(u_Previous) + (alpha * u_Sum)

        critConvergence = CalculateCritConvergence(u_Current, u_Previous, areaPoints_Coords, dimTask, relation_PointsElements)
        #print(f"Additive CritConvergence = {critConvergence}", end = "\r")
        if critConvergence < coefConvergence:
            break
    
    Eps = CalculateEps(areaElements, areaPoints_Coords, dimTask, u_Current, coef_u)
    Sigma = np.dot(D, Eps.transpose())

    return u_Current, Eps, Sigma, graph, it

def mainProgram(nameFile, func = sparse.linalg.cg):
    """
    Программа возвращает:
    u_Current, Eps, Sigma, graph
    """
    *arg, = readInputFile(nameFile)
    
    bounds               = np.array(arg[0])
    areaPoints_Coords    = np.array(arg[1])
    areaElements         = arg[2]
    dirichlet_conditions = arg[3]
    neumann_conditions   = arg[4]
    dimTask              = arg[5]
    splitCoef            = arg[6]
    coefOverlap          = arg[7]
    amntSubdomains       = arg[8]
    E                    = arg[9]
    nyu                  = arg[10]
    coef_u               = arg[11]
    coef_sigma           = arg[12]

    D = np.array([[1, nyu/(1-nyu), 0], [nyu/(1-nyu), 1, 0], [0, 0, (1-2 * nyu) / 2 / (1-nyu)]]) * E * (1-nyu) / (1-2 * nyu) / (1+nyu)
    
    boundLimits = lambda cond, type: [bounds[cond[0], type], bounds[cond[0] + 1, type]]

    dirichletPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                    if min(boundLimits(cond, 0)) <= val[0] <= max(boundLimits(cond, 0)) and
                       min(boundLimits(cond, 1)) <= val[1] <= max(boundLimits(cond, 1))], cond[1:]] for cond in dirichlet_conditions]

    neumannPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                    if min(boundLimits(cond, 0)) <= val[0] <= max(boundLimits(cond, 0)) and
                       min(boundLimits(cond, 1)) <= val[1] <= max(boundLimits(cond, 1))], cond[1], cond[2]] for cond in neumann_conditions]

    areaBoundaryPoints = [idx for idx, val in enumerate(areaPoints_Coords) if val[0] in [bounds[0, 0], bounds[1, 0]] or val[1] in [bounds[0, 1], bounds[2, 1]]]
    areaPoints = [x for x in range(len(areaPoints_Coords))]

    #dirichletPointsAll = sorted(list(set(sum([side[0] for side in dirichletPoints], []))))
    #neumannPointsAll = sorted(list(set(sum([side[0] for side in neumannPoints], []))))
    #areaBoundaryPoints_Coords = [areaPoints_Coords[x] for x in areaBoundaryPoints]

    F = np.zeros(areaPoints_Coords.size)
    
    K = calculateK(areaElements, areaPoints_Coords, D, dimTask)

    print(K)

    for condition in dirichletPoints:
        for node in condition[0]:
            if condition[1][0] == 2:
                K, F = boundCondition_Dirichlet(K, F, dimTask, node, condition[1][1], 0)
                K, F = boundCondition_Dirichlet(K, F, dimTask, node, condition[1][2], 1)
            else:
                K, F = boundCondition_Dirichlet(K, F, dimTask, node, condition[1][1], condition[1][0])
    
    
    for condition in neumannPoints:
        for element in [element for element in areaElements for x in list(combinations(condition[0], 2)) if x[0] in element and x[1] in element]:
            F = boundCondition_Neumann(F, element, condition[0], dimTask, condition[1], areaPoints_Coords, 0)
            F = boundCondition_Neumann(F, element, condition[0], dimTask, condition[2], areaPoints_Coords, 1)
    
    *arg, = np.linalg.solve(K, F)
    u = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))
    
    Eps = []
    for element in areaElements:
        B, _ = CalculateStiffnessMatrix(element, areaPoints_Coords + u, dimTask)
        Eps.append(np.dot(B, np.ravel(np.array([u[element[0]], u[element[1]], u[element[2]]])).transpose()))
    Eps = np.array(Eps).transpose()
    Sigma = np.dot(D, Eps)

    graph = PlotDisplacements(areaPoints_Coords, areaElements, coef_u)

    return u, Eps, Sigma, graph

def mainProgramSparse(nameFile, func = sparse.linalg.cg):
    """
    Программа возвращает:
    u_Current, Eps, Sigma, graph
    """
    *arg, = readInputFile(nameFile)
    
    bounds               = np.array(arg[0])
    areaPoints_Coords    = np.array(arg[1])
    areaElements         = arg[2]
    dirichlet_conditions = arg[3]
    neumann_conditions   = arg[4]
    dimTask              = arg[5]
    splitCoef            = arg[6]
    coefOverlap          = arg[7]
    amntSubdomains       = arg[8]
    E                    = arg[9]
    nyu                  = arg[10]
    coef_u               = arg[11]
    coef_sigma           = arg[12]

    D = np.array([[1, nyu/(1-nyu), 0], [nyu/(1-nyu), 1, 0], [0, 0, (1-2 * nyu) / 2 / (1-nyu)]]) * E * (1-nyu) / (1-2 * nyu) / (1+nyu)
    
    boundLimits = lambda cond, type: [bounds[cond[0], type], bounds[cond[0] + 1, type]]

    dirichletPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                    if min(boundLimits(cond, 0)) <= val[0] <= max(boundLimits(cond, 0)) and
                       min(boundLimits(cond, 1)) <= val[1] <= max(boundLimits(cond, 1))], cond[1:]] for cond in dirichlet_conditions]

    neumannPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                    if min(boundLimits(cond, 0)) <= val[0] <= max(boundLimits(cond, 0)) and
                       min(boundLimits(cond, 1)) <= val[1] <= max(boundLimits(cond, 1))], cond[1], cond[2]] for cond in neumann_conditions]

    areaBoundaryPoints = [idx for idx, val in enumerate(areaPoints_Coords) if val[0] in [bounds[0, 0], bounds[1, 0]] or val[1] in [bounds[0, 1], bounds[2, 1]]]
    areaPoints = [x for x in range(len(areaPoints_Coords))]

    #dirichletPointsAll = sorted(list(set(sum([side[0] for side in dirichletPoints], []))))
    #neumannPointsAll = sorted(list(set(sum([side[0] for side in neumannPoints], []))))
    #areaBoundaryPoints_Coords = [areaPoints_Coords[x] for x in areaBoundaryPoints]

    F = np.zeros(areaPoints_Coords.size)
    
    row, col, data = CalculateMatrixKInfo(areaElements, areaPoints_Coords, D, dimTask)

    K = lil_matrix(coo_matrix((data, (row, col)), shape = (F.size, F.size)))

    for condition in dirichletPoints:
        for node in condition[0]:
            if condition[1][0] == 2:
                K, F = boundCondition_Dirichlet(K, F, dimTask, node, condition[1][1], 0)
                K, F = boundCondition_Dirichlet(K, F, dimTask, node, condition[1][2], 1)
            else:
                K, F = boundCondition_Dirichlet(K, F, dimTask, node, condition[1][1], condition[1][0])
    
    for condition in neumannPoints:
        for element in [element for element in areaElements for x in list(combinations(condition[0], 2)) if x[0] in element and x[1] in element]:
            F = boundCondition_Neumann(F, element, condition[0], dimTask, condition[1], areaPoints_Coords, 0)
            F = boundCondition_Neumann(F, element, condition[0], dimTask, condition[2], areaPoints_Coords, 1)
    
    *arg, = func(K.tocsr(), F)

    u = np.array(arg[0]).reshape(-1, 2) if len(arg) == 2 else np.reshape(arg, (-1, 2))

    Eps = CalculateEps(areaElements, areaPoints_Coords, dimTask, u_Current, coef_u)
    Sigma = np.dot(D, Eps.transpose())

    graph = PlotDisplacements(areaPoints_Coords, areaElements, coef_u)

    return u, Eps, Sigma, graph


if __name__ == "__main__":
    nameFile = "test_1_30_2.dat"
    func = sparse.linalg.spsolve
    coefConvergence = 1e-3

    @benchmark
    def SolveTask(mainTask, nameFile, func):
        u, Eps, Sigma, graph = mainTask(nameFile, func)
        return u, Eps, Sigma, graph

    @benchmarkSchwarz
    def SolveTaskSchwarz(SchwarzTask, nameFile, coefConvergence, func):
        u, Eps, Sigma, graph, it = SchwarzTask(nameFile, coefConvergence, func)
        print(f"{min(Sigma[1]):.3g}, {max(Sigma[1]):.3g}")
        graph(u)
        return u, Eps, Sigma, graph, it
    
    u, Eps, Sigma, graph, it = SolveTaskSchwarz(SchwarzMultiplicativeProgram, "test_1_30_4.dat",coefConvergence ,func)
    u, Eps, Sigma, graph, it = SolveTaskSchwarz(SchwarzAdditiveProgram, "test_1_30_4.dat",coefConvergence ,func)
    u, Eps, Sigma, graph, it = SolveTaskSchwarz(SchwarzMultiplicativeProgram, "test_2_30_4.dat",coefConvergence ,func)
    u, Eps, Sigma, graph, it = SolveTaskSchwarz(SchwarzAdditiveProgram, "test_2_30_4.dat",coefConvergence ,func)

