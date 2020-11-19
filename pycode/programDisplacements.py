import numpy as np
from itertools import combinations
from itertools import groupby
import matplotlib.pyplot as plt
import sys
import os
import pprint

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

    F -= np.copy(K[:, node * dimTask + dim]).reshape(F.shape) * value
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
               f"Количество подобластей - [{amntSubdomains[0]}, {amntSubdomains[1]}] \n"
               f"Модуль Юнга - {E} Па, коэффициент Пуассона - {nyu} \n"
               f"Коэффициент масштаба для перемещения - {coef_u}, для напряжений - {coef_sigma} \n")
    return message

def SchwarzMultiplicativeProgram(nameFile):
    """
    Программа возвращает:
    u_global, Eps, Sigma, graph, it
    """
    bounds, areaPoints_Coords, areaElements, dirichlet_conditions, neumann_conditions, dimTask, bounds, splitCoef, coefOverlap, amntSubdomains, E, nyu, coef_u, coef_sigma = readInputFile(nameFile)
    
    bounds = np.array(bounds)
    areaPoints_Coords = np.array(areaPoints_Coords)

    areaPoints = [x for x in range(len(areaPoints_Coords))]
    areaBoundaryPoints = [idx for idx, val in enumerate(areaPoints_Coords) if val[0] in [bounds[0, 0], bounds[1, 0]] or val[1] in [bounds[0, 1], bounds[2, 1]]]

    dirichletPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                     if min([bounds[cond[0], 0], bounds[cond[0] + 1, 0]]) <= val[0] <= max([bounds[cond[0], 0], bounds[cond[0] + 1, 0]]) and
                        min([bounds[cond[0], 1], bounds[cond[0] + 1, 1]]) <= val[1] <= max([bounds[cond[0], 1], bounds[cond[0] + 1, 1]])], cond[1:]] for cond in dirichlet_conditions]

    neumannPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                     if min([bounds[cond[0], 0], bounds[cond[0] + 1, 0]]) <= val[0] <= max([bounds[cond[0], 0], bounds[cond[0] + 1, 0]])
                     and min([bounds[cond[0], 1], bounds[cond[0] + 1, 1]]) <= val[1] <= max([bounds[cond[0], 1], bounds[cond[0] + 1, 1]])], cond[1], cond[2]] for cond in neumann_conditions]

    dirichletPointsAll = sorted(list(set(sum([side[0] for side in dirichletPoints], []))))
    neumannPointsAll = sorted(list(set(sum([side[0] for side in neumannPoints], []))))

    relation_PointsElements = {}
    boundPoints = []
    subdElements = []
    subdPoints = []
    subdPoints_Coords = []
    subdInternalPoints = []
    subdBoundary = []

    subdBounds = [bounds[0][0]] + [bounds[0][0] + i*(bounds[1][0] - bounds[0][0])/amntSubdomains[0] for i in range(1, amntSubdomains[0])] + [bounds[1][0]]
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

    backlog = []
    it = 0
    u_global = np.zeros((areaPoints_Coords.shape[0], 2))
    D = np.array([[1, nyu/(1-nyu), 0], [nyu/(1-nyu), 1, 0], [0, 0, (1-2 * nyu) / 2 / (1-nyu)]]) * E * (1-nyu) / (1-2 * nyu) / (1+nyu)
    while True:
        u_Previous = np.copy(u_global)
        for idv, subd in enumerate(subdElements):

            ratioPoints_LocalGlobal = dict(zip(range(len(subdPoints[idv])), subdPoints[idv]))
            ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}
        
            K_global = np.zeros((len(ratioPoints_GlobalLocal) * 2, len(ratioPoints_GlobalLocal) * 2))
            F = np.zeros((len(ratioPoints_GlobalLocal) * 2, 1))
            subdElements_Local = np.array([ratioPoints_GlobalLocal[x] for x in np.array(subd).ravel()]).reshape(len(subd), 3)

            for element in subdElements_Local:
                B, A = CalculateStiffnessMatrix(element, subdPoints_Coords[idv], dimTask)
                K = np.dot(np.dot(B.transpose(), D), B) * A
                for i in range(3):
                    for j in range(3):
                        K_global[element[i]*dimTask:element[i]*dimTask+2, element[j]*dimTask:element[j]*dimTask+2] += K[i*dimTask:i*dimTask+2,j*dimTask:j*dimTask+2]

            for condition in dirichletPoints:
                listPoints = list(set(condition[0]) & set(subdPoints[idv]))
                for node in listPoints:
                    if condition[1][0] == 2:
                        K_global, F = boundCondition_Dirichlet(K_global, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][1], 0)
                        K_global, F = boundCondition_Dirichlet(K_global, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][2], 1)
                    else:
                        K_global, F = boundCondition_Dirichlet(K_global, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][1], condition[1][0])

            for condition in neumannPoints:
                listPoints = list(set(condition[0]) & set(subdPoints[idv]))
                segmentPoints = list(combinations(listPoints, 2))
                for element in [element for element in subd for x in segmentPoints if x[0] in element and x[1] in element]:                
                    F = boundCondition_Neumann(F, [ratioPoints_GlobalLocal[x] for x in element], [ratioPoints_GlobalLocal[x] for x in listPoints], dimTask, condition[1], subdPoints_Coords[idv], 0)
                    F = boundCondition_Neumann(F, [ratioPoints_GlobalLocal[x] for x in element], [ratioPoints_GlobalLocal[x] for x in listPoints], dimTask, condition[2], subdPoints_Coords[idv], 1)
        
            listPoints_Schwarz = sum([list(set(subdBoundary[idv]) & set(subd)) for idx, subd in enumerate(subdPoints) if idx != idv], [])

            for node in listPoints_Schwarz:
                K_global, F = boundCondition_Dirichlet(K_global, F, dimTask, ratioPoints_GlobalLocal[node], u_global[node, 0], dim = 0)
                K_global, F = boundCondition_Dirichlet(K_global, F, dimTask, ratioPoints_GlobalLocal[node], u_global[node, 1], dim = 1)
            
            u_local = np.linalg.solve(K_global, F).reshape((subdPoints_Coords[idv].shape[0], 2))     
            for x in list(ratioPoints_LocalGlobal.keys()):
                u_global[ratioPoints_LocalGlobal[x], :] = np.copy(u_local[x, :])

            if it == 0:
                backlog.append(np.copy(u_global))
        it += 1
        if np.linalg.norm(np.copy(u_global) - np.copy(u_Previous)) < 1e-7:
            break
    
    Eps=[]
    for element in areaElements:
        B, A = CalculateStiffnessMatrix(element, areaPoints_Coords, dimTask)
        Eps.append(np.dot(B, np.ravel(np.array([u_global[element[0]], u_global[element[1]], u_global[element[2]]])).transpose()))
    Eps = np.array(Eps)
    Sigma = np.dot(D, Eps.transpose())
    graph = PlotDisplacements(areaPoints_Coords, areaElements, coef_u)

    return u_global, Eps, Sigma, graph, it

def SchwarzAdditiveProgram(nameFile):
    """
    Программа возвращает:
    u_global, Eps, Sigma, graph, it
    """
    bounds, areaPoints_Coords, areaElements, dirichlet_conditions, neumann_conditions, dimTask, bounds, splitCoef, coefOverlap, amntSubdomains, E, nyu, coef_u, coef_sigma = readInputFile(nameFile)
    
    bounds = np.array(bounds)
    areaPoints_Coords = np.array(areaPoints_Coords)

    areaPoints = [x for x in range(len(areaPoints_Coords))]
    areaBoundaryPoints = [idx for idx, val in enumerate(areaPoints_Coords) if val[0] in [bounds[0, 0], bounds[1, 0]] or val[1] in [bounds[0, 1], bounds[2, 1]]]

    dirichletPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                     if min([bounds[cond[0], 0], bounds[cond[0] + 1, 0]]) <= val[0] <= max([bounds[cond[0], 0], bounds[cond[0] + 1, 0]]) and
                        min([bounds[cond[0], 1], bounds[cond[0] + 1, 1]]) <= val[1] <= max([bounds[cond[0], 1], bounds[cond[0] + 1, 1]])], cond[1:]] for cond in dirichlet_conditions]

    neumannPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                     if min([bounds[cond[0], 0], bounds[cond[0] + 1, 0]]) <= val[0] <= max([bounds[cond[0], 0], bounds[cond[0] + 1, 0]])
                     and min([bounds[cond[0], 1], bounds[cond[0] + 1, 1]]) <= val[1] <= max([bounds[cond[0], 1], bounds[cond[0] + 1, 1]])], cond[1], cond[2]] for cond in neumann_conditions]

    dirichletPointsAll = sorted(list(set(sum([side[0] for side in dirichletPoints], []))))
    neumannPointsAll = sorted(list(set(sum([side[0] for side in neumannPoints], []))))

    relation_PointsElements = {}
    boundPoints = []
    subdElements = []
    subdPoints = []
    subdPoints_Coords = []
    subdInternalPoints = []
    subdBoundary = []

    subdBounds = [bounds[0][0]] + [bounds[0][0] + i*(bounds[1][0] - bounds[0][0])/amntSubdomains[0] for i in range(1, amntSubdomains[0])] + [bounds[1][0]]
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

    graph_backlog = PlotBacklog(areaPoints_Coords, areaElements, coef_u)

    graph = PlotDisplacements(areaPoints_Coords, areaElements, coef_u)

    backlog = []
    alpha = 0.5
    it = 0
    u_global = np.zeros((areaPoints_Coords.shape[0], 2))
    u_sum = []
    D = np.array([[1, nyu/(1-nyu), 0], [nyu/(1-nyu), 1, 0], [0, 0, (1-2 * nyu) / 2 / (1-nyu)]]) * E * (1-nyu) / (1-2 * nyu) / (1+nyu)
    while True:
        u_Previous = np.copy(u_global)
        u_global = np.zeros_like(u_Previous)
        u_temp = np.zeros_like(u_Previous)
        for idv, subd in enumerate(subdElements):
            u_temp = np.copy(u_Previous)

            ratioPoints_LocalGlobal = dict(zip(range(len(subdPoints[idv])), subdPoints[idv]))
            ratioPoints_GlobalLocal = {v: k for k, v in ratioPoints_LocalGlobal.items()}
        
            K_global = np.zeros((len(ratioPoints_GlobalLocal) * 2, len(ratioPoints_GlobalLocal) * 2))
            F = np.zeros((len(ratioPoints_GlobalLocal) * 2, 1))
            subdElements_Local = np.array([ratioPoints_GlobalLocal[x] for x in np.array(subd).ravel()]).reshape(len(subd), 3)

            for element in subdElements_Local:
                B, A = CalculateStiffnessMatrix(element, subdPoints_Coords[idv], dimTask)
                K = np.dot(np.dot(B.transpose(), D), B) * A
                for i in range(3):
                    for j in range(3):
                        K_global[element[i]*dimTask:element[i]*dimTask+2, element[j]*dimTask:element[j]*dimTask+2] += K[i*dimTask:i*dimTask+2,j*dimTask:j*dimTask+2]

            for condition in dirichletPoints:
                listPoints = list(set(condition[0]) & set(subdPoints[idv]))
                for node in listPoints:
                    if condition[1][0] == 2:
                        K_global, F = boundCondition_Dirichlet(K_global, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][1], 0)
                        K_global, F = boundCondition_Dirichlet(K_global, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][2], 1)
                    else:
                        K_global, F = boundCondition_Dirichlet(K_global, F, dimTask, ratioPoints_GlobalLocal[node], condition[1][1], condition[1][0])

            for condition in neumannPoints:
                listPoints = list(set(condition[0]) & set(subdPoints[idv]))
                segmentPoints = list(combinations(listPoints, 2))
                for element in [element for element in subd for x in segmentPoints if x[0] in element and x[1] in element]:                
                    F = boundCondition_Neumann(F, [ratioPoints_GlobalLocal[x] for x in element], [ratioPoints_GlobalLocal[x] for x in listPoints], dimTask, condition[1], subdPoints_Coords[idv], 0)
                    F = boundCondition_Neumann(F, [ratioPoints_GlobalLocal[x] for x in element], [ratioPoints_GlobalLocal[x] for x in listPoints], dimTask, condition[2], subdPoints_Coords[idv], 1)
        
            listPoints_Schwarz = sum([list(set(subdBoundary[idv]) & set(subd)) for idx, subd in enumerate(subdPoints) if idx != idv], [])

            for node in listPoints_Schwarz:
                K_global, F = boundCondition_Dirichlet(K_global, F, dimTask, ratioPoints_GlobalLocal[node], u_Previous[node, 0], dim = 0)
                K_global, F = boundCondition_Dirichlet(K_global, F, dimTask, ratioPoints_GlobalLocal[node], u_Previous[node, 1], dim = 1)
            
            u_local = np.linalg.solve(K_global, F).reshape((subdPoints_Coords[idv].shape[0], 2))

            for x in list(ratioPoints_LocalGlobal.keys()):
                u_temp[ratioPoints_LocalGlobal[x], :] = np.copy(u_local[x, :])

            backlog.append(np.copy(u_temp))

            u_sum.append(u_temp - u_Previous)
        #graph_backlog(np.copy(backlog))  
        #for val in u_sum:
        #    graph(val)
        it += 1
        #graph(u_Previous)
        u_global = np.copy(u_Previous) + alpha * (sum(u_sum))
        #graph(u_global)
        backlog.clear()
        u_sum.clear()

        if np.linalg.norm(np.copy(u_global) - np.copy(u_Previous)) < 1e-7:
        #if it == 5:
            break
    
    Eps=[]
    for element in areaElements:
        B, A = CalculateStiffnessMatrix(element, areaPoints_Coords, dimTask)
        Eps.append(np.dot(B, np.ravel(np.array([u_global[element[0]], u_global[element[1]], u_global[element[2]]])).transpose()))
    Eps = np.array(Eps)
    Sigma = np.dot(D, Eps.transpose())

    return u_global, Eps, Sigma, graph, it

def mainProgram(nameFile):
    """
    Программа возвращает:
    u_global, Eps, Sigma, graph
    """
    bounds, areaPoints_Coords, areaElements, dirichlet_conditions, neumann_conditions, dimTask, bounds, splitCoef, coefOverlap, amntSubdomains, E, nyu, coef_u, coef_sigma = readInputFile(nameFile)
    
    bounds = np.array(bounds)
    areaPoints_Coords = np.array(areaPoints_Coords)
    areaPoints = [x for x in range(len(areaPoints_Coords))]
    areaBoundaryPoints = [idx for idx, val in enumerate(areaPoints_Coords) if val[0] in [bounds[0, 0], bounds[1, 0]] or val[1] in [bounds[0, 1], bounds[2, 1]]]
    areaBoundaryPoints_Coords = [areaPoints_Coords[x] for x in areaBoundaryPoints]
    
    dirichletPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                     if min([bounds[cond[0], 0], bounds[cond[0] + 1, 0]]) <= val[0] <= max([bounds[cond[0], 0], bounds[cond[0] + 1, 0]]) and
                        min([bounds[cond[0], 1], bounds[cond[0] + 1, 1]]) <= val[1] <= max([bounds[cond[0], 1], bounds[cond[0] + 1, 1]])], cond[1:]] for cond in dirichlet_conditions]

    neumannPoints = [[[idx for idx, val in enumerate(areaPoints_Coords) 
                     if min([bounds[cond[0], 0], bounds[cond[0] + 1, 0]]) <= val[0] <= max([bounds[cond[0], 0], bounds[cond[0] + 1, 0]])
                     and min([bounds[cond[0], 1], bounds[cond[0] + 1, 1]]) <= val[1] <= max([bounds[cond[0], 1], bounds[cond[0] + 1, 1]])], cond[1], cond[2]] for cond in neumann_conditions]

    dirichletPointsAll = sorted(list(set(sum([side[0] for side in dirichletPoints], []))))
    neumannPointsAll = sorted(list(set(sum([side[0] for side in neumannPoints], []))))

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
    u_1, Eps_1, Sigma_1, graph = mainProgram("test_1_5_2.dat")
    u_2, * arg, it = SchwarzAdditiveProgram("test_1_5_2.dat")
    graph(u_1)
    graph(u_2)
    print(np.linalg.norm(u_1 - u_2))
    print(it)
    #print(getMessage(Sigma, "Sigma"))
    #graph(u)