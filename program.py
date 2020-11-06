bounds = []
dirichlet_conditions = []
neumann_conditions = []
points = []
elements = []
with open("test_1.dat") as f:
    dimTask = int(f.readline())
    for _ in range(int(f.readline())):
       bounds.append([float(x) for x in f.readline().split()])
    splitCoef = float(f.readline())
    coefOverlap = float(f.readline())
    amntSubdomains = list(map(int, f.readline().split()))
    E, nyu = list(map(float, f.readline().split()))
    coef_u, coef_sigma = list(map(float, f.readline().split()))
    for _ in range(int(f.readline())):
        dirichlet_conditions.append([int(x) for x in f.readline().split()])
    for _ in range(int(f.readline())):
        neumann_conditions.append([int(val) if idx == 0 else float(val) for idx, val in enumerate(f.readline().split())])
    for _ in range(int(f.readline())):
        points.append([float(val) for val in f.readline().split()])
    for _ in range(int(f.readline())):
        elements.append([int(val) for val in f.readline().split()])

    boundaryPoints = 
    
