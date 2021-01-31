import time
from modules.class_test import Test

# def benchmark(func):
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         u, Eps, Sigma, graph = func(*args, **kwargs)
#         end_time = time.time()
#         print(f"Время выполнения обычной задачи: {end_time - start_time}\n")
#         return u, Eps, Sigma, graph
#     return wrapper

# def benchmark_schwarz(func):
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         u, Eps, Sigma, graph, it = func(*args, **kwargs)
#         end_time = time.time()
#         print(f"Время выполнения методом Шварца: {end_time - start_time}\n"
#               f"Количество выполненных итераций: {it}\n"
#               f"min={abs(abs(min(Sigma[1])) - abs(2e+7)):.4e}, max={abs(abs(max(Sigma[1])) - abs(2e+7)):.4e}")
#         return u, Eps, Sigma, graph, it
#     return wrapper

if __name__ == "__main__":
    obj = Test(1, 5)
    obj.get_solution()
    print(obj.Sigma)
    obj.plot_displacements()
    # cur_task = 1
    # cur_mesh = 4
    # cur_amnt_subds = [2, 1]
    # coef_alpha = 0.5
    # func = sparse.linalg.spsolve
    # coef_convergence = 1e-3
    # cur_coarse_mesh = 6

    # @benchmark
    # def basic_task(method, cur_task, cur_mesh, func):
    #     u, Eps, Sigma, graph = method(cur_task, cur_mesh, func)
    #     return u, Eps, Sigma, graph

    # @benchmark_schwarz
    # def schwarz_task(method, cur_task, cur_mesh, cur_amnt_subds, coef_convergence, func, *args):
    #     u, Eps, Sigma, graph, it = method(cur_task, cur_mesh, cur_amnt_subds, coef_convergence, func, args)
    #     return u, Eps, Sigma, graph, it

    # u, *arg, graph, it = schwarz_task(schwarz_two_level_additive_method, cur_task, cur_mesh, cur_amnt_subds, coef_convergence, func, coef_alpha, cur_coarse_mesh)
    # #graph(u, False)