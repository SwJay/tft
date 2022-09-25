import os
from multiprocessing import Pool, cpu_count
import time
import sys
import numpy as np
import network as nx
import matplotlib.pyplot as plt
from scipy.stats import norm, fisk, weibull_min
from scipy.optimize import minimize, differential_evolution, Bounds
# from tabulate import tabulate
import traceback


'''
Announcement:
1. use function ended with '_all', global optimize over positions for all nodes
2. pos & dis are expanded to match given median propagation time
'''


# graph

def exp_mean_dis(graph, m_pow):
    n = len(graph)
    res = 0
    for i in range(n-1):
        for j in range(n):
            res += m_pow[i] * graph[i][j]
    res = res / (n-1)
    return res


def exp_med_dis(graph, power):
    # remove diagonal since they are all 0's
    n = len(graph)
    alpha = power[n-1]
    pos_dis = graph[~np.eye(n, dtype=bool)].reshape(n, -1)[:-1,]

    m_pow = power[:n - 1] / (1 - alpha)
    a = np.random.choice(np.arange(n - 1), size=1000, replace=True, p=m_pow)
    counts = np.bincount(a, minlength=n-1)

    assert(len(pos_dis) == len(counts))

    res = pos_dis.repeat(counts, axis=0)
    med = np.median(res)
    return med


def compute_gamma(pos, m_pow, cv):
    n = len(pos)
    dif = pos[:, None, :] - pos[None, :, :]
    dis = np.sqrt(np.sum(dif ** 2, axis=-1))  # euclidean

    sigma = exp_mean_dis(dis, m_pow) * cv # k_dis * dis = time

    gamma = 0
    for i in range(n - 1):
        for j in range(n - 1):
            if i != j:
                gamma += m_pow[i] * m_pow[j] * norm.cdf(dis[i][j] - dis[i][n - 1], scale=np.sqrt(2) * sigma)
    return gamma


def gap(x, n, m_pow, pos, target, cv):
    # get dis mat
    pos[n - 1][0] = x[0]
    pos[n - 1][1] = x[1]

    # compute gamma
    gamma = compute_gamma(pos, m_pow, cv)

    return abs(gamma - target)


# best
def _trust_constr(num_node, m_pow, pos, target_g, cv):

    # lower & upper bound
    lb_b = np.ones(2)
    ub_b = np.ones(2)
    bounds = Bounds(lb_b, ub_b)

    # args
    args = (num_node, m_pow, pos, target_g, cv)
    x0 = np.random.rand(2)
    res = minimize(gap, x0, args=args, method='trust-constr', bounds=bounds, tol=1e-2)

    return res.x


def _diff_evol(num_node, m_pow, pos, target_g):
    print('diff evol')

    # lower & upper bound
    lb_b = np.zeros(2)
    ub_b = np.ones(2)
    bounds = Bounds(lb_b, ub_b)

    # args
    args = (num_node, m_pow, pos, target_g)

    start = time.process_time()
    res = differential_evolution(gap, bounds, args=args, tol=0.01)
    end = time.process_time()
    print(end - start)

    print(res.fun)


# pos for all nodes (h&sm) are computed from optimized result to match given gamma
# randomness come from power distribution and random initial point selection

def gap_all(x, n, m_pow, target, cv):
    # get dis mat
    pos = np.zeros((n, 2))
    for i in range(n):
        for j in range(2):
            pos[i][j] = x[2 * i + j]

    # compute gamma
    gamma = compute_gamma(pos, m_pow, cv)

    return abs(gamma - target)


def _trust_constr_all(num_node, m_pow, target_g, cv):

    # x is 1-d with shape (n,)
    # lower & upper bound
    lb_b = np.zeros(num_node * 2)
    ub_b = np.ones(num_node * 2)
    bounds = Bounds(lb_b, ub_b)

    # args
    args = (num_node, m_pow, target_g, cv)
    x0 = np.random.rand(num_node * 2)
    res = minimize(gap_all, x0, args=args, method='trust-constr', bounds=bounds, tol=1e-5)

    return res.x.reshape(num_node, 2)


def _new_graph_all(num_node, power, alpha, target_g, cv, mediantime, tol=5e-2):
    # generate random complete graph ~ N
    if alpha == 0:  # random generate network
        # pos = np.random.randint(1, MAX_DIS, size=(num, dim))
        pos = np.random.rand(num_node, 2)

    else:  # specified alpha and gamma
        m_pow = power[:num_node-1] / (1 - alpha)  # modified power

        count = 0
        while True:
            count += 1
            print('generate network ... shot {}'.format(count))

            # start = time.process_time()
            pos = _trust_constr_all(num_node, m_pow, target_g, cv)
            # end = time.process_time()
            # print('time: {}'.format(end - start))

            g = compute_gamma(pos, m_pow, cv)
            if abs(g - target_g) < tol:
                break

    # match median time
    dif = pos[:, None, :] - pos[None, :, :]
    dis = np.sqrt(np.sum(dif ** 2, axis=-1))  # euclidean

    # dis_med_1 = median_dis(dis)
    # print(dis_med_1)
    dis_med_2 = exp_med_dis(dis, power)
    # print(dis_med_2)

    # new pos, dis
    pos = pos / dis_med_2 * mediantime
    dif = pos[:, None, :] - pos[None, :, :]
    dis = np.sqrt(np.sum(dif ** 2, axis=-1))  # euclidean

    return pos, dis


def _show_graph(pos, dis):
    # create a weighted, undirected graph in networkx
    graph = nx.from_numpy_matrix(dis, create_using=nx.Graph())
    nx.draw(graph, pos, with_labels=True)
    labels = {e: graph.edges[e]['weight'] for e in graph.edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.show()


def get_graph(num_node, is_wg, power, alpha, gamma, cv, mediantime, case):
    full_path = 'data/{}/{}-{}'.format(num_node, alpha, gamma)
    pos_path = full_path + '/pos.csv'
    dis_path = full_path + '/dis.csv'

    if case >= 0:
        dis_path = 'batch-data/{}-{}-{}/{:.2}-{:.2}/{}-dis.csv'.format(num_node, cv, mediantime, alpha, gamma, case)
        dis = np.loadtxt(dis_path)

    elif is_wg or not os.path.exists(pos_path) or not os.path.exists(dis_path):
        pos, dis = _new_graph_all(num_node, power, alpha, gamma, cv, mediantime)
        # _show_graph(pos, dis)
        np.savetxt(pos_path, pos)
        np.savetxt(dis_path, dis)
    else:
        pos = np.loadtxt(pos_path)
        dis = np.loadtxt(dis_path)
        # _show_graph(pos, dis)

    return dis


# power

def get_power(num_node, is_wp, is_i, alpha, gamma, case, t_alpha=0, cv=0.02, mediantime=8.7):
    if case >= 0:
        pow_path = 'batch-data/{}-{}-{}/{:.2}-{:.2}/{}-pow.csv'.format(num_node, cv, mediantime, alpha, gamma, case)
        pow = np.loadtxt(pow_path)
        if t_alpha > 0:
            pow = pow / (1-alpha) * (1-t_alpha)
            pow[num_node-1] = t_alpha
        return pow

    elif is_i:
        return np.ones(num_node) / num_node
    else:
        full_path = 'data/{}/{}-{}'.format(num_node, alpha, gamma)
        pow_path = full_path + '/pow.csv'

        if not os.path.exists(full_path):
            os.makedirs(full_path)
            if alpha == 0:  # random, pow[num_node-1] is the largest
                pow = np.random.dirichlet(np.ones(num_node), size=1)[0]
                max_id = np.argmax(pow)
                pow[max_id], pow[num_node-1] = pow[num_node-1], pow[max_id]
            else:           # specified alpha, pow[num_node-1] is the alpha
                pow = np.random.dirichlet(np.ones(num_node - 1), size=1)[0] * (1 - alpha)
                pow = np.append(pow, alpha)

            np.savetxt(pow_path, pow)
        elif is_wp:
            if alpha == 0:  # random, pow[num_node-1] is the largest
                pow = np.random.dirichlet(np.ones(num_node), size=1)[0]
                max_id = np.argmax(pow)
                pow[max_id], pow[num_node - 1] = pow[num_node - 1], pow[max_id]
            else:  # specified alpha, pow[num_node-1] is the alpha
                pow = np.random.dirichlet(np.ones(num_node - 1), size=1)[0] * (1 - alpha)
                pow = np.append(pow, alpha)

            # write to disk
            np.savetxt(pow_path, pow)
        else:
            # read from disk
            pow = np.loadtxt(pow_path)

        return pow


def test_AG():
    num_node = 16
    # alpha_list = [0.1, 0.2, 0.3, 0.4, 0.45]
    alpha_list = [0.3]
    gamma_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    for alpha in alpha_list:
        power = np.random.dirichlet(np.ones(num_node - 1), size=1)[0] * (1 - alpha)
        power = np.append(power, alpha)
        m_pow = power[:num_node - 1] / (1 - alpha)
        print(power)
        for gamma in gamma_list:
            pos, _ = _new_graph_all(num_node, power, alpha, gamma)
            g = compute_gamma(pos, m_pow, cv)
            print('alpha:{}\ttarget_g:{}\tres_g:{:.4}\terr:{:.4}'.format(alpha, gamma, g, abs(g - gamma)))


def test_maxmin():
    num_node_list = [8]
    alpha_list = [0.2]
    iters = 5

    cv = 0.001
    mt = 8.7

    headers = ['num\\alpha [min,max]'] + alpha_list
    table = []

    # min: 0, max: 1
    target_g = [0, 1]

    for i, n in enumerate(num_node_list):
        table.append([n])
        for alpha in alpha_list:
            # init, min: 1, max: 0
            best_g = [1, 0]
            for k in range(iters):
                power = np.random.dirichlet(np.ones(n - 1), size=1)[0] * (1 - alpha)
                power = np.append(power, alpha)
                m_pow = power[:n - 1] / (1 - alpha)

                for j, tg in enumerate(target_g):
                    pos, _ = _new_graph_all(n, power, alpha, tg, cv, mt)
                    cur_g = compute_gamma(pos, m_pow, cv)
                    if abs(best_g[j] - target_g[j]) > abs(cur_g - target_g[j]):
                        best_g[j] = cur_g
            table[i].append(best_g)


def single_write(n, alpha, gamma, cases, cv, mt):
    full_path = 'batch-data/{}-{}-{}/{:.2}-{:.2}'.format(n, cv, mt, alpha, gamma)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    for case in range(cases):
        pow_path = '{}/{}-pow.csv'.format(full_path, case)
        pos_path = '{}/{}-pos.csv'.format(full_path, case)
        dis_path = '{}/{}-dis.csv'.format(full_path, case)

        # pow
        pow = np.random.dirichlet(np.ones(n - 1), size=1)[0] * (1 - alpha)
        pow = np.append(pow, alpha)
        np.savetxt(pow_path, pow)

        # graph
        try:
            pos, dis = _new_graph_all(n, pow, alpha, gamma, cv, mt)
        except:
            current_filename = str(os.path.basename(sys.argv[0]))[:-3]
            cur_err_filname = current_filename + '_error.txt'
            error_info = sys.exc_info()
            with open(f'{cur_err_filname}', 'a') as f:
                error_str = f'{alpha}-{gamma}-case{case}:ERROR OCCURRED,{time.strftime("%Y-%m-%d %H:%M:%S")}:\n {error_info[0]}: {error_info[1]}'
                print(error_str, file=f)
                traceback.print_tb(error_info[2], file=f)
                f.write(f"{'=' * 50}\n")
        np.savetxt(pos_path, pos)
        np.savetxt(dis_path, dis)

        print('\033[45m {}-{}/{} done \033[0m'.format(alpha, gamma, case))


def batch_write(cases, n=16, cv=0.02, mt=8.7):
    alpha_list = np.round(np.arange(1,10) * 0.05, decimals=2)
    gamma_list = np.round(np.arange(1, 15) * 0.05, decimals=2)

    dir_path = 'batch-data/{}-{}-{}'.format(n, cv, mt)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    start = time.time()

    p = Pool(cpu_count())

    for alpha in alpha_list:
        for gamma in gamma_list:
            p.apply_async(single_write, args=(n, alpha, gamma, cases, cv, mt,))
    p.close()

    p.join()

    end = time.time()
    print('time: {}'.format(end - start))


def histogram():
    n = 16
    # draw histogram of dis distribution
    pos = np.random.rand(n, 2)
    dif = pos[:, None, :] - pos[None, :, :]
    dis = np.sqrt(np.sum(dif ** 2, axis=-1))  # euclidean
    dis = dis[~np.eye(n, dtype=bool)].reshape(n, -1)[:-1,]
    mean = np.mean(dis)
    median = np.median(dis)
    print('mean: {}\tmedian: {}'.format(mean, median))
    dis_1d = dis.flatten()
    plt.hist(dis_1d, bins=50)
    plt.show()

if __name__ == '__main__':
    cases = 10 if len(sys.argv) == 1 else sys.argv[1]
    batch_write(int(cases))