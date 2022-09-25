import argparse
from scipy.optimize import minimize, differential_evolution, Bounds, LinearConstraint
import numpy as np
from scipy.stats import norm
from tabulate import tabulate
import time
from multiprocessing import Manager, Pool, cpu_count
import os
import sys
import traceback


'''
0. step
1. ipm 0
2. ipm 1
3. ipm 2
4. de 0
5. de 1
6. de 2
'''

# NUM_LIST = [128]
NUM_LIST = [8, 16, 32, 64, 128]

MAXMU_LIST = [5, 10, 15, 20, 25]

CV_LIST = [0.01, 0.02, 0.04, 0.08, 0.16]

# algo_list = [6]
algo_list = [0, 1, 2, 3, 4, 5, 6]

PARA_LIST = [NUM_LIST, MAXMU_LIST, CV_LIST]
PARA_DICT = {'n': 0, 'mu': 1, 'cv': 2}

ipm_tols = [1e-4, 1e-2, 1e-1]
de_tols = [0.05, 0.1, 0.2]



T_MED = 8.7


def bisect_step(left, right, lb_b, a, b, sigma, res_wspt, maxmu, tol=0.1):
    step = (right + left) / 2
    x = step_algo(lb_b, a, b, sigma, step=step)
    try:
        x_max = np.max(x)
        ub = np.max(lb_b) + maxmu
        if abs(x_max - ub) < tol:
            return step
        elif x_max > ub:
            return bisect_step(left, step, lb_b, a, b, sigma, res_wspt, maxmu, tol)
        else:
            return bisect_step(step, right, lb_b, a, b, sigma, res_wspt, maxmu, tol)
    except:
        current_filename = str(os.path.basename(sys.argv[0]))[:-3]
        cur_err_filname = current_filename + '_error.txt'
        error_info = sys.exc_info()
        with open(f'{cur_err_filname}', 'a') as f:
            error_str = f'ERROR OCCURRED,{time.strftime("%Y-%m-%d %H:%M:%S")}:\n {error_info[0]}: {error_info[1]}'
            print(error_str, file=f)
            print(f'step: {step}, x: {x}, lb: {lb_b}, maxmu: {maxmu}', file=f)
            traceback.print_tb(error_info[2], file=f)
            f.write(f"{'=' * 50}\n")


def get_med(dis, pow):
    # remove diagonal since they are all 0's
    n = len(dis)
    a = np.random.choice(np.arange(n), size=1000, replace=True, p=pow)
    counts = np.bincount(a, minlength=n)

    res = dis.repeat(counts)
    med = np.median(res)
    return med


def get_mean(dis, pow):
    return np.average(dis, weights=pow)


def get_ab(num):
    a = np.random.rand(num)
    p = np.random.rand(num)
    a /= np.sum(a)
    p /= np.sum(p)
    b = p / (1 - a)
    b_a = b / a
    ba_order = np.argsort(b_a)
    a = a[ba_order]
    b = b[ba_order]

    return a, b


def get_x0(lb, delta):
    num = len(lb)
    x0 = np.zeros(num)
    for i in range(num):
        if i != 0:
            x0[i] = max(lb[i] + delta, x0[i-1] + delta)
        else:
            x0[i] = lb[i] + delta
    return x0


def get_A(num):
    A = np.mat(np.zeros((num, num)))
    for i in range(num - 1):
        A[i, i] = 1
        A[i, i+1] = -1

    return A


def fun(x, a, b, sigma):
    # a, b, sigma = args
    n = len(a)
    gamma = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                gamma += b[i] * a[j] * norm.cdf(0, x[i] - x[j], np.sqrt(2) * sigma)
    return gamma


def upper_bound(x, a, b, sigma):
    n = len(a)
    res = 0
    for i in range(n-1):
        for j in range(i+1, n):
            res += norm.cdf(0, x[j] - x[i], np.sqrt(2) * sigma) * (b[j] * a[i]/(b[i] * a[j]) - 1)
    return res


def wspt_res(a, b):
    n = len(a)
    # res = -a.dot(b)
    res = 0
    for i in range(n-1):
        for j in range(i+1,n):
            res += b[i] * a[j]
    return res


def get_beta(a, b):
    n = len(a)
    beta = 0
    bi = 0
    bj = 0
    for i in range(n-1):
        for j in range(i+1,n):
            tmp = b[j]*a[i]-b[i]*a[j]
            if beta < tmp:
                beta = tmp
                bi = i
                bj = j
    return beta

'''
check impact from (n, beta), epsilon on cumper
'''

def get_step(sigma, epsilon, beta, n, cum=0):
    if cum:
        cum_per = cum
    else:
        cum_per = 1-epsilon/(beta * np.math.factorial(n-1))
    print('cummulative percentage: {}'.format(cum_per))
    return np.sqrt(2) * sigma * norm.ppf(cum_per)


# P_12=0, P_23=0, ...
def step_algo(lb, a, b, sigma, epsilon=0, cum=0, step=0):
    num = len(lb)
    x = np.zeros(num)
    if step:
        x_step = step
    else:
        print('deprecated')
        beta = get_beta(a,b)
        x_step = get_step(sigma, epsilon, beta, num, cum)
    for i in range(num):
        if i != 0:
            x[i] = max(lb[i], x[i-1] + x_step)
        else:
            x[i] = lb[i]

    return x


def consume(q, para, iters):
    try:
        para_id = PARA_DICT[para]
        para_list = PARA_LIST[para_id]

        path = 'result/algo/{}'.format(para)
        if not os.path.exists(path):
            os.makedirs(path)

        t_path = '{}/time.csv'.format(path)
        e_path = '{}/error.csv'.format(path)

        time = [[[] for _ in range(len(algo_list))] for _ in range(len(para_list))]
        error = [[[] for _ in range(len(algo_list))] for _ in range(len(para_list))]

        count = 0
        total = len(para_list) * len(algo_list) * iters

        while True and count < total:
            i, j, t, err = q.get()
            time[i][j].append(t)
            error[i][j].append(err)
            count += 1
            print('\033[45m Consumed - i: {} j: {} total count: {}\033[0m'.format(i, j, count))

        np.savetxt(t_path, np.mean(time, axis=2))
        np.savetxt(e_path, np.mean(error, axis=2))
    except:
        current_filename = str(os.path.basename(sys.argv[0]))[:-3]
        cur_err_filname = current_filename + '_error.txt'
        error_info = sys.exc_info()
        with open(f'{cur_err_filname}', 'a') as f:
            error_str = f'ERROR OCCURRED,{time.strftime("%Y-%m-%d %H:%M:%S")}:\n {error_info[0]}: {error_info[1]}'
            print(error_str, file=f)
            traceback.print_tb(error_info[2], file=f)
            f.write(f"{'=' * 50}\n")


def produce(q, n, maxmu, cv, algo, a, b, lb_b, i, j, k):
    # wspt res
    res_wspt = wspt_res(a, b)

    lb_b = lb_b * T_MED / get_med(lb_b, a)

    sigma = get_mean(lb_b, a) * cv

    args = (a, b, sigma)
    ub_b = np.ones(n) * (maxmu + lb_b.max())
    bounds = Bounds(lb_b, ub_b)

    # constraints for interior method
    lb_c = np.ones(n) * -np.inf
    ub_c = np.zeros(n)
    A = get_A(n)
    linear_con = LinearConstraint(A, lb_c, ub_c, keep_feasible=False)

    try:
        if algo == 0:
            # step = maxmu / (n - 1 - np.argmax(lb_b))
            max_step = maxmu + lb_b.max() - lb_b.min()
            step = bisect_step(0, max_step, lb_b, a, b, sigma, res_wspt, maxmu)
            start = time.process_time()
            x = step_algo(lb_b, a, b, sigma, step=step)
            end = time.process_time()
            t = end - start
            rel_err = (fun(x, a, b, sigma) - res_wspt) / res_wspt
            # err_ub = upper_bound(x, a, b, sigma)
            max_dif = np.max(x) - np.max(lb_b)
            print('\033[46m n: {}, sigma: {:.4}, step: {:.4}\033[0m'.format(n, sigma, step))

        elif 0 < algo <= 3:
            # x0 for interior method
            delta = 3 * sigma
            x0 = get_x0(lb_b, delta)

            start = time.process_time()
            res = minimize(fun, x0, args=args, method='trust-constr', constraints=linear_con, bounds=bounds,
                           tol=ipm_tols[algo - 1])
            end = time.process_time()
            t = end - start
            rel_err = (res.fun - res_wspt) / res_wspt
            max_dif = np.max(res.x) - np.max(lb_b)

        else:
            assert (3 < algo <= 6)
            start = time.process_time()
            res = differential_evolution(fun, bounds, args=args, tol=de_tols[algo - 4])
            end = time.process_time()
            t = end - start
            rel_err = (res.fun - res_wspt) / res_wspt
            max_dif = np.max(res.x) - np.max(lb_b)

        q.put((i, j, t, rel_err))
        print('\033[44m Produced - i: {} algo: {} k: {} - time: {:.4} error: {:.4} max_dif: {:.4}\033[0m'.format(i, algo, k, t, rel_err, max_dif))
    except:
        current_filename = str(os.path.basename(sys.argv[0]))[:-3]
        cur_err_filname = current_filename + '_error.txt'
        error_info = sys.exc_info()
        with open(f'{cur_err_filname}', 'a') as f:
            error_str = f'{i}-{j}-{k}:ERROR OCCURRED,{time.strftime("%Y-%m-%d %H:%M:%S")}:\n {error_info[0]}: {error_info[1]}'  # 记录当前进程特征值，错误发生时间 ，错误类型，错误概述
            print(error_str, file=f)  # 通过打印方式写入文件
            traceback.print_tb(error_info[2], file=f)  # 错误细节描述（包括bug的代码位置）
            f.write(f"{'=' * 50}\n")  # 分行
#
def test_num(iters):
    '''
    parameter: n
    fix: max mu
    metrics: t & err
    '''

    maxmu = MAXMU_LIST[1]
    cv = CV_LIST[1]

    if not os.path.exists('result/algo'):
        os.makedirs('result/algo')

    queue = Manager().Queue()
    pool = Pool(cpu_count())

    for i, n in enumerate(NUM_LIST):
        for j, algo in enumerate(algo_list):
            for k in range(iters):
                a, b = get_ab(n)
                lb_b = np.random.rand(n)
                # lb_b = np.sort(lb_b)[::-1]

                pool.apply_async(produce, args=(queue, n, maxmu, cv, algo, a, b, lb_b, i, j, k,))
    pool.apply_async(consume, args=(queue, 'n', iters))
    pool.close()
    pool.join()

    return


def test_maxmu(iters):
    '''
    parameter: maxmu
    fix: n
    metrics: t & err
    '''

    n = NUM_LIST[1]
    cv = CV_LIST[1]

    if not os.path.exists('result/algo'):
        os.makedirs('result/algo')

    queue = Manager().Queue()
    pool = Pool(cpu_count())

    for i, maxmu in enumerate(MAXMU_LIST):
        for j, algo in enumerate(algo_list):
            for k in range(iters):
                a, b = get_ab(n)
                lb_b = np.random.rand(n)

                pool.apply_async(produce, args=(queue, n, maxmu, cv, algo, a, b, lb_b, i, j, k,))
    pool.apply_async(consume, args=(queue, 'mu', iters))
    pool.close()
    pool.join()

    return


def test_cv(iters):
    '''
    parameter: cv
    fix: n, max mu
    metrics: t & err
    '''

    n = NUM_LIST[1]
    maxmu = MAXMU_LIST[1]

    if not os.path.exists('result/algo'):
        os.makedirs('result/algo')

    queue = Manager().Queue()
    pool = Pool(cpu_count())

    for i, cv in enumerate(CV_LIST):
        for j, algo in enumerate(algo_list):
            for k in range(iters):
                a, b = get_ab(n)
                lb_b = np.random.rand(n)

                pool.apply_async(produce, args=(queue, n, maxmu, cv, algo, a, b, lb_b, i, j, k,))
    pool.apply_async(consume, args=(queue, 'cv', iters))
    pool.close()
    pool.join()

    return


'''
parameter: tol for interior
fix: n, max mu, cv
metrics: t & err
'''
def test_it(iters):
    maxmu = 50
    n = 16
    cv = 0.04

    dis = 25  # mean for delay is 12.6s from Decker
    sigma = dis * cv

    delta = 1

    tol_list = [1e-1, 1e-2, 1e-4, 1e-8]

    # algo_list = ['interior', 'step', 'diff_evo']
    algo_list = ['interior']
    algo_dict = {'interior': 0}
    headers_t = ['tol\\algo (t)'] + algo_list
    headers_e = ['tol\\algo (err)'] + algo_list
    table_t = []
    table_e = []

    for i, tol in enumerate(tol_list):
        table_t.append([tol])
        table_e.append([tol])
        iter_t = np.zeros(len(algo_list))
        iter_e = np.zeros(len(algo_list))

        for j in range(iters):
            # a, b
            a, b = get_ab(n)
            args = (a, b, sigma)

            # wspt res
            res_wspt = wspt_res(a, b)

            # lower&upper bounds
            lb_b = np.random.rand(n) * dis
            ub_b = np.ones(n) * maxmu
            bounds = Bounds(lb_b, ub_b)

            # constraints for interior method
            lb_c = np.ones(n) * -np.inf
            ub_c = np.zeros(n)
            A = get_A(n)
            linear_con = LinearConstraint(A, lb_c, ub_c, keep_feasible=False)

            # x0 for interior method
            x0 = get_x0(lb_b, delta)

            for algo in algo_list:
                if algo == 'interior':
                    start = time.process_time()
                    res = minimize(fun, x0, args=args, tol=tol, method='trust-constr', constraints=linear_con, bounds=bounds)
                    end = time.process_time()
                    t = end - start
                    err = res.fun - res_wspt

                elif algo == 'step':
                    start = time.process_time()
                    step = (maxmu - lb_b[0]) / (n - 1)
                    x = step_algo(lb_b, a, b, sigma, step=step)
                    end = time.process_time()
                    t = end - start
                    err = fun(x, a, b, sigma) - res_wspt

                elif algo == 'diff_evo':
                    start = time.process_time()
                    res = differential_evolution(fun, bounds, args=args)
                    end = time.process_time()
                    t = end - start
                    err = res.fun - res_wspt

                print('tol: {}, iter: {}, algo: {}, t: {}, err: {}'.format(tol, j, algo, t, err))
                iter_t[algo_dict[algo]] += t
                iter_e[algo_dict[algo]] += err
        for algo in algo_list:
            table_t[i].append(iter_t[algo_dict[algo]] / iters)
            table_e[i].append(iter_e[algo_dict[algo]] / iters)

    print(tabulate(table_t, headers=headers_t, tablefmt='github'))
    print(tabulate(table_e, headers=headers_e, tablefmt='github'))
    return


'''
parameter: tol for diff_evo
fix: n, max mu, cv
metrics: t & err
'''
def test_dt(iters):
    maxmu = 50
    n = 16
    cv = 0.04

    dis = 25  # mean for delay is 12.6s from Decker
    sigma = dis * cv

    delta = 1

    tol_list = [0.2, 0.1, 0.05, 0.01]

    headers = ['tol', 'time', 'error']
    table = []

    for i, tol in enumerate(tol_list):
        table.append([tol])
        iter_t = 0
        iter_e = 0

        for j in range(iters):
            # a, b
            a, b = get_ab(n)
            args = (a, b, sigma)

            # wspt res
            res_wspt = wspt_res(a, b)

            # lower&upper bounds
            lb_b = np.random.rand(n) * dis
            ub_b = np.ones(n) * maxmu
            bounds = Bounds(lb_b, ub_b)

            # constraints for interior method
            lb_c = np.ones(n) * -np.inf
            ub_c = np.zeros(n)
            A = get_A(n)
            linear_con = LinearConstraint(A, lb_c, ub_c, keep_feasible=False)

            # x0 for interior method
            x0 = get_x0(lb_b, delta)

            start = time.process_time()
            res = differential_evolution(fun, bounds, args=args, tol=tol)
            end = time.process_time()
            t = end - start
            err = res.fun - res_wspt

            print('tol: {}, iter: {}, algo: {}, t: {}, err: {}'.format(tol, j, 'diff_evo', t, err))
            iter_t += t
            iter_e+= err

        table[i].append(iter_t/ iters)
        table[i].append(iter_e/ iters)

    print(tabulate(table, headers=headers, tablefmt='github'))
    return


# TODO: 1,2,3 vs 2,2,2 vs 3,2,1

def decay_step_algo(lb, a, b, sigma, maxmu, decay=1):
    num = len(lb)
    x = np.zeros(num)

    # step
    a = [decay ** i for i in range(num)]
    step = maxmu / np.sum(a)

    for i in range(num):
        if i != 0:
            x[i] = max(lb[i], x[i-1] + step)
        else:
            x[i] = lb[i]
        step *= decay
    return x

'''
parameter: step skew
fix: n, max mu, cv
metrics: err
'''
def test_ss(iters):
    maxmu = 50
    n = 16
    cv = 0.04

    dis = 25  # mean for delay is 12.6s from Decker
    sigma = dis * cv

    delta = 1

    decay_list = [0.8, 0.9, 1, 1.1, 1.2]
    table = []

    for i, decay in enumerate(decay_list):
        table.append([])
        for j in range(iters):
            # a, b
            a, b = get_ab(n)
            args = (a, b, sigma)

            lb_b = np.random.rand(n) * dis

            # wspt res
            res_wspt = wspt_res(a, b)

            # step = (maxmu - lb_b[0]) / (n - 1)
            x = decay_step_algo(lb_b, a, b, sigma, maxmu, decay=decay)

            err = fun(x, a, b, sigma) - res_wspt
            table[i].append(err)

    res = np.mean(table, axis=1)
    print(res)
    return


def main():
    parser = argparse.ArgumentParser(description='my sim arg')
    
    parser.add_argument('--cumulative', '-c', action='store_true')
    parser.add_argument('--diff', '-d', action='store_true')
    parser.add_argument('--num', '-n', action='store_true')
    parser.add_argument('--maxmu', '-m', action='store_true')
    parser.add_argument('--cv', '-cv', action='store_true')
    parser.add_argument('--interiortol', '-it', action='store_true')
    parser.add_argument('--difftol', '-dt', action='store_true')
    parser.add_argument('--stepskew', '-ss', action='store_true')
    parser.add_argument('--iters', '-i', default=10, type=int)

    args = parser.parse_args()

    num = 5
    dis = 25
    cv = 0.02
    maxmu = 10
    sigma = dis * cv
    delta = 0.01
    epsilon = 0.01

    if args.cumulative:
        num_list = [8, 16]
        step_list = [sigma, sigma * 2, sigma * 3]   # follow 3-sigma rule
        headers = ['num\\step']
        headers += step_list
        table = []
        iters= 10
        for i, num in enumerate(num_list):
            table.append([num])
            for step in step_list:
                rel_err = 0
                gamma = 0
                for it in range(iters):
                    a, b = get_ab(num)
                    res_wspt = wspt_res(a, b)
                    lb = np.random.rand(num) * dis
                    x = step_algo(lb, a, b, sigma, step=step)
                    res = fun(x, a, b, sigma)
                    rel_err += (res - res_wspt)/res_wspt
                    gamma += res_wspt
                rel_err /= iters
                gamma /= iters
                table[i].append('{} * (1 + {})'.format(gamma, rel_err))
        print(tabulate(table, headers=headers, tablefmt='github'))
        return
    
    if args.diff:
        num = 16

        a, b = get_ab(num)

        lb_b = np.random.rand(num)
        lb_b = lb_b * T_MED / get_med(lb_b, a)

        sigma = get_mean(lb_b, a) * cv

        args = (a, b, sigma)
        ub_b = np.ones(num) * (maxmu + lb_b.max())
        bounds = Bounds(lb_b, ub_b)

        start = time.process_time()
        res = differential_evolution(fun, bounds, args=args)
        end = time.process_time()

        res_wspt = wspt_res(a, b)

        print(f'relative error: {(res.fun-res_wspt)/res_wspt:.4}, executing time: {end - start}')
        print(res.message)

        return

    '''
    parameter: n
    fix: max mu
    metrics: t & err
    '''
    if args.num:
        test_num(args.iters)

    '''
    parameter: max mu
    fix: n
    metrics: t & err
    '''
    if args.maxmu:
        test_maxmu(args.iters)

    '''
    parameter: max mu
    fix: n
    metrics: t & err
    '''
    if args.cv:
        test_cv(args.iters)

    '''
    test tol on interior method
    parameters: tol 
    metric: t & err
    '''
    if args.interiortol:
        test_it(args.iters)

    '''
    test tol on diff_evol
    parameters: tol 
    metric: t & err
    '''
    if args.difftol:
        test_dt(args.iters)

    '''
    test step skew
    parameters: skew 
    metric: err
    '''
    if args.stepskew:
        test_ss(args.iters)


if __name__ == '__main__':
    main()
