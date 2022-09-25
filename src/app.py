import argparse
import sys
import os
import csv
import numpy as np
# from tabulate import tabulate
from multiprocessing import Manager, Pool, cpu_count
import time
import traceback

import network
import info


TEST_CASE_NUM = 10
TEST_CASE_REPEAT = 5

TEST_ALPHA = np.round(np.arange(1,10) * 0.05, decimals=2)
TEST_GAMMA = np.round(np.arange(1, 15) * 0.05, decimals=2)

TEST_CHURN_H = np.array([10, 20, 40, 80, 160])
TEST_CHURN_SM = np.array([10, 20, 40, 80, 160])

TEST_DYN_POW_CYCLE = np.array([10, 20, 40, 80, 160])
TEST_DYN_POW_RANGE = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

TEST_TFT = [info.Tft.ALL.value, info.Tft.NONE.value]

TEST_ID = [False, True]
TEST_K_DIS = [10, 20, 50]
TEST_SM = [False, True]

TEST_NUM_NODES = [16]

parser = argparse.ArgumentParser(description='my sim arg')

parser.add_argument('--tft', '-tft', help='apply tit-for-tat strategy', default=info.Tft.NONE.value, type=int)
parser.add_argument('--sm', '-sm', help='sm mode', action='store_true')
parser.add_argument('--writegraph', '-wg', help='generate new graph and write in disk', action='store_true')
parser.add_argument('--writepower', '-wp', help='generate new power distribution and write in disk', action='store_true')
parser.add_argument('--identical', '-i', help='nodes have identical computing power', action='store_true')
parser.add_argument('--nodes', '-n', help='total number of nodes', default=16, type=int)
parser.add_argument('--rounds', '-r', help='total rounds', default=10000, type=int)
parser.add_argument('--batch', '-b', help='batch test', default=info.Test.NONE.value, type=int)
parser.add_argument('--alpha', '-a', help='alpha for sm', default=0, type=float)
parser.add_argument('--gamma', '-g', help='gamma for sm', default=0, type=float)
parser.add_argument('--mediantime', '-mt', help='median time', default=8.7, type=float)
parser.add_argument('--cv', '-cv', help='cv for sigma', default=0.02, type=float)
parser.add_argument('--step', '-s', help='step scale', default=3, type=float)
parser.add_argument('--write', '-w', help='write result', action='store_true')
parser.add_argument('--case', '-c', help='pre generated graph/power', default=-1, type=int)

parser.add_argument('--session', '-ss', help='session length in peer dynamic (churn)', default=np.infty, type=int)
parser.add_argument('--smsession', '-smss', help='session length in sm churn', default=np.infty, type=int)
parser.add_argument('--length', '-l', help='window length', default=100)
parser.add_argument('--mature', '-m', help='mature length', default=50)
parser.add_argument('--powrange', '-pr', help='power range', default=0, type=float)
parser.add_argument('--powcycle', '-pc', help='power cycle', default=np.infty, type=int)
parser.add_argument('--oracle', '-o', help='static oracle', action='store_true')

args = parser.parse_args()


def relative_revenue(a, g):
    return (a*(1-a)**2*(4*a+g*(1-2*a))-a**3)/(1-a*(1+a*(2-a)))


def create_result(test_case):
    full_path = 'result'
    if not os.path.exists(full_path):
        os.mkdir(full_path)
    file_path = '{}/{}.csv'.format(full_path, info.test_name_list[test_case])
    with open(file_path, 'w') as f:
        csv_write = csv.writer(f)
        if test_case == info.Test.SM_AG.value:
            header = ['num', 'tft_mode', 'kld', 'fork rate', 'interval', 'effective_hashrate', 'gamma', 'sm power', 'sm revenue']
        elif test_case == info.Test.TFT_AG.value:
            header = ['num', 'tft_mode', 'kld', 'fork rate', 'interval', 'effective_hashrate', 'gamma', 'sm power', 'sm revenue', 'median tft']
        else:
            header = ['num','kld', 'fork rate', 'interval', 'effective_hashrate']
        csv_write.writerow(header)


def write_row(test_case, result):
    file_path = 'result/{}.csv'.format(info.test_name_list[test_case])
    with open(file_path, 'a') as f:
        csv_write = csv.writer(f)
        if type(result) == info.SmInfo:
            row = [result.num, False, result.kld, result.fork_rate, result.interval, result.effective_hashrate, result.gamma, result.sm_power, result.sm_revenue]
        elif type(result) == info.TftInfo:
            row = [result.num, True, result.kld, result.fork_rate, result.interval, result.effective_hashrate, result.gamma, result.sm_power, result.sm_revenue, result.median_tft]
        else:
            row = [result.num, result.kld, result.fork_rate, result.interval, result.effective_hashrate]
        csv_write.writerow(row)


def cases_result(results, write=False):
    nums = []
    klds = []
    forks = []
    intervals = []
    effective_hashrates = []
    gammas = []
    sm_ps = []
    sm_rs = []
    tft_mts = []

    for result in results:
        nums.append(result.num)
        klds.append(result.kld)
        forks.append(result.fork_rate)
        intervals.append(result.interval)
        effective_hashrates.append(result.effective_hashrate)
        if type(result) == info.SmInfo:
            gammas.append(result.gamma)
            sm_ps.append(result.sm_power)
            sm_rs.append(result.sm_revenue)
        elif type(results[0]) == info.TftInfo:
            gammas.append(result.gamma)
            sm_ps.append(result.sm_power)
            sm_rs.append(result.sm_revenue)
            tft_mts.append(result.median_tft)

    num = np.mean(nums)
    kld = np.mean(klds)
    fork = np.mean(forks)
    interval = np.mean(intervals)
    effective_hashrate = np.mean(effective_hashrates)
    if type(results[0]) == info.SmInfo:
        gamma = np.mean(gammas)
        sm_p = np.mean(sm_ps)
        sm_r = np.mean(sm_rs)
        print('num: {}, tft: {}, kld: {}, fork rate: {}, interval: {}, effective_hashrate: {}, gamma: {}, sm power: {}, sm revenue: {}'
              .format(num, False, kld, fork, interval, effective_hashrate, gamma, sm_p, sm_r))
        if write:
            write_row(args.batch, info.SmInfo(num, kld, fork, interval, effective_hashrate, gamma, sm_p, sm_r))
    elif type(results[0]) == info.TftInfo:
        gamma = np.mean(gammas)
        sm_p = np.mean(sm_ps)
        sm_r = np.mean(sm_rs)
        tft_mt = np.mean(tft_mts)
        print('num: {}, tft: {}, kld: {}, fork rate: {}, interval: {}, effective_hashrate: {}, gamma: {}, sm power: {}, sm revenue: {}, median tft: {}'
              .format(num, True, kld, fork, interval, effective_hashrate, gamma, sm_p, sm_r,tft_mt))
        if write:
            write_row(args.batch, info.TftInfo(num, kld, fork, interval, effective_hashrate, gamma, sm_p, sm_r, tft_mt))
    else:
        print('num: {}, honest, kld: {}, fork rate: {}, interval: {}, effective_hashrate: {}'
              .format(num, kld, fork, interval, effective_hashrate))
        if write:
            write_row(args.batch, info.Info(num, kld, fork, interval, effective_hashrate))


def produce(q, args, a_i, g_i, case, tft=info.Tft.NONE.value):
    args.alpha = TEST_ALPHA[a_i]
    args.gamma = TEST_GAMMA[g_i]
    args.case = case

    cluster = network.Network(args)
    res = cluster.run(TEST_CASE_REPEAT)
    q.put((a_i, g_i, res))

    del cluster
    print('\033[44m Produced - alpha: {} gamma: {} case: {}\033[0m'.format(args.alpha, args.gamma, case))


def consume(q):
    results = [[[] for _ in range(len(TEST_GAMMA))] for _ in range(len(TEST_ALPHA))]
    count = 0
    total = len(TEST_ALPHA) * len(TEST_GAMMA) * TEST_CASE_NUM

    while True and count < total:
        a_i, g_i, res = q.get()
        results[a_i][g_i].append(res)
        count += 1
        print('\033[45m Consumed - alpha: {} gamma: {} total count: {}\033[0m'
              .format(TEST_ALPHA[a_i], TEST_GAMMA[g_i], count))

    for i, alpha in enumerate(TEST_ALPHA):
        for j, gamma in enumerate(TEST_GAMMA):
            print('\033[42m Writing - alpha: {} gamma: {}\033[0m'.format(alpha, gamma))
            cases_result(results[i][j], True)


def test_h_ag(args):   # 5
    args.write = True
    create_result(args.batch)

    queue = Manager().Queue()
    pool = Pool(cpu_count())

    # ... 12 producer better, 1 consumer is always starving, downgrade from 12 core to 11 core lol.
    for i in range(len(TEST_ALPHA)):
        for j in range(len(TEST_GAMMA)):
            for case in range(TEST_CASE_NUM):   # specify case no.
                pool.apply_async(produce, args=(queue, args, i, j, case))
    pool.apply_async(consume, args=(queue,))

    pool.close()
    pool.join()


def test_sm_ag(args):   # 6
    args.sm = True
    args.write = True
    create_result(args.batch)

    queue = Manager().Queue()
    pool = Pool(cpu_count())

    # ... 12 producer better, 1 consumer is always starving, downgrade from 12 core to 11 core lol.
    for i in range(len(TEST_ALPHA)):
        for j in range(len(TEST_GAMMA)):
            for case in range(TEST_CASE_NUM):   # specify case no.
                pool.apply_async(produce, args=(queue, args, i, j, case))
    pool.apply_async(consume, args=(queue,))

    pool.close()
    pool.join()


def test_tft_ag(args):  # 7
    args.sm = True
    args.tft = info.Tft.ALL.value
    args.write = True
    create_result(args.batch)

    queue = Manager().Queue()
    pool = Pool(cpu_count())

    for i in range(len(TEST_ALPHA)):
        for j in range(len(TEST_GAMMA)):
            for case in range(TEST_CASE_NUM):  # specify case no.
                pool.apply_async(produce, args=(queue, args, i, j, case))
    pool.apply_async(consume, args=(queue,))

    pool.close()
    pool.join()


# Find threshold =====================


def theo_thre(gamma):
    return (1 - gamma) / (3 - 2 * gamma)


def find_alpha(q, args, tft, t_i, g_j, case, low_alpha, up_alpha, iter=0, tol=0.01, maxit=10):
    args.tft = tft
    args.alpha = TEST_ALPHA[round(up_alpha / 0.05) - 1]
    args.gamma = TEST_GAMMA[g_j]
    args.case = case

    t_alpha = (low_alpha + up_alpha) / 2

    try:
        cluster = network.Network(args, t_alpha)
        res = cluster.run(TEST_CASE_REPEAT)
        del cluster
        sm_rev = res.sm_revenue

        if abs(sm_rev - t_alpha) / t_alpha < tol or iter >= maxit:
            q.put((t_i, g_j, t_alpha))
            print(f'\033[44m Produced - tft: {t_i} gamma: {g_j} case: {case} t_alpha: {t_alpha:.4}'
                  f' - err: {abs(sm_rev - t_alpha)}, iter: {iter}\033[0m')
        elif sm_rev > t_alpha:
            find_alpha(q, args, tft, t_i, g_j, case, low_alpha, t_alpha, iter + 1)
        else:
            find_alpha(q, args, tft, t_i, g_j, case, t_alpha, up_alpha, iter + 1)

    except:
        current_filename = str(os.path.basename(sys.argv[0]))[:-3]
        cur_err_filname = current_filename + '_error.txt'
        error_info = sys.exc_info()
        with open(f'{cur_err_filname}', 'a') as f:
            error_str = f'ERROR OCCURRED,{time.strftime("%Y-%m-%d %H:%M:%S")}:\n {error_info[0]}: {error_info[1]}'
            print(error_str, file=f)
            traceback.print_tb(error_info[2], file=f)
            f.write(f"{'=' * 50}\n")


def record_alpha(q):
    path = 'result/threshold.csv'
    results = [[[] for _ in range(len(TEST_GAMMA))] for _ in range(len(TEST_TFT))]
    count = 0
    total = len(TEST_TFT) * len(TEST_GAMMA) * TEST_CASE_NUM

    while True and count < total:
        t_i, g_j, t_alpha = q.get()
        results[t_i][g_j].append(t_alpha)
        count += 1
        print(f'\033[45m Consumed - tft: {t_i} gamma: {g_j} total count: {count}\033[0m')

    results = np.mean(results, axis=2)
    np.savetxt(path, results)


def test_threshold(args):
    args.sm = True

    queue = Manager().Queue()
    pool = Pool(cpu_count())

    for i in range(len(TEST_TFT)):
        tft = TEST_TFT[i]
        for j in range(len(TEST_GAMMA)):
            if tft == info.Tft.NONE.value:
                low_alpha = theo_thre(TEST_GAMMA[j]) // 0.05 * 0.05
                up_alpha = low_alpha + 0.05
            else:
                low_alpha = 0.3
                up_alpha = 0.35
            for case in range(TEST_CASE_NUM):  # specify case no.
                pool.apply_async(find_alpha, args=(queue, args, tft, i, j, case, low_alpha, up_alpha,))
    pool.apply_async(record_alpha, args=(queue,))

    pool.close()
    pool.join()


# Test unfairness ======================


TEST_FAIR_CASE = [0, 1, 2]
TEST_FAIR_G = [0.7, 0.5, 0.7]
TEST_FAIR_T = [info.Tft.NONE.value, info.Tft.NONE.value, info.Tft.ALL.value]
TEST_FAIR_ALPHA = np.linspace(1/3, 0.45, 10)


def fair_produce(q, args, f_i, a_j, case):
    args.tft = TEST_FAIR_T[f_i]
    args.gamma = TEST_FAIR_G[f_i]
    t_alpha = TEST_FAIR_ALPHA[a_j]
    args.alpha = TEST_ALPHA[round(t_alpha / 0.05) - 1]
    args.case = case

    try:
        cluster = network.Network(args, t_alpha)
        res = cluster.run(TEST_CASE_REPEAT)
        del cluster

        kld = res.kld
        q.put((f_i, a_j, kld))
        print(f'\033[44m Produced - fair: {f_i} alpha: {a_j} case: {case} kld: {kld:.4}\033[0m')
    except:
        current_filename = str(os.path.basename(sys.argv[0]))[:-3]
        cur_err_filname = current_filename + '_error.txt'
        error_info = sys.exc_info()
        with open(f'{cur_err_filname}', 'a') as f:
            error_str = f'ERROR OCCURRED,{time.strftime("%Y-%m-%d %H:%M:%S")}:\n {error_info[0]}: {error_info[1]}'
            print(error_str, file=f)
            traceback.print_tb(error_info[2], file=f)
            f.write(f"{'=' * 50}\n")


def fair_consume(q):
    path = 'result/fair.csv'
    results = [[[] for _ in range(len(TEST_FAIR_ALPHA))] for _ in range(len(TEST_FAIR_CASE))]
    count = 0
    total = len(TEST_FAIR_CASE) * len(TEST_FAIR_ALPHA) * TEST_CASE_NUM

    while True and count < total:
        f_i, a_j, kld = q.get()
        results[f_i][a_j].append(kld)
        count += 1
        print(f'\033[45m Consumed - fair: {f_i} alpha: {a_j} total count: {count}\033[0m')

    results = np.mean(results, axis=2)
    np.savetxt(path, results)


def test_fairness(args):
    args.sm = True

    queue = Manager().Queue()
    pool = Pool(cpu_count())

    for i in range(len(TEST_FAIR_CASE)):
        for j, a in enumerate(TEST_FAIR_ALPHA):
            for case in range(TEST_CASE_NUM):  # specify case no.
                pool.apply_async(fair_produce, args=(queue, args, i, j, case,))
    pool.apply_async(fair_consume, args=(queue,))

    pool.close()
    pool.join()


# Test churn ======================

def churn_produce(q, args, i, j, case):
    args.session = TEST_CHURN_H[i]
    args.smsession = TEST_CHURN_SM[j]
    args.case = case

    try:
        cluster = network.Network(args)
        res = cluster.run(TEST_CASE_REPEAT)
        sm_rev = res.sm_revenue
        eff_hash = res.effective_hashrate
        q.put((i, j, sm_rev, eff_hash))

        del cluster
        print(f'\033[44m Produced - h session: {args.session} sm session: {args.smsession} case: {case}\033[0m')
    except:
        current_filename = str(os.path.basename(sys.argv[0]))[:-3]
        cur_err_filname = current_filename + '_error.txt'
        error_info = sys.exc_info()
        with open(f'{cur_err_filname}', 'a') as f:
            error_str = f'ERROR OCCURRED,{time.strftime("%Y-%m-%d %H:%M:%S")}:\n {error_info[0]}: {error_info[1]}'
            print(error_str, file=f)
            traceback.print_tb(error_info[2], file=f)
            f.write(f"{'=' * 50}\n")

def churn_consume(q):
    path = 'result/churn.csv'
    rev_res = [[[] for _ in range(len(TEST_CHURN_SM))] for _ in range(len(TEST_CHURN_H))]
    hash_res = [[[] for _ in range(len(TEST_CHURN_SM))] for _ in range(len(TEST_CHURN_H))]
    count = 0
    total = len(TEST_CHURN_H) * len(TEST_CHURN_SM) * TEST_CASE_NUM

    try:
        while True and count < total:
            i, j, sm_rev, eff_hash = q.get()
            rev_res[i][j].append(sm_rev)
            hash_res[i][j].append(eff_hash)
            count += 1
            print(f'\033[45m Consumed - h session: {TEST_CHURN_H[i]} sm session: {TEST_CHURN_SM[j]} total count: {count} total: {total}\033[0m')

        print(f'\033[45m Consume - queue cleaned.\033[0m')

        rev_mean = np.mean(rev_res, axis=2)
        hash_mean = np.mean(hash_res, axis=2)

        with open(path, 'w') as f:
            csv_write = csv.writer(f)
            header = ['h_session', 'sm_session', 'sm_rev', 'eff_hash']
            csv_write.writerow(header)
            for i, h_ss in enumerate(TEST_CHURN_H):
                for j, sm_ss in enumerate(TEST_CHURN_SM):
                    csv_write.writerow([h_ss, sm_ss, rev_mean[i][j], hash_mean[i][j]])
    except:
        current_filename = str(os.path.basename(sys.argv[0]))[:-3]
        cur_err_filname = current_filename + '_error.txt'
        error_info = sys.exc_info()
        with open(f'{cur_err_filname}', 'a') as f:
            error_str = f'ERROR OCCURRED,{time.strftime("%Y-%m-%d %H:%M:%S")}:\n {error_info[0]}: {error_info[1]}'
            print(error_str, file=f)
            traceback.print_tb(error_info[2], file=f)
            f.write(f"{'=' * 50}\n")


def test_churn(args):   # 12
    args.sm = True
    args.tft = 2
    args.alpha = 0.3
    args.gamma = 0.7

    queue = Manager().Queue()
    pool = Pool(cpu_count())
    print(cpu_count())

    for i in range(len(TEST_CHURN_H)):
        for j in range(len(TEST_CHURN_SM)):
            for case in range(TEST_CASE_NUM):  # specify case no.
                pool.apply_async(churn_produce, args=(queue, args, i, j, case,))
    pool.apply_async(churn_consume, args=(queue,))

    pool.close()
    pool.join()


# Test Dynamic Power ======================

def dyn_pow_produce(q, args, ci, ri, case):
    args.powcycle = TEST_DYN_POW_CYCLE[ci]
    args.powrange = TEST_DYN_POW_RANGE[ri]
    args.case = case

    try:
        cluster = network.Network(args)
        res = cluster.run(TEST_CASE_REPEAT)
        sm_rev = res.sm_revenue
        eff_hash = res.effective_hashrate
        q.put((ci, ri, sm_rev, eff_hash))

        del cluster
        print(f'\033[44m Produced - pow circle: {args.powcycle} pow range: {args.powrange} case: {case}\033[0m')
    except:
        current_filename = str(os.path.basename(sys.argv[0]))[:-3]
        cur_err_filname = current_filename + '_error.txt'
        error_info = sys.exc_info()
        with open(f'{cur_err_filname}', 'a') as f:
            error_str = f'ERROR OCCURRED,{time.strftime("%Y-%m-%d %H:%M:%S")}:\n {error_info[0]}: {error_info[1]}'
            print(error_str, file=f)
            traceback.print_tb(error_info[2], file=f)
            f.write(f"{'=' * 50}\n")

def dyn_pow_consume(q):
    path = 'result/test/dyn_pow.csv'
    assert(len(TEST_DYN_POW_RANGE) == len(TEST_DYN_POW_CYCLE))
    len_x = len(TEST_DYN_POW_CYCLE)
    rev_res = [[[] for _ in range(len_x)] for _ in range(2)]
    hash_res = [[[] for _ in range(len_x)] for _ in range(2)]
    count = 0
    total = (2 * len_x- 1) * TEST_CASE_NUM

    try:
        while True and count < total:
            ci, ri, sm_rev, eff_hash = q.get()
            if ci == 1:
                rev_res[0][ri].append(sm_rev)
                hash_res[0][ri].append(eff_hash)
            if ri == 1:
                rev_res[1][ci].append(sm_rev)
                hash_res[1][ci].append(eff_hash)
            count += 1
            print(f'\033[45m Consumed - pow circle: {TEST_DYN_POW_CYCLE[ci]} pow range: {TEST_DYN_POW_RANGE[ri]} total count: {count} total: {total}\033[0m')

        print(f'\033[45m Consume - queue cleaned.\033[0m')

        rev_mean = np.mean(rev_res, axis=2)
        hash_mean = np.mean(hash_res, axis=2)

        with open(path, 'w') as f:
            csv_write = csv.writer(f)
            header = ['pow_circle', 'pow_range', 'sm_rev', 'eff_hash']
            csv_write.writerow(header)
            # for ci = 1
            for ri, pr in enumerate(TEST_DYN_POW_RANGE):
                pc = TEST_DYN_POW_CYCLE[1]
                csv_write.writerow([pc, pr, rev_mean[0][ri], hash_mean[0][ri]])
            # for ri = 1
            for ci, pc in enumerate(TEST_DYN_POW_CYCLE):
                pr = TEST_DYN_POW_RANGE[1]
                csv_write.writerow([pc, pr, rev_mean[1][ci], hash_mean[1][ci]])
    except:
        current_filename = str(os.path.basename(sys.argv[0]))[:-3]
        cur_err_filname = current_filename + '_error.txt'
        error_info = sys.exc_info()
        with open(f'{cur_err_filname}', 'a') as f:
            error_str = f'ERROR OCCURRED,{time.strftime("%Y-%m-%d %H:%M:%S")}:\n {error_info[0]}: {error_info[1]}'
            print(error_str, file=f)
            traceback.print_tb(error_info[2], file=f)
            f.write(f"{'=' * 50}\n")

def test_dyn_pow(args):   # 13
    args.sm = True
    args.tft = 2
    args.alpha = 0.3
    args.gamma = 0.7
    args.discycle = 20
    args.disrange = 0.2

    queue = Manager().Queue()
    pool = Pool(cpu_count())
    print(cpu_count())

    ci = 1   # cycle = TEST_DYN_POW_CYCLE[1] 20
    for ri in range(len(TEST_DYN_POW_RANGE)):
        for case in range(TEST_CASE_NUM):  # specify case no.
            pool.apply_async(dyn_pow_produce, args=(queue, args, ci, ri, case,))
    ri = 1   # range = TEST_DYN_POW_RANGE[1] 20%
    for ci in range(len(TEST_DYN_POW_CYCLE)):
        if ci == 1:
            continue  # only compute (ci=1, ri=1) once
        for case in range(TEST_CASE_NUM):  # specify case no.
            pool.apply_async(dyn_pow_produce, args=(queue, args, ci, ri, case,))
    pool.apply_async(dyn_pow_consume, args=(queue,))

    pool.close()
    pool.join()


def main():
    start = time.time()

    if args.batch == info.Test.NONE.value:
        cluster = network.Network(args)
        result = cluster.run(1)
        print('=== num: {}'.format(args.nodes))
        cases_result([result])

    elif args.batch == info.Test.BATCH.value:    # batch test
        results = []
        for i, num_node in enumerate(TEST_NUM_NODES):  # num_node
            args.nodes = num_node
            print('=== num: {}'.format(args.nodes))
            for j in range(TEST_CASE_NUM):
                # init new graph and power
                args.writegraph = True
                args.writepower = True
                cluster = network.Network(args)
                result = cluster.run(TEST_CASE_REPEAT)
                results.append(result)
                del cluster
            cases_result(results)
            results.clear()

    elif args.batch == info.Test.POWER.value:    # identical tes
        for i, num_node in enumerate(TEST_NUM_NODES):  # num_node
            results = [[] for _ in range(len(TEST_ID))]
            args.nodes = num_node
            print('>>>>>> num: {}'.format(args.nodes))
            for j in range(TEST_CASE_NUM):
                # init new graph and power
                args.writegraph = True
                args.writepower = True
                for k, power_id in enumerate(TEST_ID):
                    args.identical = power_id
                    cluster = network.Network(args)
                    result = cluster.run(TEST_CASE_REPEAT)
                    results[k].append(result)
                    del cluster
                    args.writegraph = False
                    args.writepower = False
            for j, power_id in enumerate(TEST_ID):
                print('=== power identical: {}'.format(power_id))
                cases_result(results[j])

    elif args.batch == info.Test.SM.value:    # sm test
        for i, num_node in enumerate(TEST_NUM_NODES):  # num_node
            results = [[] for _ in range(len(TEST_SM))]
            args.nodes = num_node
            print('>>>>>> num: {}'.format(args.nodes))
            for j in range(TEST_CASE_NUM):
                # init new graph and power
                args.writegraph = True
                args.writepower = True
                for k, sm in enumerate(TEST_SM):
                    args.sm = sm
                    cluster = network.Network(args)
                    result = cluster.run(TEST_CASE_REPEAT)
                    results[k].append(result)
                    del cluster
                    args.writegraph = False
                    args.writepower = False
            for j, sm in enumerate(TEST_SM):
                print('=== sm: {}'.format(sm))
                cases_result(results[j])

    elif args.batch == info.Test.H_AG.value:    # alpha and gamma - Honest 5
            test_h_ag(args)

    elif args.batch == info.Test.SM_AG.value:    # alpha and gamma - SM 6
            test_sm_ag(args)

    elif args.batch == info.Test.TFT_AG.value:    # alpha and gamma - TFT 7
        test_tft_ag(args)

    elif args.batch == info.Test.MT_CV.value:    # median time & cv test 8
        args.sm = True
        args.write = True
        args.nodes = 16  # nodes num doesn't matter, validated in alpha and gamma test
        args.alpha = 0.45
        args.gamma = 0.5

        TEST_MT = [5, 8.7, 15]
        TEST_CV = [0.001, 0.01, 0.1]

        rev_res = [[[[] for _ in range(len(TEST_TFT))] for _ in range(len(TEST_CV))] for _ in range(len(TEST_MT))]
        for i in range(TEST_CASE_NUM):
            for j, mt in enumerate(TEST_MT):
                args.mediantime = mt
                for k, cv in enumerate(TEST_CV):
                    args.cv = cv
                    # init new graph and power
                    args.writegraph = True
                    args.writepower = True
                    for l, tft_mode in enumerate(TEST_TFT):  # tft
                        args.tft = tft_mode
                        cluster = network.Network(args)
                        result = cluster.run(TEST_CASE_REPEAT)
                        rev_res[j][k][l].append(result.sm_revenue)
                        del cluster
                        args.writegraph = False
                        args.writepower = False
        rev_mean = np.mean(rev_res, axis=3)
        theo_rev = [[[relative_revenue(args.alpha, args.gamma)] for _ in TEST_CV] for _ in TEST_MT]
        rev = np.append(rev_mean, theo_rev, axis=2)
        rev = np.around(rev, decimals=4)

        table = [[mt] for mt in TEST_MT]
        for i in range(len(TEST_MT)):
            for j in range(len(TEST_CV)):
                table[i].append(tuple(rev[i][j]))
        header = ['mt\\cv'] + TEST_CV
        # print(tabulate(table, headers=header, tablefmt='github'))

    elif args.batch == info.Test.STEP.value:    # STEP test 9
        args.sm = True
        args.write = True
        args.nodes = 16  # nodes num doesn't matter, validated in alpha and gamma test
        args.alpha = 0.45
        args.gamma = 0.5

        TEST_STEP = [1, 3, 10, 100]

        rev_res = [[] for _ in range(len(TEST_STEP) + 1)]
        for i in range(TEST_CASE_NUM):
            # init new graph and power
            args.writegraph = True
            args.writepower = True
            args.tft = info.Tft.NONE.value
            cluster = network.Network(args)
            result = cluster.run(TEST_CASE_REPEAT)
            del cluster
            # result for sm no tft
            rev_res[len(TEST_STEP)].append(result.sm_revenue)
            args.tft = info.Tft.ALL.value
            args.writegraph = False
            args.writepower = False
            for j, step in enumerate(TEST_STEP):
                args.step = step
                cluster = network.Network(args)
                result = cluster.run(TEST_CASE_REPEAT)
                rev_res[j].append(result.sm_revenue)
                del cluster
        rev_mean = np.mean(rev_res, axis=1)
        rev = np.around(rev_mean, decimals=4)

        table = [rev]
        header = TEST_STEP + ['no tft']
        # print(tabulate(table, headers=header, tablefmt='github'))

    elif args.batch == info.Test.THRE.value:    # TFT test 10
        test_threshold(args)

    elif args.batch == info.Test.FAIR.value:    # TFT test 11
        test_fairness(args)

    elif args.batch == info.Test.CHURN.value:    # 12
            test_churn(args)

    elif args.batch == info.Test.DYN_POW.value:    # 13
            test_dyn_pow(args)

    else:
        sys.exit('bye bye')

    end = time.time()
    print(f'time: {end - start}')


if __name__ =='__main__':
    main()