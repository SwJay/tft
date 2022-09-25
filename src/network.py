import math
import heapq as hq
import queue
import numpy as np
import sys
import time
from scipy.stats import norm
# from tabulate import tabulate
import copy
from decimal import *

import gen_network
import blockchain
import info
import node
import em
import solver


# TFT_THRESHOLD = 200 # threshold to adapt em algorithm
TENPLACE = Decimal(10) ** 10

def kl_divergence(p, r):
    kld = 0
    for i, pi in enumerate(p):
        kld += pi * math.log(pi / r[i])
    return kld


def count_rule(me, other, competitors):
    if me in competitors:
        if me == other:
            return 0
        else:
            res = Decimal(len(competitors) / (len(competitors) -  1))
            return float(Decimal(res).quantize(TENPLACE, rounding=ROUND_HALF_UP))   # For better accuracy, e.g. 4/3 == 1/3 + 1
    else:
        return 1


def exp_mean_dis(graph, power):
    n = len(graph)
    alpha = power[n-1]
    res = 0
    for i in range(n-1):
        for j in range(n):
            res += power[i] * graph[i][j]
    res = res / ((n-1) * (1-alpha))
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


def true_med_dis(graph, revenue):
    # remove diagonal since they are all 0's
    n = len(graph)
    pos_dis = graph[~np.eye(n, dtype=bool)].reshape(n, -1)[:-1,]

    counts = revenue[:-1]

    assert(len(pos_dis) == len(counts))

    res = pos_dis.repeat(counts, axis=0)
    med = np.median(res)
    return med


def gini_coef(wealths):
    cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
    sum_wealths = cum_wealths[-1]
    xarray = np.array(range(0, len(cum_wealths))) / np.float(len(cum_wealths) - 1)
    yarray = cum_wealths / sum_wealths
    B = np.trapz(yarray, x=xarray)
    A = 0.5 - B

    return A / (A + B)


class Network:

    def __init__(self, args, t_alpha=0):
        self.tft_mode = args.tft            # if apply tit-for-tat strategy
        self.is_sm = args.sm                # if have sm attacker
        is_wg = args.writegraph             # if generate new graph
        is_wp = args.writepower             # if generate new graph
        is_i = args.identical               # if node power are identical
        alpha = args.alpha                  # computing power fo sm
        gamma = args.gamma                  # network connectivity for sm

        mediantime = args.mediantime
        self.cv = args.cv

        # rounds
        self.rounds = args.rounds           # total rounds
        self.height = 1
        self.tft_work = False

        # parameter
        self.num_node = args.nodes          # total number of nodes

        # set network graph and power distribution
        # 1. alpha = 0: randomly generate
        # 2. sm node id = num_node - 1
        self.org_power = gen_network.get_power(self.num_node, is_wp, is_i, alpha, gamma, args.case, t_alpha)
        self.power = self.org_power
        # strong threat model where sm can immediately inform remaining nodes
        self.graph = gen_network.get_graph(self.num_node, is_wg, self.power, alpha, gamma, self.cv, mediantime, args.case)

        self.dis_mean = exp_mean_dis(self.graph, self.power)
        self.step_scale = args.step

        # set up nodes
        self.sm_id = -1
        self.single_tft_id = 0
        self.nodes = []
        # distance from sm are updated to 0 in this function
        self._init_nodes()

        # set up tft vector, none for sm
        self.tft_trigger = 1
        self.em_algo = em.Em(self.power)
        # row n-1 is idle
        self.tft_count_mat = np.zeros((self.num_node, self.num_node))
        self.tft_delay_mat = np.zeros((self.num_node, self.num_node))

        # single result
        self.single_gammas = []

        # record blockchain and fork
        self.blockchain = blockchain.BlockChain(self.num_node)

        # total result
        self.total_kl_diversity = []
        self.total_fork_rate = []
        self.total_gamma = []
        self.total_sm_revenue = []
        self.total_median_tft = []
        self.total_interval = []
        self.single_effect_hashrate = []
        self.total_effect_hashrate = []

        # check point for window result
        self.checkpoint = 0

        # state count for sm state machine
        # count all the way to end state, except start state, e.g. s0 -> s1 -> s0', we count s1, s0'
        # | 0   | 1   | 2   | 3    | 4   |
        # |-----|-----|-----|------|-----|
        # | s0  | s1  | s2  | s3-n | s0' |

        self.state_count = np.zeros(5)

        # churn
        self.window = queue.Queue()                             # Sliding window recoding blocks
        self.session = args.session                             # Honest churn cycle
        self.sm_session = args.smsession                        # SM disguise cycle
        self.win_len = args.length                              # Window length: L_W
        self.life_span = np.zeros(self.num_node)                # Lifespan of nodes
        self.mature_time = args.mature                          # Cold start period length: L_M
        self.mature_nodes = np.full(self.num_node, False)       # Indicator of mature

        self.alpha = args.alpha
        self.gamma = args.gamma
        self.case = args.case

        self.pow_range = args.powrange                          # Range of power dynamic
        self.pow_cycle = args.powcycle                          # Cycle of power dynamic
        self.is_oracle = args.oracle                            # Oracle: know the exact power rather than estimation in window
        self.pow_count = np.zeros(self.num_node)                # Estiamted power

    def _init_nodes(self):
        if self.is_sm:
            self.sm_id = self.num_node - 1
            # update self.graph s.t. graph[n-1]=0
            self.graph[self.sm_id] = np.zeros(self.num_node)

        for i, pow in enumerate(self.power):
            if i == self.sm_id:
                if self.is_sm:
                    self.nodes.append(node.SmNode(i, pow, self.graph[i]))
                else:
                    self.nodes.append(node.Node(i, pow, self.graph[i]))
            elif i == self.single_tft_id:
                if self.tft_mode != info.Tft.NONE.value:
                    self.nodes.append(node.TftNode(i, pow, self.graph[i]))
                else:
                    self.nodes.append(node.Node(i, pow, self.graph[i]))
            else:
                if self.tft_mode == info.Tft.ALL.value:
                    self.nodes.append(node.TftNode(i, pow, self.graph[i]))
                else:
                    self.nodes.append(node.Node(i, pow, self.graph[i]))


    def _update_tft(self):
        '''
        Modification: Now it's called every round
        1. update count
        2. update pz
        3. update delay
        :return:
        '''

        assert (self.tft_mode != info.Tft.NONE.value)

        '''
        Cases for is_update to be true:
        1. Delete/Insert competitors
        2. Churn happens
        3. Mature happens
        '''
        is_update = False

        competitors = self.blockchain.get_competitors()
        node_range = self.num_node - 1 if self.tft_mode == info.Tft.ALL.value else 1

        # Churn
        if self.height % self.session == 0:
            is_update = True
            churn_node = np.random.randint(0,self.num_node)     # select churn node
            self.life_span[churn_node] = 0                      # reset its life span
            self.mature_nodes[churn_node] = False               # reset its mature
            self.pow_count[churn_node] = 0
            for i in range(node_range):
                self.tft_count_mat[i][churn_node] = 0           # reset the tft count to it

        # SM Churn
        if self.height % self.sm_session == 0:
            is_update = True
            self.life_span[self.sm_id] = 0                      # reset its life span
            self.mature_nodes[self.sm_id] = False               # reset its mature
            self.pow_count[self.sm_id] = 0
            for i in range(node_range):
                self.tft_count_mat[i][self.sm_id] = 0           # reset the tft count to it

        # delete old competitors
        if self.window.qsize() < self.win_len:
            pass
        else:
            assert(self.window.qsize() == self.win_len)
            old_competitors = self.window.get()

            # update pow count
            for i in old_competitors:
                if self.life_span[i] >= self.win_len:
                    self.pow_count[i] -= 1
                    assert(self.pow_count[i] >= 0)

            # have old competitors, remove from tft_count_mat
            if len(old_competitors) > 1:
                for i in range(node_range):
                    for j in old_competitors:
                        if self.life_span[j] >= self.win_len:
                            is_update = True
                            # old_tft_count = self.tft_count_mat[i][j]
                            self.tft_count_mat[i][j] -= count_rule(i, j, old_competitors)

        # add new competitors
        self.window.put(competitors)
        for i in competitors:
            self.pow_count[i] += 1
        if len(competitors) > 1:
            is_update = True
            for i in range(node_range):
                for j in competitors:
                    self.tft_count_mat[i][j] += count_rule(i, j, competitors)

        # Update life span and mature
        for i in range(self.num_node):
            self.life_span[i] += 1
            if not self.mature_nodes[i] and self.life_span[i] >= self.mature_time:
                is_update = True
                self.mature_nodes[i] = True

        # TFT update condition
        # if self.blockchain.fork_count == TFT_THRESHOLD:
        if self.window.qsize() == self.win_len and is_update:
            self.tft_work = True

            if self.checkpoint == 0:
                self.checkpoint = self.blockchain.current_height()
            # self.tft_trigger += 1

            # Step 2: update pz
            # EM algorithm to compute prob

            assert ((np.diagonal(self.tft_count_mat) == np.zeros(self.num_node)).all())
            # row[n-1] for sm should be 0
            assert ((self.tft_count_mat[node_range] == np.zeros(self.num_node)).all())
            assert (np.all(self.tft_count_mat) >= 0)

            # compute data matrix where node's count is stretched according to win_len/life_span
            mature_data = copy.deepcopy(self.tft_count_mat)
            for i in range(self.num_node):
                if self.mature_time <= self.life_span[i] < self.win_len:
                    for j in range (self.num_node - 1):
                        mature_data[j][i] *= self.win_len / self.life_span[i]

            # compute mask matrix
            mask_mat = np.tile(self.mature_nodes, (self.num_node, 1))
            diag_mat = ~np.eye(self.num_node, dtype=bool)
            mask_mat = mask_mat * diag_mat
            mask_mat[-1] = np.full(self.num_node, False)

            # estimate pow
            estm_pow = np.zeros(self.num_node)
            for i in range(self.num_node):
                if self.life_span[i] < self.mature_time:
                    assert(self.mature_nodes[i] == False)
                elif self.mature_time <= self.life_span[i] < self.win_len:
                    assert (self.mature_nodes[i])
                    estm_pow[i] = self.pow_count[i] / self.life_span[i]
                else:
                    assert (self.mature_nodes[i])
                    estm_pow[i] = self.pow_count[i] / self.win_len

            # If oracle, knows the exact power
            if self.is_oracle:
                estm_pow = self.power

            # masked elements in pz_mat are zeros.
            # Get the suspicious probability by SM algorithm
            pz_mat = self.em_algo.run(mature_data, mask_mat, estm_pow)

            # Step 3: update delay vector
            self.tft_delay_mat = np.zeros((self.num_node, self.num_node))

            A = np.tile(self.power, (self.num_node, 1))                         # computing power : a
            LB = self.graph                                                     # lower bound: distance

            step = self.dis_mean * self.cv * self.step_scale                    # step size

            for node_id in range(node_range):
                if np.all(pz_mat[node_id] == 0):
                    delay_row = LB[node_id]
                    max_mature = max(LB[node_id][mask_mat[node_id]])
                    max_immature = max(LB[node_id][~mask_mat[node_id]])
                    last_arrival = max(max_immature, max_mature)
                    last_mature_node = np.where(LB[node_id] == max_mature)
                    delay_row[last_mature_node] = last_arrival
                    delay_row[~mask_mat[node_id]] = last_arrival
                    delay_row[node_id] = 0
                else:
                    a = A[node_id][mask_mat[node_id]]
                    lb =LB[node_id][mask_mat[node_id]]
                    b = pz_mat[node_id][mask_mat[node_id]] / (np.ones(len(a)) - a)      # Weighted suspcious probability: b = p / (1-a)
                    order = np.argsort(b / a)                                           # Ascending order of b/a
                    a = a[order]
                    b = b[order]
                    lb = lb[order]

                    x = solver.step_algo(lb, a, b, self.dis_mean * self.cv, step=step)  # STA algorithm to compute the delay vector

                    # Update last one and immature nodes have the same expected arrivals
                    max_arrival = max(LB[node_id][~mask_mat[node_id]])
                    last_arrival = max(max_arrival, x[-1])
                    x[-1] = last_arrival

                    delay_row = np.ones(self.num_node) * last_arrival
                    # assign 0 at i-th
                    delay_row[node_id] = 0
                    # assign tft delays to mature nodes
                    rev_order = np.argsort(order)
                    delay_row[mask_mat[node_id]] = x[rev_order]

                # print(delay_row)
                self.tft_delay_mat[node_id] = delay_row

    def _get_M(self, seed):
        '''
        :param seed: use round as seed to generate same norm variates in the same round
        :return: M
        '''

        n = self.num_node
        # mean_mat = self.tft_delay_mat if self.tft_mode and seed > TFT_THRESHOLD else self.graph
        if self.tft_work:
            mean_mat = self.tft_delay_mat
        else:
            mean_mat = self.graph
        # diagonal of mean_mat is 0
        assert ((np.diagonal(mean_mat) == np.zeros(n)).all())

        var_mat = np.ones((n, n)) * self.dis_mean * self.cv
        M = norm.rvs(loc=mean_mat, scale=var_mat, size=(n,n), random_state=seed)

        # negative -> mean val * decay factor
        M[M < 0] = mean_mat[M<0] * self.cv
        # diagonal -> 0
        np.fill_diagonal(M, 0)
        # M[s] = 0
        M[n-1] = np.zeros(n)

        return M

    def _winners(self, seed, sm_time = info.SmTime.LATEST):
        '''
        :param sm_time: LATEST: sm.timestamp, RELEASE: ddl, NONENGAGE: np.inf
        :return:
        '''
        w_pool = []
        w = np.ones(self.num_node, dtype=int) * -1
        c = 0

        # sort p[i] for all i, O(n^2logn)
        n = self.num_node
        a = [node.get_timestamp(sm_time) for node in self.nodes]

        M = self._get_M(seed)

        # use lexsort to break tie in sorting for sm:
        b = np.ones(n)
        b[self.sm_id] = 0 # higher priority

        p = np.array([np.lexsort((b, a + M[:, i])) for i in range(n)])

        # compute w
        while True:
            assert (c < n)
            # initial winner select
            for i in range(n):
                if w[i] not in w_pool:
                    w[i] = p[i][c]  # c-th min, O(1) for sroted p[i], p[i][c]
            w_pool = np.unique(w)
            len0 = len(w_pool)

            # check w_pool
            w_pool = [win for win in w_pool if w[win] == win]
            len1 = len(w_pool)

            # stop criteria
            if len0 == len1:
                break
            c += 1

        return w

    # winner selection: latest timestamp
    # post update: release/arrival timestamp
    def _post_timestamps(self, seed, winners, sm_block_time=0):
        M = self._get_M(seed)
        effect_hashrate = 0

        for i in range(self.num_node):
            # compute effective hash rate
            # interval = self.nodes[winners[i]].get_timestamp(info.SmTime.RELEASE) - self.nodes[i].timestamp + self.nodes[i].interval + self.nodes[i].last_delay
            arrival = self.nodes[i].last_delay
            interval = self.nodes[i].last_delay + self.nodes[winners[i]].get_timestamp(info.SmTime.RELEASE) + M[winners[i]][i] - (self.nodes[i].timestamp - self.nodes[i].interval)
            effect_hashrate += (1 - arrival / interval) * self.nodes[i].power

            # assert(interval > 0)
            if interval <= 0:
                print("height: {}, node: {}, sm_release: {}, sm_timestamp: {}, i_timestamp: {}, i_interval: {}, i_last_delay: {}"
                      .format(self.height, i, self.nodes[winners[i]].get_timestamp(info.SmTime.RELEASE), self.nodes[winners[i]].get_timestamp(info.SmTime.LATEST), self.nodes[i].timestamp, self.nodes[i].interval, self.nodes[i].last_delay))
                print(arrival, interval)

            # only update when winners[i] != i
            if winners[i] != i:
                self.nodes[i].timestamp = self.nodes[winners[i]].get_timestamp(info.SmTime.RELEASE) + \
                                      M[winners[i]][i] # self.tft_delay_mat[winners[i]][i]

            # update last-delay
            if self.is_sm and winners[i] == self.sm_id:
                assert(sm_block_time > 0)
                self.nodes[i].last_delay = self.nodes[i].timestamp - sm_block_time
            else:
                self.nodes[i].last_delay = M[winners[i]][i]  #self.tft_delay_mat[winners[i]][i]

        self.single_effect_hashrate.append(effect_hashrate)

    def _current_process(self, last_winner):
        blocks = []

        # set winner, add block
        winners = self._winners(self.height)
        w_pool = np.unique(winners)
        for w in w_pool:
            blocks.append([blockchain.Block(self.height, w, self.nodes[w].timestamp, last_winner[w])])

        # update timestamp for next round
        self._post_timestamps(self.height, winners)
        # update height
        self.height += 1

        return winners, blocks


    def _current_process_sm_full(self, last_winner):
        '''
        :param height:
        :param last_winner:
        :return: blocks

        state for sm:
        s0 -> s1 -> s2 -> sn: these transitions are triggered by sm itself.
        sn->s2->s0, s1->s0'->s0, are triggered by h. need to recompute their starting time.

        therefore, for each iteration in the main loop in run(), we consider it as a round where h mines one block,
        while no restriction is cast on sm.

        unified (no fork from sm)
        split (fork)

        s0:
            - 0: h win, unified
            - 1: -> s0', split
            - 2: ->s0, unified | reveal 2 consecutive blocks
            - k>2: -> sk-1, split
        s0':
            - 0: -> s0, unified, | h_h/h_sm win
            - k>0: -> s0, unified | sm win, recompute starting time
        s2: (s2 can only be reached when s0+3(s3)->s2, s2+1(s3)->s2 or directly s3->s2)
            - 0: ->s0', split
            - k>0：->s2+n-1, split
        sn (n>2):
            - 0: ->sn-1, split
            - k>0：->sn+k-1, split

        measure gamma at state when split
        '''

        blocks = []

        M = self._get_M(self.height)
        s = self.sm_id
        sm = self.nodes[s]

        # set winners
        # 1. winners_all: all miners (honest miners and selfish miner) mine 1 round, compute the normal result
        # 2. winners_no_sm: set sm's timestamp as np.inf, i.e. compute the result where sm never mines,
        #                   s.t. we know the time t_h when sm hears from honest miners
        # 3. winners_reveal: set sm's timestamp as t_h, compute the result where sm conceal and reveals.

        winners_all = self._winners(self.height)
        w_pool_all = np.unique(winners_all)

        # decide when sm hears from h
        if winners_all[s] == s: # sm mines at least 1 block
            if sm.state == -1:  # is sm.state == -1, only compute one block in pre-timestamp
                k = 1
                sm.release_time = sm.timestamp
                winners = winners_all
                w_pool = w_pool_all
            else:
                # Compute winners_no_sm to get when will sm first hear from honest miners.
                winners_no_sm = self._winners(self.height, info.SmTime.NONENGAGE)
                sm_h = winners_no_sm[s]

                # Compute ddl
                sm_h_time = self.nodes[sm_h].timestamp + M[sm_h][s]

                # compute k (#blocks mined by sm in this round)
                assert(sm_h_time - self.nodes[s].timestamp > 0)
                k = sm.sm_mine(sm_h_time)
                assert(k >= 1)

                # compute winners_reveal & w_pool_reveal, which are real when sm mines.
                winners_reveal = self._winners(self.height, info.SmTime.RELEASE)
                w_pool_reveal = np.unique(winners_reveal)
                winners = winners_reveal
                w_pool = w_pool_reveal
        else:
            k = 0
            winners = winners_all
            w_pool = w_pool_all

        # print(sm.state, k)

        # state machine
        if sm.state == 0:       # s0
            # h win, unified
            if k == 0:
                assert(s not in w_pool_all)
                assert(sm.sm_timestamps.qsize() == 1)

                # add block
                for w in w_pool:
                    blocks.append([blockchain.Block(self.height, w, self.nodes[w].timestamp, last_winner[w])])

                # empty sm_timestamps
                sm.sm_timestamps.get()
                assert (sm.sm_timestamps.empty())

                # update timestamp for next round
                self._post_timestamps(self.height, winners)

                new_state = 0  # unchanged

                # s0
                self.state_count[0] += 1

            #  -> s0', split
            elif k == 1:
                assert (s in w_pool_all and s in w_pool)
                assert (sm.sm_timestamps.qsize() == 2)

                # measure gamma when split
                if self.tft_mode == info.Tft.NONE.value or self.tft_mode == info.Tft.ALL.value and self.tft_work:
                    sm_win_powers = [node.power for node in self.nodes if winners[node.id] == s and node.id != s]
                    gamma = np.sum(sm_win_powers) / (1 - sm.power)
                    self.single_gammas.append(gamma)

                # add block
                for w in w_pool:
                    if w == s:
                        sm_block_time = sm.sm_timestamps.get()
                        blocks.append([blockchain.Block(self.height, s, sm_block_time, last_winner[s])])
                    else:
                        blocks.append([blockchain.Block(self.height, w, self.nodes[w].timestamp, last_winner[w])])

                assert (sm.sm_timestamps.qsize() == 1)

                # update timestamp for next round
                self._post_timestamps(self.height, winners, sm_block_time)

                new_state = -1  # s0'

                # s1, s0'
                self.state_count[1] += 1
                self.state_count[4] += 1

            # -> s0, unified | reveal 2 consecutive blocks
            # winners should all be s
            elif k == 2:
                assert (s in w_pool_all and s in w_pool)
                assert (sm.sm_timestamps.qsize() == 3)

                # winners should all be s, since it reveals two blocks
                winners = s * np.ones(self.num_node, dtype=int)

                # add block
                for w in w_pool_reveal:
                    if w == s:
                        sm_block_time = sm.sm_timestamps.get()
                        blocks.append(
                                [blockchain.Block(self.height, s, sm_block_time, last_winner[s]),
                                blockchain.Block(self.height + 1, s, sm.sm_timestamps.get(), s)]
                        )
                    else:
                        blocks.append([blockchain.Block(self.height, w, self.nodes[w].timestamp, last_winner[w])])

                assert (len(blocks) >= 2)

                assert(sm.sm_timestamps.qsize() == 1)

                # update timestamp for next round
                self._post_timestamps(self.height, winners, sm_block_time)
                self.height += 1

                new_state = 0

                # s1, s2, s0
                self.state_count[1] += 1
                self.state_count[2] += 1
                self.state_count[0] += 1

            # -> sk - 1, split
            else:
                assert (s in w_pool_all and s in w_pool)
                assert (sm.sm_timestamps.qsize() == k + 1)

                # measure gamma when split
                if self.tft_mode == info.Tft.NONE.value or self.tft_mode == info.Tft.ALL.value and self.tft_work:
                    sm_win_powers = [node.power for node in self.nodes if winners[node.id] == s and node.id != s]
                    gamma = np.sum(sm_win_powers) / (1 - sm.power)
                    self.single_gammas.append(gamma)

                # add block
                for w in w_pool:
                    if w == s:
                        sm_block_time = sm.sm_timestamps.get()
                        blocks.append([blockchain.Block(self.height, s, sm_block_time, last_winner[s])])
                    else:
                        blocks.append([blockchain.Block(self.height, w, self.nodes[w].timestamp, last_winner[w])])

                assert(sm.sm_timestamps.qsize() == k)

                # update timestamp for next round
                self._post_timestamps(self.height, winners, sm_block_time)

                new_state = k - 1

                # s1~sk, sk-1
                self.state_count[1] += 1
                self.state_count[2] += 1
                self.state_count[3] += k - 2
                # [min, max] = [2, 3]
                self.state_count[np.clip([k-1], 2, 3)[0]] += 1
                # if k == 3:
                #     self.state_count[2] += 1
                # else:
                #     self.state_count[3] += 1

        elif sm.state == -1:    # s0'
            # -> s0, unified, | h_h/h_sm win
            if k == 0:
                assert (s not in w_pool_all)
                assert (sm.sm_timestamps.qsize() == 1)

                # add block
                for w in w_pool:
                    blocks.append([blockchain.Block(self.height, w, self.nodes[w].timestamp, last_winner[w])])

                # empty sm_timestamps
                sm.sm_timestamps.get()
                assert (sm.sm_timestamps.empty())

                # update timestamp for next round
                self._post_timestamps(self.height, winners)

                new_state = 0

                # s0
                self.state_count[0] += 1

            # -> s0, unified | sm win, recompute starting time
            else:
                assert (s in w_pool)
                assert (sm.sm_timestamps.qsize() == 1)

                # add block
                for w in w_pool:
                    if w == s:
                        sm_block_time = sm.sm_timestamps.get()
                        blocks.append([blockchain.Block(self.height, s, sm_block_time, last_winner[s])])
                    else:
                        blocks.append([blockchain.Block(self.height, w, self.nodes[w].timestamp, last_winner[w])])

                assert (sm.sm_timestamps.empty())

                # update timestamp for next round
                self._post_timestamps(self.height, winners, sm_block_time)

                new_state = 0

                # s0
                self.state_count[0] += 1

        elif sm.state == 2:     # s2
            # -> s0, unified | sm win by revealing 2
            # winners should all be sm with release time
            if k == 0:
                assert (s not in w_pool_all)
                assert (sm.sm_timestamps.qsize() == 3)

                # add block
                winners_no_sm = self._winners(self.height, info.SmTime.NONENGAGE)
                sm_h = winners_no_sm[s]

                # Compute ddl
                sm_h_time = self.nodes[sm_h].timestamp + M[sm_h][s]
                # here ddl is smaller thant sm's timestamp since sm mines 0 blocks.
                sm.release_time= sm_h_time
                assert(sm.release_time < sm.timestamp)

                winners_reveal = self._winners(self.height, info.SmTime.RELEASE)
                w_pool_reveal = np.unique(winners_reveal)

                # winners should all be s, since it reveals two blocks
                winners = s * np.ones(self.num_node, dtype=int)

                # add block
                for w in w_pool_reveal:
                    if w == s:
                        sm_block_time = sm.sm_timestamps.get()
                        blocks.append(
                            [blockchain.Block(self.height, s, sm_block_time, last_winner[s]),
                             blockchain.Block(self.height + 1, s, sm.sm_timestamps.get(), s)]
                        )
                    else:
                        blocks.append([blockchain.Block(self.height, w, self.nodes[w].timestamp, last_winner[w])])

                assert(len(blocks) >= 2)

                assert (sm.sm_timestamps.qsize() == 1)

                # update timestamp for next round
                self._post_timestamps(self.height, winners, sm_block_time)
                self.height += 1

                new_state = 0

                # s0
                self.state_count[0] += 1

            # ->s2+n-1, split
            else:
                assert (s in w_pool_all and s in w_pool)
                assert (sm.sm_timestamps.qsize() == k + 3)

                # measure gamma when split
                if self.tft_mode == info.Tft.NONE.value or self.tft_mode == info.Tft.ALL.value and self.tft_work:
                    sm_win_powers = [node.power for node in self.nodes if winners[node.id] == s and node.id != s]
                    gamma = np.sum(sm_win_powers) / (1 - sm.power)
                    self.single_gammas.append(gamma)

                # add block
                for w in w_pool:
                    if w == s:
                        sm_block_time = sm.sm_timestamps.get()
                        blocks.append([blockchain.Block(self.height, s, sm_block_time, last_winner[s])])
                    else:
                        blocks.append([blockchain.Block(self.height, w, self.nodes[w].timestamp, last_winner[w])])

                assert (sm.sm_timestamps.qsize() == k + 2)

                # update timestamp for next round
                self._post_timestamps(self.height, winners, sm_block_time)

                new_state = 2 + k - 1

                # s3~sk+2, sk+1
                self.state_count[3] += k
                self.state_count[np.clip([k + 1], 2, 3)[0]] += 1

        else:
            assert(sm.state > 2)  # sn (n>2)

            # ->sn - 1, split
            if k == 0:
                assert (s not in w_pool_all)
                assert (sm.sm_timestamps.qsize() == sm.state + 1)

                # recompute winners
                winners_no_sm = self._winners(self.height, info.SmTime.NONENGAGE)
                sm_h = winners_no_sm[s]

                # Compute ddl
                sm_h_time = self.nodes[sm_h].timestamp + M[sm_h][s]
                # here ddl is smaller thant sm's timestamp since sm mines 0 blocks.
                sm.release_time = sm_h_time
                assert (sm.release_time < sm.timestamp)

                winners_reveal = self._winners(self.height, info.SmTime.RELEASE)
                w_pool_reveal = np.unique(winners_reveal)
                winners = winners_reveal
                w_pool = w_pool_reveal

                # measure gamma when split
                if self.tft_mode == info.Tft.NONE.value or self.tft_mode == info.Tft.ALL.value and self.tft_work:
                    sm_win_powers = [node.power for node in self.nodes if winners[node.id] == s and node.id != s]
                    gamma = np.sum(sm_win_powers) / (1 - sm.power)
                    self.single_gammas.append(gamma)

                # add block
                for w in w_pool:
                    if w == s:
                        sm_block_time = sm.sm_timestamps.get()
                        blocks.append([blockchain.Block(self.height, s, sm_block_time, last_winner[s])])
                    else:
                        blocks.append([blockchain.Block(self.height, w, self.nodes[w].timestamp, last_winner[w])])

                assert (sm.sm_timestamps.qsize() == sm.state)

                # update timestamp for next round
                self._post_timestamps(self.height, winners, sm_block_time)

                new_state = sm.state - 1

                # sn-1 (n>=3) sn-1 >= s2
                self.state_count[np.clip([new_state], 2, 3)[0]] += 1


            # ->sn+k-1, split
            else:
                assert (s in w_pool_all and s in w_pool)
                assert (sm.sm_timestamps.qsize() == sm.state + k + 1)

                # measure gamma when split
                if self.tft_mode == info.Tft.NONE.value or self.tft_mode == info.Tft.ALL.value and self.tft_work:
                    sm_win_powers = [node.power for node in self.nodes if winners[node.id] == s and node.id != s]
                    gamma = np.sum(sm_win_powers) / (1 - sm.power)
                    self.single_gammas.append(gamma)

                # add block
                for w in w_pool:
                    if w == s:
                        sm_block_time = sm.sm_timestamps.get()
                        blocks.append([blockchain.Block(self.height, s, sm_block_time, last_winner[s])])
                    else:
                        blocks.append([blockchain.Block(self.height, w, self.nodes[w].timestamp, last_winner[w])])

                assert (sm.sm_timestamps.qsize() == sm.state + k)

                # update timestamp for next round
                self._post_timestamps(self.height, winners, sm_block_time)

                new_state = sm.state + k - 1

                # sn+1~sn+k, sn+k-1
                self.state_count[3] += k + 1

        sm.state = new_state
        self.height += 1

        return winners, blocks


    def _single_result(self):
        rev_count = self.blockchain.get_revenue(self.checkpoint)
        # revenue = rev_count / (self.height + self.num_node)
        # print(f'height: {self.height}, sum_revenue: {np.sum(rev_count)}')
        revenue = rev_count / np.sum(rev_count)
        kld = kl_divergence(self.power, revenue)
        fork_rate = self.blockchain.fork_count / self.height
        fork_distrib = self.blockchain.forks / np.sum(self.blockchain.forks)

        self.total_kl_diversity.append(kld)
        self.total_fork_rate.append(fork_rate)
        self.total_interval.append(self.blockchain.get_interval())
        self.total_effect_hashrate.append(np.mean(self.single_effect_hashrate))

        if self.is_sm:
            # response for every single run
            alpha = self.power[self.sm_id]
            gamma = np.mean(np.array(self.single_gammas))
            sm_rev = revenue[self.sm_id]

            self.total_gamma.append(gamma)
            self.total_sm_revenue.append(sm_rev)

            if self.tft_mode == info.Tft.ALL.value:
                median_tft = true_med_dis(self.tft_delay_mat, rev_count)
                self.total_median_tft.append(median_tft)


    def _batch_result(self):
        mean_kl_diversity = np.mean(self.total_kl_diversity)
        mean_fork_rate = np.mean(self.total_fork_rate)
        mean_interval = np.mean(self.total_interval)
        mean_effect_hashrate = np.mean(self.total_effect_hashrate)
        if self.is_sm:
            mean_gamma = np.mean(self.total_gamma)
            sm_power = self.power[self.sm_id]
            mean_sm_revenue = np.mean(self.total_sm_revenue)
            if self.tft_mode:
                mean_median_tft = np.mean(self.total_median_tft)
                return info.TftInfo(self.num_node, mean_kl_diversity, mean_fork_rate, mean_interval, mean_effect_hashrate, mean_gamma, sm_power, mean_sm_revenue, mean_median_tft)
            else:
                return info.SmInfo(self.num_node, mean_kl_diversity, mean_fork_rate, mean_interval, mean_effect_hashrate, mean_gamma, sm_power, mean_sm_revenue)
        else:
            return info.Info(self.num_node, mean_kl_diversity, mean_fork_rate, mean_interval, mean_effect_hashrate)

    def _clear(self):
        for node in self.nodes:
            node.clear()
        self.single_gammas.clear()
        self.single_effect_hashrate.clear()

        self.height = 1

        self.tft_trigger = 1
        self.tft_count_mat = np.zeros((self.num_node, self.num_node))
        self.tft_delay_mat = np.zeros((self.num_node, self.num_node))

        del self.blockchain
        del self.em_algo
        self.blockchain = blockchain.BlockChain(self.num_node)
        self.em_algo = em.Em(self.power)

        self.checkpoint = 0
        self.tft_work = False
        self.state_count = np.zeros(5)

        self.window.queue.clear()
        self.life_span = np.zeros(self.num_node)
        self.mature_nodes = np.full(self.num_node, False)

        self.power = self.org_power
        self.pow_count = np.zeros(self.num_node)
        for i in range(self.num_node):
            self.graph[i] = self.nodes[i].dis

    def run(self, repeats):
        # print(self.graph)
        for repeat in range(repeats):

            winner = -1 * np.ones(self.num_node, dtype=int)

            for round in range(self.rounds):   # enter each round

                if round % self.pow_cycle == 0 and self.pow_range: # update pow
                    for i, node in enumerate(self.nodes):
                        self.power[i] = node.update_pow(self.pow_range)
            
                # pre-timestamp
                for node in self.nodes:
                    # sm only mines when sm_timestamps is empty
                    node.mine()
            
                # compute winner and current available blocks (fork happens if len > 1)
                if self.is_sm:
                    winner, available_blocks = self._current_process_sm_full(winner)
                else:
                    winner, available_blocks = self._current_process(winner)
                assert len(available_blocks) >= 1

                # add new block and deal with fork
                if len(available_blocks) == 1: # no fork
                    self.blockchain.new_blocks(available_blocks[0])
                else: # fork happens
                    self.blockchain.new_competing_blocks(available_blocks)

                # update tft count
                if self.tft_mode != info.Tft.NONE.value:
                    self._update_tft()
            self._single_result()
            self._clear()

        return self._batch_result()