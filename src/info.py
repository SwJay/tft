from enum import Enum


class Info:
    def __init__(self, num=0, kld=0, fork_rate=0, interval=0, effective_hashrate=0, gamma=0, sm_power=0, sm_revenue=0):
        self.num = num
        self.kld = kld
        self.fork_rate = fork_rate
        self.interval = interval
        self.effective_hashrate = effective_hashrate

        self.gamma = gamma
        self.sm_power = sm_power
        self.sm_revenue = sm_revenue


class SmInfo(Info):
    pass


class TftInfo(Info):
    def __init__(self, num=0, kld=0, fork_rate=0, interval=0, effective_hashrate=0, gamma=0, sm_power=0, sm_revenue=0, median_tft=0):
        super(TftInfo, self).__init__(num, kld, fork_rate, interval, effective_hashrate, gamma, sm_power, sm_revenue)
        self.median_tft = median_tft


class Tft(Enum):
    """
    0. no tft
    1. single tft
    2. all tft
    """
    NONE, SINGLE, ALL = range(3)


class Test(Enum):
    """
    Test Cases: all batch test require results from different num_node
    0. single result
    1. batch result
    2. power distribution: identical vs random
    3. k_dis: k_dis_list
    4. sm vs non-sm
    5. honest: alpha and gamma
    6. sm: alpha and gamma
    7. tft: alpha and gamma
    8. test median time & cv
    9. test step size
    10. test sm's profitable threshold
    11. test fairness
    12. test peer churn
    13. test dynamic power
    """

    NONE, BATCH, POWER, K_DIS, SM, H_AG, SM_AG, TFT_AG, MT_CV, STEP, THRE, FAIR, CHURN, DYN_POW = range(14)


class SmTime(Enum):
    """
    This is used to get timestamp from sm.
    0. true latest timestamp
    1. release timestamp
    2. not engage: inf
    """
    LATEST, RELEASE, NONENGAGE = range(3)

test_name_list = ['NONE', 'BATCH', 'POWER', 'K_DIS', 'SM', 'H_AG', 'SM_AG',
                  'TFT_AG', 'MT_CV', 'STEP', 'THRE', 'FAIR', 'CHURN', 'DYN_POW']