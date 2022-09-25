import numpy as np
import queue
from info import SmTime

INTERVAL = 600      # block interval 10 min


class Node:

    def __init__(self, id, pow, dis):
        self.id = id
        self.power = pow
        self.dis = dis
        self.timestamp = 0

        self.interval = 0
        self.last_delay = 0

        self.org_pow = pow                                      # Record initial power
        self.rat_pow = 1                                        # Varying ratio of power

    def show(self):
        print('==============')
        print(f'id:{self.id}, power: {self.power}, dis:{self.dis}')

    def mine(self):
        t = INTERVAL / self.power                               # Expected interval time for node
        self.interval = np.random.exponential(t, size=1)[0]     # Follow exponential distribution
        self.timestamp += self.interval

    def get_timestamp(self, sm_time):
        return self.timestamp

    def update_pow(self, pow_range):                            # Randomly update power within range
        self.power = self.org_pow * np.random.uniform(1 - pow_range, 1 + pow_range)
        return self.power

    def clear(self):
        self.timestamp = 0
        self.interval = 0
        self.last_delay = 0

        self.power = self.org_pow                               # Reset its power
        self.rat_pow = 1

class SmNode(Node):

    def __init__(self, id, pow, dis):
        self.id = id
        self.power = pow
        self.revenue = 0
        self.dis = dis
        self.timestamp = 0
        self.interval = 0
        self.last_delay = 0

        self.release_time = 0                                   # Time to reveal
        self.sm_timestamps = queue.Queue()                      # Queue storing timestamps of private blocks
        self.state = 0                                          # Record state in selfish mining state machine

        self.org_pow = pow                                      # Record initial power
        self.rat_pow = 1                                        # Varying ratio of power

    # mining(timestamp) for sm
    # mine: only mines at initial block
    # sm_mine: record last block that surpass first arrival to sm and set that as timestamp.

    def mine(self):
        # only pre mine when sm_timestamp queue is empty
        if self.sm_timestamps.empty():
            t = INTERVAL / self.power                           # Expected interval time for node
            self.interval = np.random.exponential(t, size=1)[0] # Follow exponential distribution
            self.timestamp += self.interval
            self.release_time = self.timestamp
            self.sm_timestamps.put(self.timestamp)


    def sm_mine(self, ddl):

        '''
        :param ddl: the timestamp of h's first arrival to sm
        :return:
        '''

        k = 1
        self.release_time = ddl

        while True:
            t = INTERVAL / self.power
            self.interval = np.random.exponential(t, size=1)[0]
            self.timestamp += self.interval
            self.sm_timestamps.put(self.timestamp)
            if self.timestamp >= ddl:                           # SM may secretly mine multiple blocks before hearing from honest
                break
            k += 1

        return k

    # timestamp for sm
    def get_timestamp(self, sm_time=SmTime.LATEST):
        if sm_time == SmTime.LATEST:                            # Return the latest timestamp
            return self.timestamp
        elif sm_time == SmTime.RELEASE:                         # Retuen the withheld timestamp to reveal
            return self.release_time
        elif sm_time == SmTime.NONENGAGE:                       # np.infty: used to simulate the absence of SM
            return np.inf

    def clear(self):
        self.state = 0
        self.timestamp = 0
        self.interval = 0
        self.last_delay = 0

        self.power = self.org_pow
        self.rat_pow = 1

        self.release_time = 0
        self.sm_timestamps.queue.clear()

class TftNode(Node):
    pass
