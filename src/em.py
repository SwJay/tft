import numpy as np
import copy
import time
import os
import sys

EM_REPEATS = 5


class Em():

    def __init__(self, power):
        self.n = len(power)
        self.datas = np.zeros((self.n, self.n))
        self.d_range = np.zeros(self.n)

        self.power = power                                          # Will be replaced with estimated power in the window
        self.thetas = np.tile(power, (self.n, 1))                   # Initial parameter as power
        self.pzs = np.zeros((self.n, self.n))                       # Probability of latent variables

        self.mask_mat = np.zeros((self.n, self.n))                  # Diagonal, row n-1 and immature nodes are masked.

    # Expectation
    def e_step(self, node):                                         # Update pzs
        for i, data in enumerate(self.datas[node]):
            if self.mask_mat[node][i]:                              # Fork count from mature node
                for j, theta in enumerate(self.thetas[node]):
                    if self.mask_mat[node][j]:                      # Theta from mature node
                        if theta == 0:
                            self.pzs[i][j] = 0
                        elif theta == 1:
                            self.pzs[i][j] = 1
                        else:                                       # Update latent probability
                            self.pzs[i][j] = np.power(theta, data) * np.power((1 - theta), (self.d_range[node] - data))
                self.pzs[i] = self.pzs[i] / np.sum(self.pzs[i])     # Normalize

    # Maximization
    def m_step(self, node):
        contribs = np.zeros((self.n, 2))
        for i, pz in enumerate(self.pzs):
            if self.mask_mat[node][i]:                              # Fork counte from mature node
                for j in range(self.n):
                    if self.mask_mat[node][j]:                      # MLE on theta of mature nodes
                        contribs[j][0] += pz[j] * self.datas[node][i]
                        contribs[j][1] += pz[j] * (self.d_range[node] - self.datas[node][i])
        for j in range(self.n):
            if self.mask_mat[node][j]:
                if contribs[j][0] + contribs[j][1] == 0:
                    self.thetas[node][j] = 0
                else:                                               # Normalize
                    self.thetas[node][j] = contribs[j][0] / (contribs[j][0] + contribs[j][1])

    def run(self, count_mat, mask_mat, estm_pow):
        n = len(count_mat)
        assert(n == self.n)

        self.mask_mat = mask_mat                                    # Mask immature's count, diagonal and sm
        self.power = estm_pow                                       # Estimated power from the window
        self.datas = count_mat * mask_mat
        self.d_range = np.sum(self.datas, axis=1)                   # Sum of fork count each node observed
        self.thetas = np.tile(self.power, (n, 1)) * self.mask_mat   # Initial parameter setting as their estiamted power

        pz_sm_matrix = np.zeros((n, n))

        for node_id in range(n-1):
            if np.all(self.datas[node_id] == 0):                    # Early stop
                continue
            for i in range(EM_REPEATS):                             # Begin iteration of EM algorithm
                self.e_step(node_id)
                self.m_step(node_id)

            match_sm = np.argmax(self.thetas[node_id])              # Find node with max theta
            pz_sm = self.pzs[:, match_sm] / np.sum(self.pzs[:, match_sm])
            pz_sm_matrix[node_id] = pz_sm

            # clean
            self.thetas = np.tile(self.power, (n, 1)) * self.mask_mat
            self.pzs = np.zeros((n, n))

        return pz_sm_matrix