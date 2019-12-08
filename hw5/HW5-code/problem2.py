import numpy as np

nIter = 1000
V0 = np.array([0, 0])
V = np.zeros([nIter, 2])

for i in range(nIter):
    W01 = 1 + 0.75 * ((1 / 3) * V0[0] + (2 / 3) * V0[1])
    W02 = 4 + 0.75 * ((1 / 2) * V0[0] + (1 / 2) * V0[1])
    W11 = 3 + 0.75 * ((1 / 4) * V0[0] + (3 / 4) * V0[1])
    W12 = 2 + 0.75 * ((2 / 3) * V0[0] + (1 / 3) * V0[1])
    V[i, 0] = max(W01, W02)
    V[i, 1] = max(W11, W12)
    V0 = V[i, :]


V_opt = V[nIter - 1, :]
# [14.15384615, 12.92307692]

W01 = 1 + 0.75 * ((1 / 3) * V_opt[0] + (2 / 3) * V_opt[1])
W02 = 4 + 0.75 * ((1 / 2) * V_opt[0] + (1 / 2) * V_opt[1])
W11 = 3 + 0.75 * ((1 / 4) * V_opt[0] + (3 / 4) * V_opt[1])
W12 = 2 + 0.75 * ((2 / 3) * V_opt[0] + (1 / 3) * V_opt[1])

pi0 = 1 if W01 > W02 else 2
# 2

pi1 = 1 if W11 > W12 else 2
# 1

