import numpy as np


# ------------------------------------------------------------------------
# ----------------- Part 1: policy evaluation ----------------------------
# ------------------------------------------------------------------------
#
# functions: rules of the model
def rule(s, a):
    if a == 1:
        if s == 0:
            s1 = np.random.binomial(1, 2/3, 1)
        if s == 1:
            s1 = np.random.binomial(1, 3/4, 1)
    if a == 2:
        if s == 0:
            s1 = np.random.binomial(1, 0.5, 1)
        if s == 1:
            s1 = np.random.binomial(1, 1/3, 1)
    return s1


# generate one episode E of 10000 triplets of (Ri, Si, Ai)
Chain = np.zeros([10000, 3])
Chain[0, 0] = 0
Chain[0, 1] = 0
Chain[0, 2] = 1

for k in range(1, 10000):
    if Chain[k - 1, 0] == 0:
        s_temp = rule(0, 1)
    else:
        s_temp = rule(1, 2)
    Chain[k, 0] = s_temp
    Chain[k, 1] = 1 if s_temp == 0 else 2
    Chain[k, 2] = 1 if s_temp == 0 else 2


# function: Monte Carlo policy evaluation to estimate the value function V.
def val_func_mc(chain, gamma, tol):
    N = chain.shape[0]
    G = np.zeros(N)
    for i in range(N):
        temp = 0
        for j in range(N-i):
            temp0 = temp
            temp += (gamma ** j) * chain[i+j, 1]
            if abs(temp - temp0) / (abs(temp0) + 1e-6) < tol:
                break
        G[i] = temp
    idx0 = np.where(chain[:, 0] == 0)[0]
    idx1 = np.where(chain[:, 0] == 1)[0]
    v0 = np.mean(G[idx0[1:]])
    v1 = np.mean(G[idx1[1:]])
    return [v0, v1]


# function: n-step temporal difference policy evaluation to estimate the value function V.
def val_func_boots(chain, gamma, n):
    N = chain.shape[0]
    Ns = np.zeros([N-n])
    V = np.zeros([N-n])
    idx0 = np.where(chain[:(N-n), 0] == 0)[0]
    idx1 = np.where(chain[:(N-n), 0] == 1)[0]
    Ns[idx0] = np.array(range(len(idx0))) + 1
    Ns[idx1] = np.array(range(len(idx1))) + 1
    v_old = np.array([0., 0.])
    for i in range(N-n):
        v_temp = v_old[0] if chain[i, 0] == 0 else v_old[1]
        G = (gamma ** n) * (v_old[0] if chain[i+n, 0] == 0 else v_old[1])
        for j in range(n):
            G += (gamma ** j) * chain[i+j, 1]
        V[i] = v_temp + (G - v_temp) / Ns[i]
        if chain[i, 0] == 0:
            v_old[0] = V[i]
        else:
            v_old[1] = V[i]
    return [V[idx0][len(idx0)-1], V[idx1][len(idx1)-1]]


# policy evaluation results
MC_out = val_func_mc(Chain, 0.75, 1e-6)
# 5.593493758422694, 6.401581991229474
boots_out = val_func_boots(Chain, 0.75, 5)
# 5.586777066512087, 6.393824622590407


# ---------------------------------------------------------------------
# ----------------- Part 2: policy control ----------------------------
# ---------------------------------------------------------------------
#
# reward function
def reward(s, a):
    if a == 1:
        r = 1 if s == 0 else 3
    if a == 2:
        r = 4 if s == 0 else 2
    return r


# SARSA algorithm for estimating the optimal action-value
def SARSA(s, gamma, alpha, N):
    Q = np.zeros([2, 2])
    a = 1 if np.random.binomial(1, 0.5, 1) == 0 else 2
    for t in range(N):
        r = reward(s, a)
        s1 = rule(s, a)
        flag = np.random.binomial(1, 1/np.sqrt(t + 1), 1)
        if flag == 0:
            a1 = 1 if Q[s1, 0] > Q[s1, 1] else 2
        else:
            a1 = 1 if np.random.binomial(1, 0.5, 1) == 0 else 2
        Q[s, a-1] += (r + gamma * Q[s1, a1-1] - Q[s, a-1]) * alpha
        s = s1
        a = a1
    return Q


# Q learning algorithm for estimating the optimal action-value
def Q_learning(s, gamma, alpha, N):
    Q = np.zeros([2, 2])
    for t in range(N):
        flag = np.random.binomial(1, 1/np.sqrt(t + 1), 1)
        if flag == 0:
            a = 1 if Q[s, 0] > Q[s, 1] else 2
        else:
            a = 1 if np.random.binomial(1, 0.5, 1) == 0 else 2
        r = reward(s, a)
        s1 = rule(s, a)
        Q[s, a-1] += (r + gamma * max(Q[s1, 0], Q[s1, 1]) - Q[s, a-1]) * alpha
        s = s1
    return Q


# policy control results
SARSA(0, 0.75, 0.1, 50000)
Q_learning(0, 0.75, 0.1, 50000)






