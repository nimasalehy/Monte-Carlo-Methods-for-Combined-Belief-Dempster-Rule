import numpy as np


# returns 1 if A is a subset of B, 0 otherwise
def asubsetb(A, B):
    l = len(A)
    if len(B) != l:
        print('Error! The length must be equal.')
    for i in range(l):
        if A[i] > B[i]:
            return 0
    return 1


# Fmj = [np.zeros((int(nfoc[j]), nt)) for j in range(n)], nfoc = # of focal sets
# mj = [np.zeros(int(nfoc[j])) for j in range(n)]
def select(j, Fmj_set, mj, u):
    # m = mj[j]
    # F = Fmj[j]
    # u = np.random.uniform(0, 1)
    k = 0
    temp = mj[j][0]
    while u >= temp:
        k += 1
        temp += mj[j][k]
    return Fmj_set[j][k]


# Number of iterations N, Number of mass functions n, Bodies of evidence Fmj,
# masses mj, Cardinality of \Theta is nt
def qsimplebel(N, n, Fmj_set, mj, Theta_set, A_set, uq):
    T1 = 0.
    T2 = 0.
    for i in range(N):
        # B = np.zeros((n, nt))
        Int = Theta_set
        for j in range(n):
            Int = Int.intersection(select(j, Fmj_set, mj, uq[i][j]))

        if sum(Int):
            T1 += 1
            if Int.issubset(A_set):
                T2 += 1
    return T2/T1
