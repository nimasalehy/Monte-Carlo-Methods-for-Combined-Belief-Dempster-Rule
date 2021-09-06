import numpy as np
from itertools import product


def selcom(qstep1, sigma, powset_set, u):
    k = 0
    temp = qstep1[0]
    while u >= temp:
        k += 1
        temp += qstep1[k]
    return powset_set[sigma[k]]


def contel(j, Fmjnew, mjnew, u):
    stemp = sum(mjnew[j])
    k = 0
    temp = mjnew[j][0]/stemp
    while u >= temp:
        k += 1
        temp += mjnew[j][k]/stemp
    return Fmjnew[j][k]


def step2fun(Fmjnew, mjnew, Theta_set, n, u):
    tempBinter = Theta_set
    for j in range(n):
        tempBinter = tempBinter.intersection(contel(j, Fmjnew, mjnew, u[j]))
    return tempBinter


def cardimp2(intBjs, pset, sig):
    count = 0
    for i in sig:
        if pset[i].issubset(intBjs):
            count += 1
    return count


# Number of iterations N, Number of mass functions n, Bodies of evidence Fmj,
# masses mj, Cardinality of \Theta is nt
def qimp2bel(N, n, Fmj_set, mj, Theta_set, nt, A_set, sigma, powset_set, qstep1, uq):
    # q (list of commonalities, i-th term is an array of numbers for q[i])
    # q = [np.zeros(len(sigma)) for j in range(n)]
    # q = calq(Fmj, mj, sigma, powset, n)
    # qstep1 = qstep1fun(q)

    T1 = 0.
    T2 = 0.
    i = 0
    while i < N:
        # B = np.zeros((n, nt))
        # T = np.ones(nt)
        T = selcom(qstep1, sigma, powset_set, uq[i][0])
        Fmjnew = []
        mjnew = []
        for ii in range(n):
            check = 0
            Fmjnewlists = []
            mjnewlists = []
            for jj in range(len(Fmj_set[ii])):
                if T.issubset(Fmj_set[ii][jj]):
                    Fmjnewlists.append(Fmj_set[ii][jj])
                    mjnewlists.append(mj[ii][jj])
                    check = 1
            if check == 0:
                i += 1
                break
            else:
                Fmjnew.append(Fmjnewlists)
                mjnew.append(mjnewlists)
        if check:
            intersectBjs = step2fun(Fmjnew, mjnew, Theta_set, n, uq[i][1:])
            # print(sum(intersectBjs))
            if len(sigma) == nt:
                W = 1/len(intersectBjs)
            elif len(sigma) == 2**nt - 1:
                W = 1/(2**len(intersectBjs)-1)
            else:
                W = 1/cardimp2(intersectBjs, powset_set, sigma)
            if intersectBjs.issubset(A_set):
                T1 += W
            else:
                T2 += W
            i += 1
    return T1/(T1+T2)
