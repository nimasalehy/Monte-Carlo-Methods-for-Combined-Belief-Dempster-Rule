# Written by Nima Salehy.
# Updated as of 08/29/2021
# This is essentially imp2


def selcom(qstep1, sigma, powset_set, u):
    # u = np.random.uniform(0, 1)
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


def step2fun(Fmjnew, mjnew, Theta, n, u):
    tempBinter = Theta
    for j in range(n):
        tempBinter = tempBinter.intersection(contel(j, Fmjnew, mjnew, u[j]))
    return tempBinter


def cardimp2(intBjs, pset, sig):
    count = 0
    for i in sig:
        if pset[i].issubset(intBjs):
            count += 1
    return count


# Implements Algorithm 3 in our paper to approximate bel(A)
# Number of iterations N, Number of mass functions n, Bodies of evidence Fmj,
# mass values mj, Cardinality of \Theta is nt
# Depending on the type of the random sequences uq, it will apply MC or RQMC
def qimp3bel(N1, N2, n, Fmj_set, mj, Theta, nt, A_set, sigma, powset_set, sigmaA, qstep1, sumq, qAstep1, sumqA, uq):
    TA = 0.
    i = 0
    check = 0
    while i < N1:
        T = selcom(qAstep1, sigmaA, powset_set, uq[i][0])
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
            intersectBjs = step2fun(Fmjnew, mjnew, Theta, n, uq[i][1:])

            if len(sigmaA) == nt:
                W = 1/len(intersectBjs)
            elif len(sigmaA) == 2**nt -1:
                W = 1/(2**len(intersectBjs)-1)
            else:
                W = 1/cardimp2(intersectBjs, powset_set, sigmaA)

            if intersectBjs.issubset(A_set):
                TA += W
            i += 1

    TTH = 0.
    i = 0
    while i < N2:
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
            intersectBjs = step2fun(Fmjnew, mjnew, Theta, n, uq[i][1:])

            if len(sigma) == nt:
                W = 1 / len(intersectBjs)
            elif len(sigma) == 2 ** nt - 1:
                W = 1 / (2 ** len(intersectBjs) - 1)
            else:
                W = 1 / cardimp2(intersectBjs, powset_set, sigma)

            TTH += W
            i += 1
    return (TA/TTH)*(sumqA/sumq)

