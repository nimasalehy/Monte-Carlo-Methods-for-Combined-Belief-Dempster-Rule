
# = C_{j}
def mass(Y, j, Fmj_set, mj):
    templist = []
    temp = 0.
    for i in range(len(mj[j])):
        if Y.intersection(Fmj_set[j][i]):  # sum(intersect(Y, Fmj[j][i]))
            templist.append(i)
            temp += mj[j][i]
    return [temp, templist]


def selectco(Cj, Cjlist, j, Fmj_set, mj, u):
    temp = mj[j][Cjlist[0]]/Cj
    k = 0
    while u >= temp:
        k += 1
        temp += mj[j][Cjlist[k]]/Cj
    return Fmj_set[j][Cjlist[k]]


# Number of iterations N, Number of mass functions n, Bodies of evidence Fmj,
# masses mj, Cardinality of \Theta is nt
def qimp1bel(N, n, Fmj_set, mj, Theta, A_set, uq):
    T1 = 0.
    T2 = 0.
    i = 0
    while i < N:
        Y = Theta
        W = 1.
        for j in range(n):
            [Cj, Cjlist] = mass(Y, j, Fmj_set, mj)
            W *= Cj
            if Cj == 0.:
                i += 1
                break
            Y = Y.intersection(selectco(Cj, Cjlist, j, Fmj_set, mj, uq[i][j]))   # intersect(Y, B[j])
        if Cj:
            i += 1

            if Y.issubset(A_set):
                T1 += W
            else:
                T2 += W
    return T1/(T1+T2)
