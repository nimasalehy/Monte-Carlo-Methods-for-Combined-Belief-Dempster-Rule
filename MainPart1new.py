# ==========================================================================================
# ==========================================================================================
# By Nima Salehy.
# Modified on Nov 26, 2021
# Used to produce the numerical results in the following paper:
#       Nima Salehy, Giray Okten "Monte Carlo and quasi-Monte Carlo methods for Dempster's rule of combination"
# ==========================================================================================
# Estimating bel(A) using different algorithms by MC, and compare their Relative RMSEs and running times
# Algorithm for calculating bel(A) exactly is based on Wilson 2000 paper
# Also, each mass function is generated randomly as follows:
#    1. Form the powerset of the frame of discernment, and randomly
#       select nfoc (number of focal sets which is fixed) of its elements
#    2. Uniformly generate nfoc numbers between 0 and 1.  Then, normalize
#       them and assign them as the mass values to the focal sets.
# ==========================================================================================
# ==========================================================================================


import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
import random
from itertools import product
from qsimple import qsimplebel
from qimp1 import qimp1bel
from qimp2 import qimp2bel
from qimp3 import qimp3bel
import time
import copy


# ==========================================================================================
# ==========================================================================================

# calculating the corresponding plausibilities
def PL_Calc(n, nt, masses, Fmj_set):
    PL = []
    for i in range(n):
        PLi = []
        for j in range(nt):
            PLTemp = 0.
            for jj in range(len(Fmj_set[i])):
                if j in Fmj_set[i][jj]:
                    PLTemp += masses[i][jj]
            PLi.append(PLTemp)
        PL.append(PLi)
    return PL


def Rec_term_or_R_Calc(BB, sigma_theta_copy, Fmj_set, masses, PL):
    # part(b)
    if BB == set():
        return 1.
    # part(c)
    elif len(BB) == 1:
        tempR = 1.
        for theta in BB:
            for i in sigma_theta_copy[theta]:
                tempR = tempR * PL[i][theta]
        return 1-tempR
    else:
        for theta in BB:
            # part (a)
            if sigma_theta_copy[theta] == set():
                return 0.0

    # Finding l that minimizes \sum |B\intersect \GammaSet[l]|
    card_list = [sum([len(BB.intersection(elmnt)) for elmnt in Fmj_set[i]]) for i in range(len(Fmj_set))]
    indices_sorted_card_list = np.argsort(card_list)
    l = -1
    for inde in range(len(indices_sorted_card_list)):
        for i in range(len(sigma_theta_copy)):
            if indices_sorted_card_list[inde] in sigma_theta_copy[i]:
                l = indices_sorted_card_list[inde]
                break
        if l != -1:
            break

    for i in range(len(sigma_theta_copy)):
        if l in sigma_theta_copy[i]:
            sigma_theta_copy[i].remove(l)

    term = 0.
    for i in range(len(Fmj_set[l])):
        sigma_theta_copy_copy = copy.deepcopy(sigma_theta_copy)
        term += masses[l][i] * Rec_term_or_R_Calc(BB.intersection(Fmj_set[l][i]), sigma_theta_copy_copy, Fmj_set, masses, PL)
    return term


# Calculating bel(A) exactly based on Wilson 2000 paper
def Bel(n, Fmj_set, A, masses, PL, Theta):
    sigma_theta_one = set()
    for i in range(n):
        sigma_theta_one.add(i)
    sigma_theta = [sigma_theta_one for i in range(nt)]

    # B is A complement
    B = Theta.difference(A)

    sigma_theta_copy1 = copy.deepcopy(sigma_theta)
    sigma_theta_copy2 = copy.deepcopy(sigma_theta)

    term1 = Rec_term_or_R_Calc(B, sigma_theta_copy1, Fmj_set, masses, PL)
    term2 = Rec_term_or_R_Calc(Theta, sigma_theta_copy2, Fmj_set, masses, PL)

    print('\nterm1:', term1)
    print('term2:', term2)
    print('\nconflict:', term2)
    exact_bel = (term1 - term2) / (1. - term2)
    return exact_bel


# ==========================================================================================


# indices of the singletons in powersetTheta
def singletons(powset_set):
    indices = []
    for ind, i in enumerate(powset_set):
        if len(i) == 1:
            indices.append(ind)
    return indices


# Returns indices in powersetTheta of the subsets of A which are singletons
def makesigmaA(powersetTheta_set, A_set):
    ss = []
    for ind, i in enumerate(powersetTheta_set[1:]):
        if len(i) == 1:
            if i.issubset(A_set):
                ss.append(ind + 1)
    return ss


# Calculates RMSE
def RMSEcalculate(VecOfEst, truemean):
    rmsetemp = 0.
    for i in range(len(VecOfEst)):
        rmsetemp += (VecOfEst[i] - truemean)**2
    return np.sqrt(rmsetemp/len(VecOfEst))


def calqj(Fmj_set, mj, j, sigma, powset_set):
    tempqj = np.zeros(len(sigma))
    for i in range(len(sigma)):
        for k in range(len(mj[j])):
            if powset_set[sigma[i]].issubset(Fmj_set[j][k]):
                tempqj[i] += mj[j][k]
    return tempqj


def calq(Fmj_set, mj, sigma, powset_set, n):
    # tempq (list of commonalities; i-th term is an array of commonality numbers corresponding to the i-th mass function)
    tempq = [np.zeros(len(sigma)) for j in range(n)]
    for i in range(n):
        tempq[i] = calqj(Fmj_set, mj, i, sigma, powset_set)

    return tempq


# preparing the probablities to be used in Lines 4 of Algorithm 2 and in Lines 4 and 14 of Algorithm 3 in our paper
def qstep1fun(q):
    qstep1list = []
    for j in range(len(q[0])):  # len(sigma)
        temp = 1.
        for i in range(len(q)):  # n
            temp *= q[i][j]
        qstep1list.append(temp)
    sumtemp = sum(qstep1list)
    for i in range(len(qstep1list)):
        qstep1list[i] = qstep1list[i]/sumtemp
    return qstep1list, sumtemp


# Returns indices in powersetTheta of all the subsets of A except the empty set
def makesigmaAnew(powersetTheta_set, A_set):
    ss = []
    for ind, i in enumerate(powersetTheta_set[1:]):
        if i.issubset(A_set):
            ss.append(ind + 1)
    return ss


# ==========================================================================================
# ==========================================================================================


def fun(n, Fmj_set, mj, A_set, N, NN, m, usi1, tusi1, ui2i3, tui2i3):
    print('Number of mass functions (or belief functions) to combine:', n)
    print('Array containing the number of focal sets of each mass function (nfoc[1], ..., nfoc[n]):', nfoc[:n])

    print('Fmj_set (focal sets):', Fmj_set)
    print('mj (corresponding mass values):\n', mj)

    start = time.time()
    # calculating the corresponding plausibilities
    PL = PL_Calc(n, nt, mj, Fmj_set)

    # Calculating bel(A) exactly based on Wilson 2000 paper
    exbel = Bel(n, Fmj_set, A_set, mj, PL, Theta)
    end = time.time()
    t_exact = end - start
    print('bel(A) is', exbel, 'exactly, calculated in', t_exact, 'seconds!')
    print('==========================================================================')

    # ==========================================================================================
    # ==========================================================================================


    print('***Estimating bel(A) using different algorithms***\n')


    print('Preparing the probablities to be used in Line 4 of Algorithm 2 or in Lines 4 and 14 of Algorithm 3 in our paper\n')

    print('Preparing the probablities to be used in Line 4 of Algorithm 3 in our paper, when \Sigma_A = Singletons of A')
    probtime_sigmaA_start = time.time()
    # indices in powersetTheta of the subsets of A which are singletons
    sigmaA = makesigmaA(powersetTheta_set, A_set)
    # q is the list of commonality numbers of the singletons of A;
    # i-th term is an array of commonality numbers corresponding to the i-th mass function
    qA = calq(Fmj_set, mj, sigmaA, powersetTheta_set, n)
    # Preparing the probablities to be used in Line 4 of Algorithm 3 in our paper, when \Sigma_A = Singletons of A
    qAstep1, sumqA = qstep1fun(qA)
    probtime_sigmaA_end = time.time()
    probtime_sigmaA = probtime_sigmaA_end - probtime_sigmaA_start
    print('Time taken:', probtime_sigmaA)

    print('\nPreparing the probablities to be used in Line 14 of Algorithm 3 or Line 4 of Algorithm 2 when \Sigma_\Theta = Singletons of \Theta')
    probtime_sigma1_start = time.time()
    # sigma1 contains the indices of the singletons in powersetTheta
    sigma1 = singletons(powersetTheta_set)
    # q is the list of commonality numbers of the singletons of Theta (the frame of discernment);
    # i-th term is an array of commonality numbers corresponding to the i-th mass function
    q = calq(Fmj_set, mj, sigma1, powersetTheta_set, n)
    # Preparing the probablities to be used in Line 14 of Algorithm 3 or Line 4 of Algorithm 2
    # when \Sigma_\Theta = Singletons of \Theta
    qstep1, sumq = qstep1fun(q)
    probtime_sigma1_end = time.time()
    probtime_sigma1 = probtime_sigma1_end - probtime_sigma1_start
    print('Time taken:', probtime_sigma1)

    print('\nPreparing the probablities to be used in Line 4 of Algorithm 3 in our paper, when \Sigma_A = \Omega(A)')
    probtime_sigmaAnew_start = time.time()
    # indices in powersetTheta of all the subsets of A (except for the emptyset)
    sigmaAnew = makesigmaAnew(powersetTheta_set, A_set)
    # q is the list of commonality numbers of all the subsets of A (except for the emptyset);
    # i-th term is an array of commonality numbers corresponding to the i-th mass function
    qAnew = calq(Fmj_set, mj, sigmaAnew, powersetTheta_set, n)
    # Preparing the probablities to be used in Line 4 of Algorithm 3 in our paper, when \Sigma_A = \Omega(A)
    qAnewstep1, sumqAnew = qstep1fun(qAnew)
    probtime_sigmaAnew_end = time.time()
    probtime_sigmaAnew = probtime_sigmaAnew_end - probtime_sigmaAnew_start
    print('Time taken:', probtime_sigmaAnew)

    print('\nPreparing the probablities to be used in Line 14 of Algorithm 3 or Line 4 of Algorithm 2 when \Sigma_\Theta = \Omega(\Theta)')
    probtime_sigma2_start = time.time()
    # sigma2 is the list [1, ..., 2^nt-1], which is the indices of powersetTheta\{emptyset}
    sigma2 = numlist
    # q is the list of commonality numbers of all the subsets of Theta (the frame of discernment) except for the emptyset;
    # i-th term is an array of commonality numbers corresponding to the i-th mass function
    q2 = calq(Fmj_set, mj, sigma2, powersetTheta_set, n)
    # preparing the probablities to be used in Line 14 of Algorithm 3 or Line 4 of Algorithm 2
    # when \Sigma_\Theta = \Omega(\Theta)
    q2step1, sumq2 = qstep1fun(q2)
    probtime_sigma2_end = time.time()
    probtime_sigma2 = probtime_sigma2_end - probtime_sigma2_start
    print('Time taken:', probtime_sigma2)

    print('==========================================================================')

    t = np.zeros((len(N), 8))
    RRMSE = np.zeros((len(N), 8))
    EST = np.zeros((len(N), 8))
    for indj, j in enumerate(N):
        # time taken for different algorithms to estimate bel(A)
        ts = []
        ti1 = []
        ti2 = []
        ti2new = []

        ti3 = []
        ti3new = []
        ti32 = []
        ti3new2 = []

        # -------------------------------

        # Estimates by different algorithms estimating bel(A)
        ESTs = []
        ESTi1 = []
        ESTi2 = []
        ESTi2new = []

        ESTi3 = []
        ESTi3new = []
        ESTi32 = []
        ESTi3new2 = []

        for i in range(m):
            # Simple MC
            start = time.time()
            sbel = qsimplebel(j, n, Fmj_set, mj, Theta, A_set, usi1[i])
            end = time.time()
            ts.append(end - start)

            # Imp1 MC in Moral and Wilson 1996
            start = time.time()
            i1bel = qimp1bel(j, n, Fmj_set, mj, Theta, A_set, usi1[i])
            end = time.time()
            ti1.append(end - start)

            # Alg 2, when \Sigma_\Theta = Singletons of \Theta
            start = time.time()
            i2bel = qimp2bel(j, n, Fmj_set, mj, Theta, nt, A_set, sigma1, powersetTheta_set, qstep1, ui2i3[i])
            end = time.time()
            ti2.append(end - start)

            # Alg 2, when \Sigma_\Theta = \Omega(\Theta)
            start = time.time()
            i2newbel = qimp2bel(j, n, Fmj_set, mj, Theta, nt, A_set, sigma2, powersetTheta_set, q2step1, ui2i3[i])
            end = time.time()
            ti2new.append(end - start)

            # Alg 3, when \Sigma_A = Singletons of A and \Sigma_\Theta = Singletons of \Theta
            start = time.time()
            i3bel = qimp3bel(j / 2, j / 2, n, Fmj_set, mj, Theta, nt, A_set, sigma1, powersetTheta_set, sigmaA, qstep1,
                             sumq, qAstep1, sumqA, ui2i3[i])
            end = time.time()
            ti3.append(end - start)

            # Alg 3, when \Sigma_A = Singletons of A and \Sigma_\Theta = \Omega(\Theta)
            start = time.time()
            i3newbel = qimp3bel(j / 2, j / 2, n, Fmj_set, mj, Theta, nt, A_set, sigma2, powersetTheta_set, sigmaA,
                                q2step1, sumq2, qAstep1, sumqA, ui2i3[i])
            end = time.time()
            ti3new.append(end - start)

            # Alg 3, when \Sigma_A = \Omega(A) and \Sigma_\Theta = Singletons of \Theta
            start = time.time()
            i3bel2 = qimp3bel(j / 2, j / 2, n, Fmj_set, mj, Theta, nt, A_set, sigma1, powersetTheta_set, sigmaAnew,
                              qstep1, sumq, qAnewstep1, sumqAnew, ui2i3[i])
            end = time.time()
            ti32.append(end - start)

            # Alg 3, when \Sigma_A = \Omega(A) and \Sigma_\Theta = \Omega(\Theta)
            start = time.time()
            i3newbel2 = qimp3bel(j / 2, j / 2, n, Fmj_set, mj, Theta, nt, A_set, sigma2, powersetTheta_set, sigmaAnew,
                                 q2step1, sumq2, qAnewstep1, sumqAnew, ui2i3[i])
            end = time.time()
            ti3new2.append(end - start)

            ESTs.append(sbel)
            ESTi1.append(i1bel)
            ESTi2.append(i2bel)
            ESTi2new.append(i2newbel)

            ESTi3.append(i3bel)
            ESTi3new.append(i3newbel)
            ESTi32.append(i3bel2)
            ESTi3new2.append(i3newbel2)

        print(m, 'estimates, each with sample size', j, ', done for all algorithms!')
        t[indj, 0] = sum(ts) + tusi1 * (j / NN)
        t[indj, 1] = sum(ti1) + tusi1 * (j / NN)
        t[indj, 2] = sum(ti2) + tui2i3 * (j / NN) + probtime_sigma1
        t[indj, 3] = sum(ti2new) + tui2i3 * (j / (NN)) + probtime_sigma2
        t[indj, 4] = sum(ti3) + tui2i3 * (j / (2 * NN)) + probtime_sigmaA + probtime_sigma1
        t[indj, 5] = sum(ti3new) + tui2i3 * (j / (2 * NN)) + probtime_sigmaA + probtime_sigma2
        t[indj, 6] = sum(ti32) + tui2i3 * (j / (2 * NN)) + probtime_sigmaAnew + probtime_sigma1
        t[indj, 7] = sum(ti3new2) + tui2i3 * (j / (2 * NN)) + probtime_sigmaAnew + probtime_sigma2

        print('***Estimates in order***\n')

        print(m, 'estimates of the 1-st alg (simple MC):')
        print(ESTs)
        print('their mean is:', np.mean(ESTs))

        print(m, 'estimates of the 2-nd alg (imp1):')
        print(ESTi1)
        print('their mean is:', np.mean(ESTi1), '\n')

        print(m, 'estimates of the 3-rd alg (Alg 2, \Sigma_\Theta = Singletons of \Theta):')
        print(ESTi2)
        print('their mean is:', np.mean(ESTi2), '\n')

        print(m, 'estimates of the 4-th alg (Alg 2, \Sigma_\Theta = \Omega(\Theta)):')
        print(ESTi2new)
        print('their mean is:', np.mean(ESTi2new), '\n')

        print(m,
              'estimates of the 5-th alg (Alg 3, \Sigma_A = Singletons of A and \Sigma_\Theta = Singletons of \Theta):')
        print(ESTi3)
        print('their mean is:', np.mean(ESTi3), '\n')

        print(m, 'estimates of the 6-th alg (Alg 3, \Sigma_A = Singletons of A and \Sigma_\Theta = \Omega(\Theta)):')
        print(ESTi3new)
        print('their mean is:', np.mean(ESTi3new), '\n')

        print(m, 'estimates of the 7-th alg (Alg 3, \Sigma_A = \Omega(A) and \Sigma_\Theta = Singletons of \Theta):')
        print(ESTi32)
        print('their mean is:', np.mean(ESTi32), '\n')

        print(m, 'estimates of the 8-th alg (Alg 3, \Sigma_A = \Omega(A) and \Sigma_\Theta = \Omega(\Theta)):')
        print(ESTi3new2)
        print('their mean is:', np.mean(ESTi3new2), '\n')

        # print('-------------------')
        EST[indj, 0] = np.mean(ESTs)
        EST[indj, 1] = np.mean(ESTi1)
        EST[indj, 2] = np.mean(ESTi2)
        EST[indj, 3] = np.mean(ESTi2new)

        EST[indj, 4] = np.mean(ESTi3)
        EST[indj, 5] = np.mean(ESTi3new)
        EST[indj, 6] = np.mean(ESTi32)
        EST[indj, 7] = np.mean(ESTi3new2)

        # RMSEs
        RRMSE[indj, 0] = RMSEcalculate(ESTs, exbel)/exbel
        RRMSE[indj, 1] = RMSEcalculate(ESTi1, exbel)/exbel
        RRMSE[indj, 2] = RMSEcalculate(ESTi2, exbel)/exbel
        RRMSE[indj, 3] = RMSEcalculate(ESTi2new, exbel)/exbel

        RRMSE[indj, 4] = RMSEcalculate(ESTi3, exbel)/exbel
        RRMSE[indj, 5] = RMSEcalculate(ESTi3new, exbel)/exbel
        RRMSE[indj, 6] = RMSEcalculate(ESTi32, exbel)/exbel
        RRMSE[indj, 7] = RMSEcalculate(ESTi3new2, exbel)/exbel

        print('time taken:\n', t)
        print('\nRelative RMSE\'s:\n', RRMSE)
        print('\nEstimates:\n', EST)
        print('==========================================================================')


# ==========================================================================================
# ==========================================================================================


n_masses = [15, 16]  # [min, max] Number of mass functions (or belief functions) to be combined

num_foc = 10  # Number of focal sets of each mass (or belief) function
print('Number of focal sets of each mass (or belief) function:', num_foc)

nt = 10  # Cardinality of \Theta, the frame of discernment
Theta = set()
for i in range(nt):
    Theta.add(i)
print("Theta (Frame of discernment):", Theta)

# Power set of \Theta
powersetTheta = list(product(range(2), repeat=nt))

# Power set of \Theta in set mode
powersetTheta_set = []
for vec in powersetTheta:
    setset = set()
    for ind, val in enumerate(vec):
        if val:
            setset.add(ind)
    powersetTheta_set.append(setset)

# the list [1, ..., 2^nt-1], which is the indices of powersetTheta\{emptyset}, we
# exclude 0 to have normalized mass functions
numlist = [k for k in range(1, 2 ** nt)]


# An array consists of the number of focal sets of mass functions
# (nfoc[1], ..., nfoc[n])
nfoc = []
for j in range(n_masses[1]):
    nfoc.append(num_foc)


# Bodies of evidence; more precisely the focal sets for each mass function
Fmj_set = [[] for i in range(n_masses[1])]
# generating Fmj_set's randomly
for j in range(n_masses[1]):
    Fmj_set[j] = random.sample(powersetTheta_set[1:], int(nfoc[j]))


# Randomly generate mj (mass values)
mj_unnormalized = np.random.uniform(0,1,(n_masses[1],int(nfoc[0])))
mj = mj_unnormalized/mj_unnormalized.sum(axis=1)[:, np.newaxis]



# Randomly generate A, which is a subset of \Theta and our goal is to find bel(A)
A_set = powersetTheta_set[np.random.choice(numlist)]
print('A is', A_set)
print('Goal is to calculate bel(A) exactly, and then estimating it using different MC algorithms.')


# ==========================================================================================
# ==========================================================================================

# ***Estimating bel(A) using different algorithms***

# m estimates, each with sample size N
N = [20000, 40000]  # [20000, 40000]
NN = N[len(N) - 1]  # 128000
m = 10  # 10  # Number of samples, each with size N

# Generating the needed MC sequences used in different algorithms
start = time.time()
usi1 = [np.zeros((NN, n_masses[1])) for i in range(m)]
# usi1[0] is sequence used in simple and imp1
for j in range(m):
    for i in range(NN):
        usi1[j][i] = np.random.uniform(0, 1, n_masses[1])
end = time.time()
tusi1 = end - start

start = time.time()
ui2i3 = [np.zeros((NN, n_masses[1] + 1)) for i in range(m)]
# ui2i3[0] sequence used in imp2 and imp3
for j in range(m):
    for i in range(NN):
        ui2i3[j][i] = np.random.uniform(0, 1, n_masses[1] + 1)
end = time.time()
tui2i3 = end - start

print('MC random sequences generated!\n')
print('======================================================================================================')
print('======================================================================================================')

for n in range(n_masses[0], n_masses[1] + 1):
    fun(n, Fmj_set[:n], mj[:n], A_set, N, NN, m, usi1, tusi1*(n/n_masses[1]), ui2i3, tui2i3*(n/n_masses[1]))
    print('======================================================================================================')
    print('======================================================================================================')
