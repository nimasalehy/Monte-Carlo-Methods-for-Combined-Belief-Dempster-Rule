# ==========================================================================================
# ==========================================================================================
# By Nima Salehy.
# Modified on Sep 6, 2021
# Used to produce the numerical results in the following paper:
#       Nima Salehy, Giray Okten "Monte Carlo and quasi-Monte Carlo methods for Dempster's rule of combination"
# ==========================================================================================
# Estimating bel(A) using Algorithm 3 in our paper with \Sigma_A = Singletons of A and
# \Sigma_\Theta = Singletons of \Theta, by MC and RQMC
# Algorithm for calculating bel(A) exactly is based on Wilson 2000 paper
# Also, each mass function is generated randomly as follows:
#    1. Form the powerset of the frame of discernment, and randomly
#       select nfoc (number of focal sets which is fixed) of its elements
#    2. Uniformly generate nfoc numbers between 0 and 1.  Then, normalize
#       them and assign them as the mass values to the focal sets.
# Plotting RMSE's in log-log scale for comparing RQMC and MC
# Random shifted Sobol is used for RQMC
# ==========================================================================================
# ==========================================================================================


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
from itertools import product
from qimp3 import qimp3bel
import time
import copy
import sobol_lib


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
def Bel(Fmj_set, A, masses, PL, Theta):
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


# preparing the probablities to be used in Lines 4 and 14 of Algorithm 3 in our paper
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


# ==========================================================================================

# finding the slopes of the lines fitted to data in log-log scale.
def f(x, A, B):  # this is your 'straight line' y=f(x)
    return A*x + B


# ==========================================================================================

def fun(n, Fmj_set, mj, A_set, N, NN, m):
    print('Number of mass functions (or belief functions) to combine:', n)
    print('Array containing the number of focal sets of each mass function (nfoc[1], ..., nfoc[n]):', nfoc[:n])

    print('Fmj_set (focal sets):', Fmj_set)
    print('mj (corresponding mass values):\n', mj)


    start = time.time()
    # calculating the corresponding plausibilities
    PL = PL_Calc(n, nt, mj, Fmj_set)

    # Calculating bel(A) exactly based on Wilson 2000 paper
    exbel = Bel(Fmj_set, A_set, mj, PL, Theta)
    end = time.time()
    t_exact = end - start
    print('bel(A) is', exbel, 'exactly, calculated in', t_exact, 'seconds!')

    print('==========================================================================')

    # ==========================================================================================
    # ==========================================================================================


    print('Estimating bel(A) using Algorithm 3 in our paper with \Sigma_A = Singletons of A and ')
    print('\Sigma_\Theta = Singletons of \Theta, by MC and RQMC:\n')

    # MC sequences
    start = time.time()
    ui2i3 = [np.zeros((int(NN / 2), n + 1)) for i in range(m)]
    for j in range(m):
        for i in range(int(NN / 2)):
            ui2i3[j][i] = np.random.uniform(0, 1, n + 1)
    end = time.time()
    totaltimeMC = end - start
    print('MC: random sequences generated in', totaltimeMC, 'seconds!')

    # RQMC - Random shifted Sobol
    start = time.time()
    uSi2i3 = [np.zeros((int(NN / 2), n + 1)) for i in range(m)]
    # uSi2i3[0] Sobol sequence
    uSi2i3[0] = np.transpose(sobol_lib.i4_sobol_generate(n + 1, int(NN / 2), 100))
    # random shifted Sobol sequence
    for i in range(1, m):
        ru = np.random.uniform(0, 1, n + 1)
        uSi2i3[i] = (uSi2i3[0] + ru) % 1
    end = time.time()
    totaltimeRQMC = end - start
    print('RQMC: random shifted Sobol sequences generated in', totaltimeRQMC, 'seconds!\n')

    probtime_start = time.time()

    # indices in powersetTheta of the subsets of A which are singletons
    sigmaA = makesigmaA(powersetTheta_set, A_set)
    # q is the list of commonality numbers of the singletons of A;
    # i-th term is an array of commonality numbers corresponding to the i-th mass function
    qA = calq(Fmj_set, mj, sigmaA, powersetTheta_set, n)
    # preparing the probablities to be used in Line 4 of Algorithm 3 in our paper, when \Sigma_A = Singletons of A
    qAstep1, sumqA = qstep1fun(qA)

    # sigma1 contains the indices of the singletons in powersetTheta
    sigma1 = singletons(powersetTheta_set)
    # q is the list of commonality numbers of the singletons of Theta (the frame of discernment);
    # i-th term is an array of commonality numbers corresponding to the i-th mass function
    q = calq(Fmj_set, mj, sigma1, powersetTheta_set, n)
    # preparing the probablities to be used in Line 14 of Algorithm 3 in our paper, when \Sigma_\Theta = Singletons of \Theta
    qstep1, sumq = qstep1fun(q)

    probtime_end = time.time()

    probtime = probtime_end - probtime_start
    print('Time taken to prepare the probabilities used in Lines 4 and 14 of Algorithm 3 in our paper (in seconds):',
          probtime)
    print('==========================================================================')

    RMSE = np.zeros((len(N), 2))
    for indj, j in enumerate(N):
        # time taken for MC and RQMC
        ti3 = []
        tqSi3 = []

        # Estimates by MC and RQMC
        ESTi3 = []
        ESTqSi3 = []

        for i in range(m):
            # print('Sample number', i, 'with size', j)

            # Algorithm 3 in the paper when \Sigma_A = Singletons of A and \Sigma_\Theta = Singletons of \Theta (MC)
            start = time.time()
            i3bel = qimp3bel(j / 2, j / 2, n, Fmj_set, mj, Theta, nt, A_set, sigma1, powersetTheta_set, sigmaA, qstep1,
                             sumq, qAstep1, sumqA, ui2i3[i])
            end = time.time()
            ti3.append(end - start)

            # Algorithm 3 in the paper when \Sigma_A = Singletons of A and \Sigma_\Theta = Singletons of \Theta (RQMC)
            start = time.time()
            qSi3bel = qimp3bel(j / 2, j / 2, n, Fmj_set, mj, Theta, nt, A_set, sigma1, powersetTheta_set, sigmaA,
                               qstep1, sumq, qAstep1, sumqA, uSi2i3[i])
            end = time.time()
            tqSi3.append(end - start)

            ESTi3.append(i3bel)
            ESTqSi3.append(qSi3bel)

        print(m, 'samples each with size', j, 'done!')
        print('\ntotal time taken in seconds (MC):', sum(ti3) + probtime + totaltimeMC * (j / NN))
        print('MC:', m, 'estimates are', ESTi3)
        print('their mean:', np.mean(ESTi3))
        print('\ntotal time taken in seconds (RQMC):', sum(tqSi3) + probtime + totaltimeRQMC * (j / NN))
        print('RQMC:', m, 'estimates are', ESTqSi3)
        print('their mean:', np.mean(ESTqSi3))

        # RMSE (MC):
        RMSE[indj, 0] = RMSEcalculate(ESTi3, exbel)

        # RMSE (RQMC with random shifted Sobol):
        RMSE[indj, 1] = RMSEcalculate(ESTqSi3, exbel)

        print('\nRMSE\'s:\n', RMSE)
        print('==========================================================================')

    # print('RMSE\'s:\n', RMSE)


    # ==========================================================================================
    # ==========================================================================================


    print('==========================================================================')
    A, B = curve_fit(f, np.log10(N), np.log10(RMSE[:, 0]))[0]  # your data x, y to fit
    print('The slope for MC is:', A)
    A, B = curve_fit(f, np.log10(N), np.log10(RMSE[:, 1]))[0]  # your data x, y to fit
    print('The slope for RQMC is:', A)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.loglog(N, RMSE[:, 0], '-*', markersize=12, color='k',
               label=r'$P^{*}_{\mathcal{S}(A)} // P^{*}_{\mathcal{S}(\Theta)}$\textit{ MC}')
    plt.loglog(N, RMSE[:, 1], '-gD', markersize=8, color='k',
               label=r'$P^{*}_{\mathcal{S}(A)} // P^{*}_{\mathcal{S}(\Theta)}$\textit{ RQMC}')

    plt.xlabel(r'\huge$N$')
    plt.ylabel(r'\huge\textit{RMSE}')

    plt.legend(loc=1.0, prop={'size': 20.0})

    plt.xticks(size=15.0)
    plt.yticks(size=15.0)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(12, 7)
    plt.savefig(str(n) + 'masses.pdf')
    print(str(
        n) + 'masses.pdf containing the plots in loglog-scale corresponding to MC and RQMC (RMSE vs N) was created!')
    plt.clf()
    # plt.show()


# ==========================================================================================
# ==========================================================================================


n_masses = [2, 6]  # [min, max] Number of mass functions (or belief functions) to be combined

num_foc = 50  # Number of focal sets of each mass (or belief) function
print('Number of focal sets of each mass (or belief) function:', num_foc)

nt = 10  # Cardinality of \Theta, the frame of discernment
Theta = set()
for i in range(nt):
    Theta.add(i)
print("Theta (Frame of discernment):", Theta)

# An array consists of the number of focal sets of mass functions
# (nfoc[1], ..., nfoc[n])
nfoc = []
for j in range(n_masses[1]):
    nfoc.append(num_foc)

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

# the list [1, ..., 2^nt-1], which is the indices of powersetTheta\{emptyset}, we exclude 0 to have normalized mass functions
numlist = [k for k in range(1, 2 ** nt)]

# Bodies of evidence; more precisely the focal sets for each mass function
Fmj_set = [[] for i in range(n_masses[1])]
# generating Fmj_set's randomly
for j in range(n_masses[1]):
    Fmj_set[j] = random.sample(powersetTheta_set[1:], int(nfoc[j]))
print('Fmj_set (focal sets):', Fmj_set)

# Randomly generate mj (mass values)
mj_unnormalized = np.random.uniform(0,1,(n_masses[1],int(nfoc[0])))
mj = mj_unnormalized/mj_unnormalized.sum(axis=1)[:, np.newaxis]
print('mj (corresponding mass values):\n', mj)


# Randomly generate A, which is a subset of \Theta and our goal is to find bel(A)
A_set = powersetTheta_set[np.random.choice(numlist)]
print('A is', A_set)
print('Goal is to calculate bel(A) exactly, and then estimating it using Algorithm 3 in our paper with ')
print('\Sigma_A = Singletons of A and \Sigma_\Theta = Singletons of \Theta, by MC and RQMC.')


# m estimates, each with sample size N
N = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
NN = N[len(N)-1]  # 128000
m = 100  # Number of samples, each with size N

print('======================================================================================================')
print('======================================================================================================')

for n in range(n_masses[0], n_masses[1] + 1):
    fun(n, Fmj_set[:n], mj[:n], A_set, N, NN, m)
    print('======================================================================================================')
    print('======================================================================================================')
