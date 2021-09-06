# Monte-Carlo-Methods-for-Combined-Belief-Dempster-Rule


MainPart1new.py:
# ================================
# By Nima Salehy.
# Modified on Sep 6, 2021
# Used to produce the numerical results in the following paper:
#       Nima Salehy, Giray Okten "Monte Carlo and quasi-Monte Carlo methods for Dempster's rule of combination"
# ================================
# Estimating bel(A) using different algorithms by MC, and compare their RMSEs and running times
# Algorithm for calculating bel(A) exactly is based on Wilson 2000 paper
# Also, each mass function is generated randomly as follows:
#    1. Form the powerset of the frame of discernment, and randomly select nfoc (number of focal sets which is fixed) of its elements
#    2. Uniformly generate nfoc numbers between 0 and 1.  Then, normalize them and assign them as the mass values to the focal sets.
# ================================


MainPart2new.py:
# ================================
# By Nima Salehy.
# Modified on Sep 6, 2021
# Used to produce the numerical results in the following paper:
#       Nima Salehy, Giray Okten "Monte Carlo and quasi-Monte Carlo methods for Dempster's rule of combination"
# ================================
# Estimating bel(A) using Algorithm 3 in our paper with \Sigma_A = Singletons of A and \Sigma_\Theta = Singletons of \Theta, by MC and RQMC
# Algorithm for calculating bel(A) exactly is based on Wilson 2000 paper
# Also, each mass function is generated randomly as follows:
#    1. Form the powerset of the frame of discernment, and randomly select nfoc (number of focal sets which is fixed) of its elements
#    2. Uniformly generate nfoc numbers between 0 and 1.  Then, normalize them and assign them as the mass values to the focal sets.
# Plotting RMSE's in log-log scale for comparing RQMC and MC
# Random shifted Sobol is used for RQMC
# ================================
