####################################################################################
# A script that encapsulates all the allowed functions to derive a best-fit template
# model to the pulse profiles. These are Fourier, von Mises, and Cauchy. The likelihoods
# for each is also written out.
# Normzalied (to unity) version of each of the functions is also spelled out, which
# is necessary when performing an extended maximum likelihood fit to measure a ToA.
# Note that for the latter, a phaseShift is incorporated into the model which
# translates into a ToA. An amplitude shift is also incorporated to take into account
# possible variation in pulsed fraction of magnetars, e.g., during outbursts.
#
# Input for each function:
# 1- xx: independent variable
# 2- theta: dictionary of input parameters
# 
#
# Extra input for their likelihoods:
# 3- yy: dependent variable
# 4- yy_err: uncertainties on dependent variable
#
# To do: add normalized likelihoods for Cauchy and von Mises
#        beautify by making each model its own class
####################################################################################

import sys
import numpy as np
from scipy.special import j0
from scipy import stats

sys.dont_write_bytecode = True


def fourSeries(theta, xx):
    """
    A rudimentary implementation of a Fourier series and its loglikelihood
    Note that this is normalized to a frequency of 1 (cycle folded pulse profile)

    :param theta: model parameters
    :type theta: dict
    :param xx: ppBins
    :type xx: numpy.ndarray
    :return: fourSeriesCurve, Fourier model given theta
    :rtype: numpy.ndarray
    """
    # number of fourier harmonics
    nbrComp = len(np.array([ww for harmKey, ww in theta.items() if harmKey.startswith('amp_')]))

    # Setting total fourier series curve to normalization
    fourSeriesCurve = theta["norm"]

    # Adding the harmonic to the fourier series curve above
    for jj in range(1, nbrComp + 1):
        fourSeriesCurve += theta["amp_" + str(jj)] * theta["ampShift"] * np.cos(
            jj * 2 * np.pi * xx + theta["ph_" + str(jj)] - jj * theta["phShift"])

    return fourSeriesCurve


def logLikelihoodFS(theta, xx, yy, yy_err):
    modelFourSeriesCurve = fourSeries(theta, xx)
    return np.sum(stats.norm.logpdf(yy, loc=modelFourSeriesCurve, scale=yy_err))


def logLikelihoodFSNormalized(theta, xx, exposureInt):  # Normalized log likelihood
    modelFourSeriesCurve = fourSeries(theta, xx)
    # extended maximum likelihood - Dividing Fourier model by norm to normalize it
    modelFourSeriesCurveNormalized = modelFourSeriesCurve / theta["norm"]
    if np.min(modelFourSeriesCurveNormalized) <= 0:  # in case of a 0/negative in the Fourier series estimate - results in undefined
        return -np.inf
    else:
        return (-theta["norm"] * exposureInt + len(xx) * np.log(theta["norm"] * exposureInt) + (
                len(xx) * np.log(len(xx)) - len(xx)) + np.sum(np.log(modelFourSeriesCurveNormalized)))


######################################
######################################
# A rudimentary implementation of a cauchy function and its loglikelihood
# Note that this is fit in the 0-2pi interval
def wrapCauchy(theta, xx):
    # number of Cauchy components
    nbrComp = len(np.array([ww for compKey, ww in theta.items() if compKey.startswith('amp_')]))

    # Setting total cauchy curve to normalization
    wrapCauchyCurve = theta["norm"]

    # Adding the components to the cauchy curve above
    for jj in range(1, nbrComp + 1):  # wrapped Lorentzian
        wrapCauchyCurve += (((theta["amp_" + str(jj)] * theta["ampShift"]) / (2 * np.pi)) *
                            (np.sinh(theta["wid_" + str(jj)]) / (np.cosh(theta["wid_" + str(jj)]) -
                                                                 np.cos(xx - theta["cen_" + str(jj)] - theta["cenShift"]))))

    return wrapCauchyCurve


def logLikelihoodCA(theta, xx, yy, yy_err):
    modelWrapCauchyCurve = wrapCauchy(theta, xx)
    return np.sum(stats.norm.logpdf(yy, loc=modelWrapCauchyCurve, scale=yy_err))


######################################
######################################
# A rudimentary implementation of a vonmises function and its loglikelihood
# Note that this is fit in the 0-2pi interval
def vonmises(theta, xx):
    # number of vonmises components
    nbrComp = len(np.array([ww for compKey, ww in theta.items() if compKey.startswith('amp_')]))

    # Setting total vonmises curve to normalization
    vonmisesCurve = theta["norm"]

    # Adding the components to the vonmises curve above
    for jj in range(1, nbrComp + 1):  # wrapped Gaussian, i.e., von Mises function
        vonmisesCurve += (((theta["amp_" + str(jj)] * theta["ampShift"]) / (
                2 * np.pi * j0(1 / theta["wid_" + str(jj)] ** 2))) *
                          (np.exp((1 / theta["wid_" + str(jj)] ** 2) * (
                              np.cos(xx - theta["cen_" + str(jj)] - theta["cenShift"])))))

    return vonmisesCurve


def logLikelihoodVM(theta, xx, yy, yy_err):
    modelVonmisesCurve = vonmises(theta, xx)
    return np.sum(stats.norm.logpdf(yy, loc=modelVonmisesCurve, scale=yy_err))
