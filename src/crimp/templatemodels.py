####################################################################################
# A script that encapsulates all the allowed models to derive a best-fit template
# to the pulse profiles. These are Fourier, von Mises, and Cauchy. Each model has
# its own class. Each class has three methods, (1) to calculate model curve given
# model parameters and an array of phases, (2) a binned likelihood function  with a
# gaussian pdf, and (3) an unbinned extended likelihood function with a poisson pdf
#
# Note that each model also incorporates a phaseShift which translates into a ToA,
# and an amplitude shift that takes into account possible variation in pulsed
# fraction of magnetars, e.g., during outbursts. When not calculating ToAs, these
# two could be set to 0 and 1, respectively.
#
# Input for each class:
# 1- theta: dictionary of input parameters
# 2- xx: independent variable (phases)
####################################################################################

import sys

import numpy
import numpy as np
from scipy.special import i0
from scipy import stats

sys.dont_write_bytecode = True


class Fourier:
    """
        class for Fourier series

        Attributes
        ----------
        theta : dict
            model parameters, keys are **norm**, a pair of **ampShift** and **phShift** (=1, =0, respectively
            unless a ToA is being calculated), and pairs of **amp_n** and **ph_n** for nth harmonic
        xx : numpy.ndarray
                rotational phases of photons

        Methods
        -------
        fourseries():
            calculates the total fourier series curve at each *xx* according to *theta*
        loglikelihoodFS():
            Binned log likelihood of the Fourier curve given *yy* and corresponding uncertainty *yy_err* at each *xx*
            Assumes a gaussian probability density function
        loglikelihoodFSnormalized():
            non-binned extended log likelihood of the normalized Fourier series curve. It requires **exposure** to
            calculate number of counts (occurrences) from rate (i.e., theta["norm"]*exposure)
            Assumes a poisson probability density function
        """

    def __init__(self, theta: dict, xx: numpy.ndarray):
        """
        Constructs all the necessary attributes for the Fourier object

        Parameters
        ----------
            theta : dict
                fourier model parameters, keys are **norm**, a pair of **ampShift** and **phShift** (=1, =0, respectively
                unless a ToA is being calculated), and pairs of **amp_n** and **ph_n** for nth harmonic
            xx : numpy.ndarray
                rotational phases of photons
        """
        self.theta = theta
        self.xx = np.sort(xx)

    def fourseries(self):
        """
        calculates the total fourier series curve at each *xx* according to *theta*
        :return: fourSeriesCurve
        :rtype: numpy.ndarray
        """
        # number of fourier harmonics
        nbrComp = len(np.array([ww for harmKey, ww in self.theta.items() if harmKey.startswith('amp_')]))

        # Setting total fourier series curve to normalization
        fourSeriesCurve = self.theta["norm"]

        # Adding the harmonic to the fourier series curve above
        for jj in range(1, nbrComp + 1):
            fourSeriesCurve += (self.theta["amp_" + str(jj)] * self.theta["ampShift"] *
                                np.cos(jj * 2 * np.pi * self.xx + self.theta["ph_" + str(jj)] - jj * self.theta[
                                    "phShift"]))

        return fourSeriesCurve

    def loglikelihoodFS(self, yy, yy_err):
        """
        Binned log likelihood of the Fourier curve given *yy* and corresponding uncertainty *yy_err* at each *xx*
        Assumes a gaussian probability density function
        :param yy: count rate with same length as xx
        :type yy: numpy.ndarray
        :param yy_err: uncertainty on count rate with same length as xx
        :type yy_err: numpy.ndarray
        :return: log likelihood of yy given theta
        :rtype: numpy.ndarray
        """
        modelFourSeriesCurve = Fourier(self.theta, self.xx).fourseries()
        return np.sum(stats.norm.logpdf(yy, loc=modelFourSeriesCurve, scale=yy_err))

    def loglikelihoodFSnormalized(self, exposure):  # Normalized log likelihood
        """
        non-binned log likelihood of the normalized Fourier series curve. It requires **exposure** to calculate
        number of counts (occurrences) from rate (i.e., theta["norm"]*exposure)
        This is considered the extended likelihood function since it cares about shape and normalization
        Assumes a poisson probability density function
        :param exposure: exposure time required to collect len(xx)
        :type exposure: int
        :return: extended log likelihood of xx given theta
        :rtype: numpy.ndarray
        """
        modelFourSeriesCurve = Fourier(self.theta, self.xx).fourseries()
        # extended maximum likelihood - Dividing Fourier model by norm to normalize it

        modelFourSeriesCurveNormalized = modelFourSeriesCurve / self.theta["norm"]
        if np.min(modelFourSeriesCurveNormalized) <= 0:
            # in case of a 0/negative in the Fourier series estimate - results in undefined
            return -np.inf
        else:
            # return (-self.theta["norm"] * exposure + len(self.xx) * np.log(self.theta["norm"] * exposure) +
            #        (len(self.xx) * np.log(len(self.xx)) - len(self.xx)) +
            #        np.sum(np.log(modelFourSeriesCurveNormalized)))
            return (-self.theta["norm"] * exposure + len(self.xx) * np.log(self.theta["norm"] * exposure) +
                    np.sum(np.log(modelFourSeriesCurveNormalized)))


class WrappedCauchy:
    """
        class for a wrapped Cauchy series

        Attributes
        ----------
        theta : dict
            wrapped Cauchy model parameters, keys are **norm**, a pair of **ampShift** and **cenShift** (=1, =0,
            respectively unless a ToA is being calculated), and triplets of **amp_n**, **wid_n** and **cen_n** for nth
            component
        xx : numpy.ndarray
                rotational phases of photons

        Methods
        -------
        wrapcauchy():
            calculates the total wrapped-Cauchy curve at each *xx* according to *theta*
        loglikelihoodCA():
            Binned log likelihood of the wrapped Cauchy curve given *yy* and corresponding uncertainty *yy_err* at each *xx*
            Assumes a gaussian probability density function
        loglikelihoodCAnormalized():
            non-binned extended log likelihood of the normalized wrapped Cauchy curve. It requires **exposure** to
            calculate number of counts (occurrences) from rate (i.e., theta["norm"]*exposure)
            Assumes a poisson probability density function
        """

    def __init__(self, theta: dict, xx: numpy.ndarray):
        """
        Constructs all the necessary attributes for the WrappedCauchy object

        Parameters
        ----------
            theta : dict
                wrapped Cauchy model parameters, keys are **norm**, a pair of **ampShift** and **cenShift** (=1, =0,
                respectively unless a ToA is being calculated), and triplets of **amp_n**, **wid_n** and **cen_n** for
                nth component
            xx : numpy.ndarray
                rotational phases of photons
        """
        self.theta = theta
        self.xx = np.sort(xx)

    def wrapcauchy(self):
        """
        calculates the total wrapped-Cauchy curve at each *xx* according to *theta*
        :return: wrapCauchyCurve
        :rtype: numpy.ndarray
        """
        # number of Cauchy components
        nbrComp = len(np.array([ww for compKey, ww in self.theta.items() if compKey.startswith('amp_')]))

        # Setting total cauchy curve to normalization
        wrapCauchyCurve = self.theta["norm"]

        # Adding the components to the cauchy curve above
        for jj in range(1, nbrComp + 1):  # wrapped Lorentzian
            wrapCauchyCurve += (((self.theta["amp_" + str(jj)] * self.theta["ampShift"]) / (2 * np.pi)) *
                                (np.sinh(self.theta["wid_" + str(jj)]) /
                                 (np.cosh(self.theta["wid_" + str(jj)]) -
                                  np.cos(self.xx - self.theta["cen_" + str(jj)] - self.theta["phShift"]))))

        return wrapCauchyCurve

    def loglikelihoodCA(self, yy, yy_err):
        """
        Binned log likelihood of the wrapped Cauchy curve given *yy* and corresponding uncertainty *yy_err* at each *xx*
        Assumes a gaussian probability density function
        :param yy: count rate with same length as xx
        :type yy: numpy.ndarray
        :param yy_err: uncertainty on count rate with same length as xx
        :type yy_err: numpy.ndarray
        :return: log likelihood of yy given theta
        :rtype: numpy.ndarray
        """
        modelWrapCauchyCurve = WrappedCauchy(self.theta, self.xx).wrapcauchy()
        return np.sum(stats.norm.logpdf(yy, loc=modelWrapCauchyCurve, scale=yy_err))

    def loglikelihoodCAnormalized(self, exposure):
        """
        non-binned log likelihood of the normalized wrapped-cauchy curve. It requires **exposure** to calculate
        number of counts (occurrences) from rate (i.e., theta["norm"]*exposure)
        This is considered the extended likelihood function since it cares about shape and normalization
        Assumes a poisson probability density function
        :param exposure: exposure time required to collect len(xx)
        :type exposure: int
        :return: extended log likelihood of xx given theta
        :rtype: numpy.ndarray
        """
        modelWrapCauchyCurve = WrappedCauchy(self.theta, self.xx).wrapcauchy()
        # extended maximum likelihood - normalizing Cauchy model
        nbrComp = len(np.array([ww for compKey, ww in self.theta.items() if compKey.startswith('amp_')]))
        normalizingFactor = 2 * np.pi * self.theta["norm"]
        for jj in range(1, nbrComp + 1):
            normalizingFactor += self.theta["amp_" + str(jj)]
        modelWrapCauchyCurveNormalized = modelWrapCauchyCurve / normalizingFactor

        if np.min(modelWrapCauchyCurveNormalized) <= 0:
            # in case of a 0/negative in the Cauchy series estimate - results in undefined
            return -np.inf
        else:
            return (-np.mean(modelWrapCauchyCurve) * exposure + len(self.xx) *
                    np.log(np.mean(modelWrapCauchyCurve) * exposure) + np.sum(np.log(modelWrapCauchyCurveNormalized)))


class VonMises:
    """
        class for a von Mises series

        Attributes
        ----------
        theta : dict
            von-Mises model parameters, keys are **norm**, a pair of **ampShift** and **cenShift** (=1, =0,
            respectively unless a ToA is being calculated), and triplets of **amp_n**, **wid_n** and **cen_n** for nth
            component
        xx : numpy.ndarray
                rotational phases of photons

        Methods
        -------
        vonmises():
            calculates the total von-Mises curve at each *xx* according to *theta*
        loglikelihoodVM():
            Binned log likelihood of the von-Mises curve given *yy* and corresponding uncertainty *yy_err* at each *xx*
            Assumes a gaussian probability density function
        loglikelihoodVMnormalized():
            non-binned extended log likelihood of the normalized von-Mises curve. It requires **exposure** to
            calculate number of counts (occurrences) from rate (i.e., theta["norm"]*exposure)
            Assumes a poisson probability density function
        """

    def __init__(self, theta: dict, xx: numpy.ndarray):
        """
        Constructs all the necessary attributes for the VonMises object

        Parameters
        ----------
            theta : dict
                von-Mises model parameters, keys are **norm**, a pair of **ampShift** and **cenShift** (=1, =0,
                respectively unless a ToA is being calculated), and triplets of **amp_n**, **wid_n** and **cen_n** for nth
                component
            xx : numpy.ndarray
                rotational phases of photons
        """
        self.theta = theta
        self.xx = np.sort(xx)

    def vonmises(self):
        """
        calculates the total von-Mises curve at each *xx* according to *theta*
        :return: vonmisesCurve
        :rtype: numpy.ndarray
        """
        # number of vonmises components
        nbrComp = len(np.array([ww for compKey, ww in self.theta.items() if compKey.startswith('amp_')]))

        # Setting total vonmises curve to normalization
        vonmisesCurve = self.theta["norm"]

        # Adding the components to the vonmises curve above
        for jj in range(1, nbrComp + 1):  # wrapped Gaussian, i.e., von Mises function
            vonmisesCurve += (((self.theta["amp_" + str(jj)] * self.theta["ampShift"]) /
                               (2 * np.pi * i0(1 / self.theta["wid_" + str(jj)] ** 2))) *
                              (np.exp((1 / self.theta["wid_" + str(jj)] ** 2) *
                                      (np.cos(self.xx - self.theta["cen_" + str(jj)] - self.theta["phShift"])))))

        return vonmisesCurve

    def loglikelihoodVM(self, yy, yy_err):
        """
        Binned log likelihood of the von-Mises curve given *yy* and corresponding uncertainty *yy_err* at each *xx*
        Assumes a gaussian probability density function
        :param yy: count rate with same length as xx
        :type yy: numpy.ndarray
        :param yy_err: uncertainty on count rate with same length as xx
        :type yy_err: numpy.ndarray
        :return: log likelihood of yy given theta
        :rtype: numpy.ndarray
        """
        modelVonmisesCurve = VonMises(self.theta, self.xx).vonmises()
        return np.sum(stats.norm.logpdf(yy, loc=modelVonmisesCurve, scale=yy_err))

    def loglikelihoodVMnormalized(self, exposure):
        """
        non-binned log likelihood of the normalized von-Mises curve. It requires **exposure** to calculate
        number of counts (occurrences) from rate (i.e., theta["norm"]*exposure)
        This is considered the extended likelihood function since it cares about shape and normalization
        Assumes a poisson probability density function
        :param exposure: exposure time required to collect len(xx)
        :type exposure: int
        :return: extended log likelihood of xx given theta
        :rtype: numpy.ndarray
        """
        modelVonMisesCurve = VonMises(self.theta, self.xx).vonmises()
        # extended maximum likelihood - normalizing von-Mises model
        nbrComp = len(np.array([ww for compKey, ww in self.theta.items() if compKey.startswith('amp_')]))
        normalizingFactor = 2 * np.pi * self.theta["norm"]
        for jj in range(1, nbrComp + 1):
            normalizingFactor += self.theta["amp_" + str(jj)]
        modelVonMisesCurveNormalized = modelVonMisesCurve / normalizingFactor
        if np.min(modelVonMisesCurveNormalized) <= 0:
            # in case of a 0/negative in the Fourier series estimate - results in undefined
            return -np.inf
        else:
            return (-np.mean(modelVonMisesCurve) * exposure + len(self.xx) *
                    np.log(np.mean(modelVonMisesCurve) * exposure) + np.sum(np.log(modelVonMisesCurveNormalized)))
