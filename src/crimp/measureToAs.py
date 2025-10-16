"""
This code calculates ToAs in a similar fashion to Ray et al. 2018, but with
some small differences. It takes a (barycenter-corrected) event file, a
timing model (.par file), a template model (.txt file, e.g., from pulseprofile.py),
and a .txt file defining the ToAs (e.g., ToAs start and end times - could be
built with buildtimeintervalsToAs.py). The user could apply an energy filtering
to the event file through eneLow and eneHigh. A subset of ToAs could be measured
if desired through 'toaStart' and/or 'toaEnd'. Most of the calculations are done
in the template-appropriate functions, e.g., "measToA_fourier". The appropriate
TIME array is folded using the .par file, and fit with the provided template model.
For this step, we use the extended MLE method, all model parameters are fixed,
except for the normalization and a phase-shift. The uncertainties on the phase-shift
are measured by stepping away from the best fit phase-shift in steps of 2pi/phShiftRes
(default=1000) and calculating the corresponding maximum likelihood; +/-1 sigma
uncertainty is when the latter drops by +/-0.5 (given that our Likelihood follows
a chi2 distribution with 1 dof). For debugging/testing/analysis purposes, the
user may choose to plot the pulse profile (and specify the nbrBins in the plot)
and/or the log-likelihood around the best fit phase-shift (which should be quite
symmetric). The argument "brutemin" will use the global minimization BRUTE method
(check scipy or lmfit for details) to home in on the global minimum. This is useful
in the case of, e.g., double-peaked profiles with similar amplitudes and shape
(i.e., subtle differences). A global search ensures that you are not getting stuck
in a local minimum (e.g., on the wrong peak).

Can be run via command line as "measuretoas"

To do:
This module is a bit messy (to say the least :)). It can benefit from some cleaning, e.g.,
- Create a class with three methods which measure ToAs for each template (fourier, cauchy, vonmises)
- Eliminate repetitive code in each method (e.g. ToA error calculation could be its own small function)
- I keep track of the number of free parameters manually; could be directly read from lmfit output
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd
import logging

from scipy.stats import chi2

from lmfit import Parameters, minimize

# Custom modules
from crimp.eventfile import EvtFileOps
from crimp.calcphase import calcphase
from crimp.binphases import binphases
from crimp.readPPtemplate import readPPtemplate
from crimp.templatemodels import Fourier, WrappedCauchy, VonMises
from crimp.ephemTmjd import ephemTmjd
from crimp.periodsearch import PeriodSearch
from crimp.timfile import phshiftTotimfile

sys.dont_write_bytecode = True

# Log config
############
logFormatter = logging.Formatter('[%(asctime)s] %(levelname)8s %(message)s ' +
                                 '(%(filename)s:%(lineno)s)', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('crimp_log')
logger.setLevel(logging.DEBUG)
logger.propagate = False

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
consoleHandler.setLevel(logging.WARNING)
logger.addHandler(consoleHandler)


def measureToAs(evtFile, timMod, tempModPP, toagtifile, eneLow=0.5, eneHigh=10., toaStart=0, toaEnd=None,
                phShiftRes=1000, nbrBins=15, varyAmps=False, readvaryparam=False, brutemin=False, plotPPs=False,
                plotLLs=False, toaFile='ToAs', timFile=None):
    """
    Measure ToAs given an event file (could be merged along TIME **and** gti), a timing model (.par file),
    a template pulse profile (built with **createTemplatePulseProfile**), and  file containing the details
    of the ToAs (i.e. start and end times, built with **buildTimeIntervalsForToAs**)
    :param evtFile: name of the fits event file
    :type evtFile: str
    :param timMod: name of timing model (.par file)
    :type timMod: str
    :param tempModPP: name of the template pulse profile
    :type tempModPP: str
    :param toagtifile: file containing the details of the ToAs (i.e. start and end times, built with **buildTimeIntervalsForToAs**)
    :type toagtifile: str
    :param eneLow: low energy cutoff (default = 0.5 keV)
    :type eneLow: float
    :param eneHigh: high energy cutoff (default = 10 keV)
    :type eneHigh: float
    :param toaStart: number of ToA to start from (default = first)
    :type toaStart: int
    :param toaEnd: stop ToA calculation here (default = last)
    :type toaEnd: int
    :param phShiftRes: step resolution at which to calculate uncertainties in ToAs (default = 1000)
    :type phShiftRes: int
    :param nbrBins: number of bins in pulse profile (for plotting purposes only, default = 15)
    :type nbrBins: int
    :param varyAmps: boolean flag to vary pulse amplitudes (default = False)
    :type varyAmps: bool
    :param readvaryparam: whether to read-in the 'vary' keyword from tempModPP,
    default=False, i.e., everything is fixed except for the phase-shift
    :type readvaryparam: bool
    :param brutemin: boolean flag to run the global minimizing running the BRUTE method (only applicable to Fourier templates, default = False)
    :type brutemin: bool
    :param plotPPs: boolean flag to plot pulse profile and best fit model (default = False)
    :type plotPPs: bool
    :param plotLLs: boolean flag to plot likelihood curve (default = False)
    :type plotLLs: bool
    :param toaFile: file name of ToAs (default = "ToAs".txt)
    :type toaFile: str
    :param timFile: file name of .tim file (default=None)
    :type timFile: str
    :return: ToAsTable
    :rtype: pandas.DataFrame
    """
    fileHandler = logging.FileHandler(toaFile + '.log', mode='w')
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.INFO)
    logger.addHandler(fileHandler)
    logger.info('\n Running measureToAs with input parameters: '
                '\n evtFile: ' + evtFile +
                '\n timMod: ' + timMod +
                '\n tempModPP: ' + tempModPP +
                '\n toagtifile: ' + toagtifile +
                '\n eneLow: ' + str(eneLow) +
                '\n eneHigh: ' + str(eneHigh) +
                '\n toaStart: ' + str(toaStart) +
                '\n toaEnd: ' + str(toaEnd) +
                '\n phShiftRes: ' + str(phShiftRes) +
                '\n nbrBins: ' + str(nbrBins) +
                '\n varyAmps: ' + str(varyAmps) +
                '\n readvaryparam: ' + str(readvaryparam) +
                '\n brutemin: ' + str(brutemin) +
                '\n plotPPs: ' + str(plotPPs) +
                '\n plotLLs: ' + str(plotLLs) +
                '\n output toaFile: ' + toaFile +
                '\n output timFile: ' + str(timFile) + '\n')

    # Reading data and filtering for energy
    #######################################
    EF = EvtFileOps(evtFile)
    # Reading TIME column after energy filtering
    dataTP_eneFlt = EF.build_time_energy_df().filtenergy(eneLow=eneLow, eneHigh=eneHigh)
    TIMEMJD = dataTP_eneFlt.time_energy_df['TIME'].to_numpy()

    # Reading ToA intervals from text file
    ######################################
    df_gtiToAParam = pd.read_csv(toagtifile, sep=r'\s+', comment='#')

    ToAStartMJD = df_gtiToAParam['ToA_tstart'].to_numpy()
    ToAEndMJD = df_gtiToAParam['ToA_tend'].to_numpy()
    ToA_lenInt = df_gtiToAParam['ToA_lenInt'].to_numpy()
    ToA_exposure = df_gtiToAParam['ToA_exposure'].to_numpy()
    Events = df_gtiToAParam['Events'].to_numpy()
    ct_rate = df_gtiToAParam['ct_rate'].to_numpy()

    # Stop at a certain ToA
    if toaEnd is None:  # if not given Measure ToAs for all time intervals in text file
        toaEnd = np.size(ToAEndMJD)
    else:
        toaEnd += 1  # Simply to ensure that toaEnd is inclusive

    # Initilizing relevant arrays
    ToA_MJD_all = np.zeros(toaEnd - toaStart)
    phShiBF_all = np.zeros(toaEnd - toaStart)
    phShiLL_all = np.zeros(toaEnd - toaStart)
    phShiUL_all = np.zeros(toaEnd - toaStart)

    # Calculating phase shift and corresponding properties for each ToA - writing results to text file
    ##################################################################################################
    f = open(toaFile + '.txt', "w+")
    f.write(
        'ToA \t ToA_mid \t ToA_start \t ToA_end \t ToA_lenInt \t ToA_exp \t nbr_events \t count_rate \t phShift \t phShift_LL \t phShift_UL \t Hpower \t redChi2\n')

    # Reading template model
    BFtempModPP = readPPtemplate(tempModPP)  # BFtempModPP is a dictionary of parameters of best-fit template model
    logger.info('\n Using best fit model of template {} to measure ToAs'.format(BFtempModPP["model"]))

    for ii in range(toaStart, toaEnd):

        # Here ii ignores some commented out ToA intervals in "toagtifile" - it is an easy fix though not sure if it is necessary/helpful
        # At the end of the day, we want to count the ToAs that mattered, so skipping on the numbers of ToAs might be confusing

        TIME_toa_bool = (TIMEMJD >= ToAStartMJD[ii]) & (TIMEMJD <= ToAEndMJD[ii])
        TIME_toa = TIMEMJD[TIME_toa_bool]

        ToA_ID = 'ToA' + str(ii)
        # Not sure logging is necessary here - this should suffice to give the user an idea of progress
        print('ToA {}'.format(ii))

        # reference time of each GTI, middle of TIME_toa array
        ######################################################
        ToA_mid = ((TIME_toa[-1] - TIME_toa[0]) / 2) + TIME_toa[0]

        # Fold data using timing model
        ##############################
        phases, cycleFoldedPhases = calcphase(TIME_toa, timMod)

        # Measuring the best fit phase shift through an unbinned extended maximum likelihood fit
        ########################################################################################
        if BFtempModPP["model"] == 'fourier':
            ToAProp = measureToA_fourier(BFtempModPP, cycleFoldedPhases, ToA_exposure[ii], outFile=ToA_ID,
                                         phShiftRes=phShiftRes, nbrBins=nbrBins, varyAmps=varyAmps, brutemin=brutemin,
                                         plotPPs=plotPPs, plotLLs=plotLLs, readvaryparam=readvaryparam)
        elif BFtempModPP["model"] == 'cauchy':
            cycleFoldedPhases *= (2 * np.pi)
            ToAProp = measureToA_cauchy(BFtempModPP, cycleFoldedPhases, ToA_exposure[ii], outFile=ToA_ID,
                                        phShiftRes=phShiftRes, nbrBins=nbrBins, varyAmps=varyAmps, brutemin=brutemin,
                                        plotPPs=plotPPs, plotLLs=plotLLs, readvaryparam=readvaryparam)
        elif BFtempModPP["model"] == 'vonmises':
            cycleFoldedPhases *= (2 * np.pi)
            ToAProp = measureToA_vonmises(BFtempModPP, cycleFoldedPhases, ToA_exposure[ii], outFile=ToA_ID,
                                          phShiftRes=phShiftRes, nbrBins=nbrBins, varyAmps=varyAmps, brutemin=brutemin,
                                          plotPPs=plotPPs, plotLLs=plotLLs, readvaryparam=readvaryparam)
        else:
            logger.error(
                'Model {} is not supported yet; fourier, vonmises, cauchy are supported'.format(BFtempModPP["model"]))

        # Calculating Htest power at the epoch frequency
        ################################################
        ephemeridesAtTmid = ephemTmjd(ToA_mid, timMod)
        htestAtTmid = PeriodSearch(TIME_toa * 86400, np.atleast_1d(ephemeridesAtTmid["freqAtTmjd"]), nbrHarm=5)
        htestPowAtTmid = htestAtTmid.htest()[0]

        # Dictionary of ToA properties - updating with few more keys
        ############################################################
        ToAProp = {'ToA_mid': ToA_mid, 'phShi': ToAProp["phShi"], 'phShi_LL': ToAProp["phShi_LL"],
                   'phShi_UL': ToAProp["phShi_UL"], 'htestPow': htestPowAtTmid,
                   'reducedChi2': ToAProp["reducedChi2"]}

        # writing results to text file
        ##############################
        f.write(str(ii) + '\t' + str(ToAProp['ToA_mid']) + '\t' + str(ToAStartMJD[ii]) + '\t' + str(
            ToAEndMJD[ii]) + '\t' + str(ToA_lenInt[ii]) + '\t' + str(ToA_exposure[ii]) + '\t' + str(
            Events[ii]) + '\t' + str(ct_rate[ii]) + '\t' + str(ToAProp['phShi']) + '\t' + str(
            ToAProp['phShi_LL']) + '\t' + str(ToAProp['phShi_UL']) + '\t' + str(ToAProp['htestPow']) + '\t' + str(
            ToAProp['reducedChi2']) + '\n')

        ToA_MJD_all[ii - toaStart] = ToAProp['ToA_mid']
        phShiBF_all[ii - toaStart] = ToAProp['phShi']
        phShiLL_all[ii - toaStart] = ToAProp['phShi_LL']
        phShiUL_all[ii - toaStart] = ToAProp['phShi_UL']

    f.close()
    logger.info('\n Wrote ToA properties to {}.txt'.format(toaFile))

    # Creating .tim file if specified
    #################################
    if timFile is not None:  # if given convert ToAs.txt to a .tim file
        phshiftTotimfile(toaFile + '.txt', timMod, timFile, tempModPP=tempModPP)
        logger.info('\n Wrote timfile {}.tim'.format(timFile))

    # Plotting Phase residuals of all ToAs
    ######################################
    plotPhaseResiduals(ToA_MJD_all, phShiBF_all, phShiLL_all, phShiUL_all, outFile=toaFile)
    logger.info('\n Created phase residual plot {}_phaseResiduals.pdf'.format(toaFile))

    # PANDAS table of ToAs
    ######################
    ToAsTable = pd.read_csv(toaFile + '.txt', sep=r'\s+', comment='#')

    return ToAsTable


def measureToA_fourier(tempModPP, cycleFoldedPhases, exposureInt, outFile='', phShiftRes=1000, nbrBins=15,
                       varyAmps=False, brutemin=False, plotPPs=False, plotLLs=False, readvaryparam=False):
    """
    Measure ToA according to a Fourier series template
    :param tempModPP: best fit Fourier template model
    :type tempModPP: str
    :param cycleFoldedPhases: array of cycle folded phases [0,1)
    :type cycleFoldedPhases: numpy.ndarray
    :param exposureInt: exposure of good time interval during which we accumulated len(cycleFoldedPhases) (seconds)
    :type exposureInt: int
    :param outFile: name of pulse profile and loglikelihood plots (pp_"outFile".pdf and LogL_"outFile".pdf)
    :type outFile: str
    :param phShiftRes: step resolution at which to calculate uncertainties in ToAs (default = 1000)
    :type phShiftRes: int
    :param nbrBins: number of bins in pulse profile (for plotting purposes only, default = 15)
    :type nbrBins: int
    :param varyAmps: boolean flag to vary pulse amplitudes (default = False)
    :type varyAmps: bool
    :param brutemin: boolean flag to run the global minimizing running the BRUTE method (default = False)
    :type brutemin: bool
    :param plotPPs: boolean flag to plot pulse profile and best fit model (default = False)
    :type plotPPs: bool
    :param plotLLs: boolean flag to plot likelihood curve (default = False)
    :type plotLLs: bool
    :param readvaryparam: whether to read-in the 'vary' keyword from tempModPP,
    default=False, i.e., everything is fixed except for the phase-shift and a constant normalization
    :type readvaryparam: bool
    :return: ToAPropFourier, a dictionary of ToA properties
    :rtype: dict
    """

    initTempModPPparam, nbrFreeParams = defineinitialfitparam(tempModPP, readvaryparam=readvaryparam)

    # Running the extended maximum likelihood
    def unbinnednllfourier(param, xx, exposure):
        return -Fourier(param, xx).loglikelihoodFSnormalized(exposure)

    # Run brute force minimization if requested
    if brutemin is True:
        results_mle_FSNormalized_bm = minimize(unbinnednllfourier, initTempModPPparam,
                                               args=(cycleFoldedPhases, exposureInt),
                                               method='brute', max_nfev=1.0e4, nan_policy='propagate')

        results_mle_FSNormalized = minimize(unbinnednllfourier, results_mle_FSNormalized_bm.params,
                                            args=(cycleFoldedPhases, exposureInt),
                                            method='nedler', max_nfev=1.0e4, nan_policy='propagate')
    else:
        results_mle_FSNormalized = minimize(unbinnednllfourier, initTempModPPparam,
                                            args=(cycleFoldedPhases, exposureInt),
                                            method='nedler', max_nfev=1.0e4, nan_policy='propagate')

    # In case pulsed fraction should be varied
    if varyAmps is True:
        initTempModPPparam_afterfit = copy.deepcopy(results_mle_FSNormalized.params)
        initTempModPPparam_afterfit.add('ampShift', 1, min=0.01, max=100, vary=True)
        results_mle_FSNormalized = minimize(unbinnednllfourier, initTempModPPparam_afterfit,
                                            args=(cycleFoldedPhases, exposureInt),
                                            method='nedler', max_nfev=1.0e4, nan_policy='propagate')
        nbrFreeParams += 1  # In this case we varied another parameter

    phShiBF = results_mle_FSNormalized.params.valuesdict()['phShift']  # Best fit phase shift
    LLmax = -results_mle_FSNormalized.residual  # the - sign is to flip the likelihood - no reason for it, just personal choice

    # Calculating the uncertainties on phase-shift
    ##############################################
    # Stepping over the phase-shift from best-fit +/- 2pi/phShiftRes (default phShiftRes=1000)
    phShiftStep = (2 * np.pi) / phShiftRes
    initParam_forErrCalc = copy.deepcopy(results_mle_FSNormalized.params)

    # Our LogL follows a chi2 distribution with 1 d.o.f. (approximately)
    chi2_1sig1dof = 0.5 * chi2.ppf(0.6827, 1)  # 0.68 represents 1 sigma deviation
    chi2diff1sig = 0  # initial difference
    kk = 1  # counter for number of steps we shifted phShift
    logLikeDist = []  # In case we want to produce a loglikeliood plot
    phShiftDist = []

    # Calculating the 1-sigma lower-limit uncertainty
    while chi2diff1sig <= chi2_1sig1dof:
        if initParam_forErrCalc['phShift'].value <= -np.pi:
            initParam_forErrCalc['phShift'].min = phShiBF - kk * phShiftStep
            initParam_forErrCalc['phShift'].value = phShiBF - kk * phShiftStep
        else:
            initParam_forErrCalc['phShift'].value = phShiBF - kk * phShiftStep
        initParam_forErrCalc['phShift'].vary = False
        # Performing the fitting
        results_mle_forErrCalc = minimize(unbinnednllfourier, initParam_forErrCalc,
                                          args=(cycleFoldedPhases, exposureInt),
                                          method='nelder', max_nfev=1.0e4, nan_policy='propagate')
        logLikeDist = np.hstack((logLikeDist, -results_mle_forErrCalc.residual))
        phShiftDist = np.hstack((phShiftDist, phShiBF - kk * phShiftStep))
        # New difference 
        chi2diff1sig = LLmax - (-results_mle_forErrCalc.residual)
        # Updating counter
        kk += 1
        if kk > phShiftRes / 2:
            logger.warning('Could not estimate lower-bound uncertainty on {}'.format(outFile))
            break
    phShiBF_LL = (kk * phShiftStep + phShiftStep / 2)

    # Calculating the 1-sigma upper-limit uncertainty
    chi2diff1sig = 0  # Reset initial difference
    kk = 1  # Reset counter for number of steps we shifted phShift
    while chi2diff1sig <= chi2_1sig1dof:
        if initParam_forErrCalc['phShift'].value >= np.pi:
            initParam_forErrCalc['phShift'].max = phShiBF + kk * phShiftStep
            initParam_forErrCalc['phShift'].value = phShiBF + kk * phShiftStep
        else:
            initParam_forErrCalc['phShift'].value = phShiBF + kk * phShiftStep
        initParam_forErrCalc['phShift'].vary = False
        # Performing the fitting
        results_mle_forErrCalc = minimize(unbinnednllfourier, initParam_forErrCalc,
                                          args=(cycleFoldedPhases, exposureInt),
                                          method='nelder', max_nfev=1.0e4, nan_policy='propagate')
        logLikeDist = np.hstack((logLikeDist, -results_mle_forErrCalc.residual))
        phShiftDist = np.hstack((phShiftDist, phShiBF + kk * phShiftStep))
        # New difference 
        chi2diff1sig = LLmax - (-results_mle_forErrCalc.residual)
        # Updating counter
        kk += 1
        if kk > phShiftRes / 2:
            logger.warning('Could not estimate upper-bound uncertainty on {}'.format(outFile))
            break
    phShiBF_UL = (kk * phShiftStep + phShiftStep / 2)

    # Plotting the Log(L) distribution
    # For debugging purposes only
    if plotLLs is True:
        plotLogLikelihood(phShiftDist, logLikeDist, outFile=outFile)

    # Measuring chi2 of each profile from model template
    ####################################################
    binnedProfile = binphases(cycleFoldedPhases, nbrBins)
    ppBins = binnedProfile["ppBins"]
    ctRate = binnedProfile["ctsBins"] / (exposureInt / nbrBins)
    ctRateErr = binnedProfile["ctsBinsErr"] / (exposureInt / nbrBins)
    # Best fit model
    bfModel = Fourier(results_mle_FSNormalized.params, ppBins).fourseries()
    # Chi2 and reduced chi2
    chi2_pp = np.sum(np.divide(((bfModel - ctRate) ** 2), ctRateErr ** 2))
    redchi2_pp = np.divide(chi2_pp, np.size(ppBins) - nbrFreeParams)

    # Plotting pulse profile of each ToA along with the best fit template model before and after correcting the phase shift
    #######################################################################################################################
    if plotPPs is True:
        initModel = Fourier(initTempModPPparam, ppBins).fourseries()
        plotPPofToAs(ppBins, ctRate, ctRateErr, bfModel, initModel, outFile=outFile)

    ToAPropFourier = {'phShi': phShiBF, 'phShi_LL': phShiBF_LL, 'phShi_UL': phShiBF_UL, 'reducedChi2': redchi2_pp}

    return ToAPropFourier


def measureToA_cauchy(tempModPP, cycleFoldedPhases, exposureInt, outFile='', phShiftRes=1000, nbrBins=15,
                      varyAmps=False, brutemin=False, plotPPs=False, plotLLs=False, readvaryparam=False):
    """
    Measure ToA according to a wrapped-cauchy template
    :param tempModPP: best fit Fourier template model
    :type tempModPP: str
    :param cycleFoldedPhases: array of cycle folded phases [0,2*np.pi)
    :type cycleFoldedPhases: numpy.ndarray
    :param exposureInt: exposure of good time interval during which we accumulated len(cycleFoldedPhases) (seconds)
    :type exposureInt: int
    :param outFile: name of pulse profile and loglikelihood plots (pp_"outFile".pdf and LogL_"outFile".pdf)
    :type outFile: str
    :param phShiftRes: step resolution at which to calculate uncertainties in ToAs (default = 1000)
    :type phShiftRes: int
    :param nbrBins: number of bins in pulse profile (for plotting purposes only, default = 15)
    :type nbrBins: int
    :param varyAmps: boolean flag to vary pulse amplitudes (default = False)
    :type varyAmps: bool
    :param brutemin: boolean flag to run the global minimizing running the BRUTE method (default = False)
    :type brutemin: bool
    :param plotPPs: boolean flag to plot pulse profile and best fit model (default = False)
    :type plotPPs: bool
    :param plotLLs: boolean flag to plot likelihood curve (default = False)
    :type plotLLs: bool
    :param readvaryparam: whether to read-in the 'vary' keyword from tempModPP,
    default=False, i.e., everything is fixed except for the phase-shift and a constant normalization
    :type readvaryparam: bool
    :return: ToAPropFourier, a dictionary of ToA properties
    :rtype: dict
    """

    initTempModPPparam, nbrFreeParams = defineinitialfitparam(tempModPP, readvaryparam=readvaryparam)

    # Running the extended maximum likelihood
    def unbinnednllcauchy(param, xx, exposure):
        return -WrappedCauchy(param, xx).loglikelihoodCAnormalized(exposure)

    # Run brute force minimization if requested
    if brutemin is True:
        results_mle_CANormalized_bm = minimize(unbinnednllcauchy, initTempModPPparam,
                                               args=(cycleFoldedPhases, exposureInt),
                                               method='brute', max_nfev=1.0e4, nan_policy='propagate')

        results_mle_CANormalized = minimize(unbinnednllcauchy, results_mle_CANormalized_bm.params,
                                            args=(cycleFoldedPhases, exposureInt),
                                            method='nedler', max_nfev=1.0e4, nan_policy='propagate')

    else:
        results_mle_CANormalized = minimize(unbinnednllcauchy, initTempModPPparam,
                                            args=(cycleFoldedPhases, exposureInt),
                                            method='nedler', max_nfev=1.0e4, nan_policy='propagate')

    # In case pulsed fraction should be varied
    if varyAmps is True:
        initTempModPPparam_afterfit = copy.deepcopy(results_mle_CANormalized.params)
        initTempModPPparam_afterfit.add('ampShift', 1, min=0.0, max=np.inf, vary=True)
        results_mle_CANormalized = minimize(unbinnednllcauchy, initTempModPPparam_afterfit,
                                            args=(cycleFoldedPhases, exposureInt),
                                            method='nelder', max_nfev=1.0e3, nan_policy='propagate')
        nbrFreeParams += 1  # In this case we varied another parameter

    phShiBF = results_mle_CANormalized.params.valuesdict()['phShift']  # Best fit phase shift
    LLmax = -results_mle_CANormalized.residual  # the - sign is to flip the likelihood - no reason for it, just personal choice

    # Calculating the uncertainties on phase-shift
    ##############################################
    # Stepping over the phase-shift from best-fit +/- 2pi/phShiftRes (default phShiftRes=1000)
    phShiftStep = (2 * np.pi) / phShiftRes
    initParam_forErrCalc = copy.deepcopy(results_mle_CANormalized.params)

    # Our LogL follows a chi2 distribution with 1 d.o.f. (approximately)
    chi2_1sig1dof = 0.5 * chi2.ppf(0.6827, 1)  # 0.68 represents 1 sigma deviation
    chi2diff1sig = 0  # initial difference
    kk = 1  # counter for number of steps we shifted phShift
    logLikeDist = []  # In case we want to produce a loglikeliood plot
    phShiftDist = []

    # Calculating the 1-sigma lower-limit uncertainty
    while chi2diff1sig <= chi2_1sig1dof:
        initParam_forErrCalc['phShift'].value = phShiBF - kk * phShiftStep
        initParam_forErrCalc['phShift'].vary = False
        # Performing the fitting
        results_mle_forErrCalc = minimize(unbinnednllcauchy, initParam_forErrCalc,
                                          args=(cycleFoldedPhases, exposureInt),
                                          method='nedler', max_nfev=1.0e4, nan_policy='propagate')
        logLikeDist = np.hstack((logLikeDist, -results_mle_forErrCalc.residual))
        phShiftDist = np.hstack((phShiftDist, phShiBF - kk * phShiftStep))
        # New difference
        chi2diff1sig = LLmax - (-results_mle_forErrCalc.residual)
        # Updating counter
        kk += 1
        if kk > phShiftRes / 2:
            logger.warning('Could not estimate lower-bound uncertainty on {}'.format(outFile))
            break
    phShiBF_LL = (kk * phShiftStep + phShiftStep / 2)

    # Calculating the 1-sigma upper-limit uncertainty
    chi2diff1sig = 0  # Reset initial difference
    kk = 1  # Reset counter for number of steps we shifted phShift
    while chi2diff1sig <= chi2_1sig1dof:
        initParam_forErrCalc['phShift'].value = phShiBF + kk * phShiftStep
        initParam_forErrCalc['phShift'].vary = False
        # Performing the fitting
        results_mle_forErrCalc = minimize(unbinnednllcauchy, initParam_forErrCalc,
                                          args=(cycleFoldedPhases, exposureInt),
                                          method='nedler', max_nfev=1.0e4, nan_policy='propagate')
        logLikeDist = np.hstack((logLikeDist, -results_mle_forErrCalc.residual))
        phShiftDist = np.hstack((phShiftDist, phShiBF + kk * phShiftStep))
        # New difference
        chi2diff1sig = LLmax - (-results_mle_forErrCalc.residual)
        # Updating counter
        kk += 1
        if kk > phShiftRes / 2:
            logger.warning('Could not estimate upper-bound uncertainty on {}'.format(outFile))
            break
    phShiBF_UL = (kk * phShiftStep + phShiftStep / 2)

    # Plotting the Log(L) distribution
    # For debugging purposes only
    if plotLLs is True:
        plotLogLikelihood(phShiftDist, logLikeDist, outFile=outFile)

    # Measuring chi2 of each profile from model template
    ####################################################
    binnedProfile = binphases(cycleFoldedPhases, nbrBins)
    ppBins = binnedProfile["ppBins"]
    ctRate = binnedProfile["ctsBins"] / (exposureInt / nbrBins)
    ctRateErr = binnedProfile["ctsBinsErr"] / (exposureInt / nbrBins)
    # Best fit model
    bfModel = WrappedCauchy(results_mle_CANormalized.params, ppBins).wrapcauchy()
    # Chi2 and reduced chi2
    chi2_pp = np.sum(np.divide(((bfModel - ctRate) ** 2), ctRateErr ** 2))
    redchi2_pp = np.divide(chi2_pp, np.size(ppBins) - nbrFreeParams)

    # Plotting pulse profile of each ToA along with the best fit template model before and after correcting the phase shift
    #######################################################################################################################
    if plotPPs is True:
        initModel = WrappedCauchy(initTempModPPparam, ppBins).wrapcauchy()
        plotPPofToAs(ppBins, ctRate, ctRateErr, bfModel, initModel, outFile=outFile)

    ToAPropCauchy = {'phShi': phShiBF, 'phShi_LL': phShiBF_LL, 'phShi_UL': phShiBF_UL, 'reducedChi2': redchi2_pp}

    return ToAPropCauchy


def measureToA_vonmises(tempModPP, cycleFoldedPhases, exposureInt, outFile='', phShiftRes=1000, nbrBins=15,
                        varyAmps=False, brutemin=False, plotPPs=False, plotLLs=False, readvaryparam=False):
    """
    Measure ToA according to a von Mises template
    :param tempModPP: best fit von Mises template model
    :type tempModPP: str
    :param cycleFoldedPhases: array of cycle folded phases [0,2*np.pi)
    :type cycleFoldedPhases: numpy.ndarray
    :param exposureInt: exposure of good time interval during which we accumulated len(cycleFoldedPhases) (seconds)
    :type exposureInt: int
    :param outFile: name of pulse profile and loglikelihood plots (pp_"outFile".pdf and LogL_"outFile".pdf)
    :type outFile: str
    :param phShiftRes: step resolution at which to calculate uncertainties in ToAs (default = 1000)
    :type phShiftRes: int
    :param nbrBins: number of bins in pulse profile (for plotting purposes only, default = 15)
    :type nbrBins: int
    :param varyAmps: boolean flag to vary pulse amplitudes (default = False)
    :type varyAmps: bool
    :param brutemin: boolean flag to run the global minimizing running the BRUTE method (default = False)
    :type brutemin: bool
    :param plotPPs: boolean flag to plot pulse profile and best fit model (default = False)
    :type plotPPs: bool
    :param plotLLs: boolean flag to plot likelihood curve (default = False)
    :type plotLLs: bool
    :param readvaryparam: whether to read-in the 'vary' keyword from tempModPP,
    default=False, i.e., everything is fixed except for the phase-shift and a constant normalization
    :type readvaryparam: bool
    :return: ToAPropFourier, a dictionary of ToA properties
    :rtype: dict
    """

    initTempModPPparam, nbrFreeParams = defineinitialfitparam(tempModPP, readvaryparam=readvaryparam)

    # Running the extended maximum likelihood
    def unbinnednllvonmises(param, xx, exposure):
        return -VonMises(param, xx).loglikelihoodVMnormalized(exposure)

    if brutemin is True:
        results_mle_VMNormalized_bm = minimize(unbinnednllvonmises, initTempModPPparam,
                                               args=(cycleFoldedPhases, exposureInt),
                                               method='brute', max_nfev=1.0e4, nan_policy='propagate')

        results_mle_VMNormalized = minimize(unbinnednllvonmises, results_mle_VMNormalized_bm.params,
                                            args=(cycleFoldedPhases, exposureInt),
                                            method='nedler', max_nfev=1.0e4, nan_policy='propagate')

    else:
        results_mle_VMNormalized = minimize(unbinnednllvonmises, initTempModPPparam,
                                            args=(cycleFoldedPhases, exposureInt),
                                            method='nedler', max_nfev=1.0e4, nan_policy='propagate')

    # In case pulsed fraction should be varied
    if varyAmps is True:
        initTempModPPparam_afterfit = copy.deepcopy(results_mle_VMNormalized.params)
        initTempModPPparam_afterfit.add('ampShift', 1, min=0.0, max=np.inf, vary=True)
        results_mle_VMNormalized = minimize(unbinnednllvonmises, initTempModPPparam_afterfit,
                                            args=(cycleFoldedPhases, exposureInt),
                                            method='nelder', max_nfev=1.0e3, nan_policy='propagate')
        nbrFreeParams += 1  # In this case we varied another parameter

    phShiBF = results_mle_VMNormalized.params.valuesdict()['phShift']  # Best fit phase shift
    LLmax = -results_mle_VMNormalized.residual  # the - sign is to flip the likelihood - no reason for it, just personal choice

    # Calculating the uncertainties on phase-shift
    ##############################################
    # Stepping over the phase-shift from best-fit +/- 2pi/phShiftRes (default phShiftRes=1000)
    phShiftStep = (2 * np.pi) / phShiftRes
    initParam_forErrCalc = copy.deepcopy(results_mle_VMNormalized.params)

    # Our LogL follows a chi2 distribution with 1 d.o.f. (approximately)
    chi2_1sig1dof = 0.5 * chi2.ppf(0.6827, 1)  # 0.68 represents 1 sigma deviation
    chi2diff1sig = 0  # initial difference
    kk = 1  # counter for number of steps we shifted phShift
    logLikeDist = []  # In case we want to produce a loglikeliood plot
    phShiftDist = []

    # Calculating the 1-sigma lower-limit uncertainty
    while chi2diff1sig <= chi2_1sig1dof:
        initParam_forErrCalc['phShift'].value = phShiBF - kk * phShiftStep
        initParam_forErrCalc['phShift'].vary = False
        # Performing the fitting
        results_mle_forErrCalc = minimize(unbinnednllvonmises, initParam_forErrCalc,
                                          args=(cycleFoldedPhases, exposureInt),
                                          method='nedler', max_nfev=1.0e4, nan_policy='propagate')
        logLikeDist = np.hstack((logLikeDist, -results_mle_forErrCalc.residual))
        phShiftDist = np.hstack((phShiftDist, phShiBF - kk * phShiftStep))
        # New difference
        chi2diff1sig = LLmax - (-results_mle_forErrCalc.residual)
        # Updating counter
        kk += 1
        if kk > phShiftRes / 2:
            logger.warning('Could not estimate lower-bound uncertainty on {}'.format(outFile))
            break
    phShiBF_LL = (kk * phShiftStep + phShiftStep / 2)

    # Calculating the 1-sigma upper-limit uncertainty
    chi2diff1sig = 0  # Reset initial difference
    kk = 1  # Reset counter for number of steps we shifted phShift
    while chi2diff1sig <= chi2_1sig1dof:
        initParam_forErrCalc['phShift'].value = phShiBF + kk * phShiftStep
        initParam_forErrCalc['phShift'].vary = False
        # Performing the fitting
        results_mle_forErrCalc = minimize(unbinnednllvonmises, initParam_forErrCalc,
                                          args=(cycleFoldedPhases, exposureInt),
                                          method='nedler', max_nfev=1.0e4, nan_policy='propagate')
        logLikeDist = np.hstack((logLikeDist, -results_mle_forErrCalc.residual))
        phShiftDist = np.hstack((phShiftDist, phShiBF + kk * phShiftStep))
        # New difference
        chi2diff1sig = LLmax - (-results_mle_forErrCalc.residual)
        # Updating counter
        kk += 1
        if kk > phShiftRes / 2:
            logger.warning('Could not estimate upper-bound uncertainty on {}'.format(outFile))
            break
    phShiBF_UL = (kk * phShiftStep + phShiftStep / 2)

    # Plotting the Log(L) distribution
    # For debugging purposes only
    if plotLLs is True:
        plotLogLikelihood(phShiftDist, logLikeDist, outFile=outFile)

    # Measuring chi2 of each profile from model template
    ####################################################
    binnedProfile = binphases(cycleFoldedPhases, nbrBins)
    ppBins = binnedProfile["ppBins"]
    ctRate = binnedProfile["ctsBins"] / (exposureInt / nbrBins)
    ctRateErr = binnedProfile["ctsBinsErr"] / (exposureInt / nbrBins)
    # Best fit model
    bfModel = VonMises(results_mle_VMNormalized.params, ppBins).vonmises()

    # Chi2 and reduced chi2
    chi2_pp = np.sum(np.divide(((bfModel - ctRate) ** 2), ctRateErr ** 2))
    redchi2_pp = np.divide(chi2_pp, np.size(ppBins) - nbrFreeParams)

    # Plotting pulse profile of each ToA along with the best fit template model before and after correcting the phase shift
    #######################################################################################################################
    if plotPPs is True:
        initModel = VonMises(initTempModPPparam, ppBins).vonmises()
        plotPPofToAs(ppBins, ctRate, ctRateErr, bfModel, initModel, outFile=outFile)

    ToAPropCauchy = {'phShi': phShiBF, 'phShi_LL': phShiBF_LL, 'phShi_UL': phShiBF_UL, 'reducedChi2': redchi2_pp}

    return ToAPropCauchy


##########################################
# A simple function to read in the initial template best fit parameters
def defineinitialfitparam(tempModPP, readvaryparam=False):
    """
    Define the template model parameters for initiating the ToA calculation
    :param tempModPP: template model parameter file
    :type tempModPP: dict
    :param readvaryparam: whether to read-in the 'vary' keyword from tempModPP,
    default=False, i.e., everything is fixed except for the phase-shift
    :type readvaryparam: bool
    :return: initTempModPPparam
    :rtype: Parameters instance of lmfit
    """

    if tempModPP["model"] == 'fourier':

        if not readvaryparam:
            initTempModPPparam = Parameters()  # Initializing an instance of Parameters based on the best-fit template model
            initTempModPPparam.add('norm', tempModPP['norm']['value'], min=tempModPP['norm']['value']/5,
                                   max=tempModPP['norm']['value']*5, vary=True)
            # Number of components in template model
            nbrComp = len(np.array([ww for harmKey, ww in tempModPP.items() if harmKey.startswith('amp_')]))
            for kk in range(1, nbrComp + 1):  # Adding the amplitudes and phases of the harmonics, they are fixed
                initTempModPPparam.add('amp_' + str(kk), tempModPP['amp_' + str(kk)]['value'], vary=False)
                initTempModPPparam.add('ph_' + str(kk), tempModPP['ph_' + str(kk)]['value'], vary=False)
            initTempModPPparam.add('phShift', 0, vary=True, min=-np.pi, max=np.pi,
                                   brute_step=0.05)  # Phase shift - parameter of interest
            initTempModPPparam.add('ampShift', 1, vary=False, min=0, max=100)
            nbrFreeParams = 2  # phshift and model normalization

        elif readvaryparam:
            initTempModPPparam = Parameters()  # Initializing an instance of Parameters based on the best-fit template model
            initTempModPPparam.add('norm', tempModPP['norm']['value'], min=tempModPP['norm']['value']/5,
                                   max=tempModPP['norm']['value']*5, vary=tempModPP['norm']['vary'])
            # setting up the number of free parameters
            if tempModPP['norm']['vary']:
                nbrFreeParams = 1
            else:
                nbrFreeParams = 0
            # Number of components in template model
            nbrComp = len(np.array([ww for harmKey, ww in tempModPP.items() if harmKey.startswith('amp_')]))
            for kk in range(1, nbrComp + 1):  # Adding the amplitudes and phases of the harmonics, they are fixed
                initTempModPPparam.add('amp_' + str(kk), tempModPP['amp_' + str(kk)]['value'],
                                       vary=tempModPP['amp_' + str(kk)]['vary'], min=0, max=1000)
                initTempModPPparam.add('ph_' + str(kk), tempModPP['ph_' + str(kk)]['value'],
                                       vary=tempModPP['ph_' + str(kk)]['vary'], min=-np.pi, max=np.pi)

                # properly dealing with number of free parameters
                for key in ('amp_' + str(kk), 'ph_' + str(kk)):
                    if tempModPP[key]['vary']:
                        nbrFreeParams += 1

            initTempModPPparam.add('phShift', 0, vary=True, min=-np.pi, max=np.pi,
                                   brute_step=0.05)  # Phase shift - parameter of interest
            initTempModPPparam.add('ampShift', 1, vary=False, min=0, max=100)

    elif tempModPP["model"] in ('vonmises', 'cauchy'):

        if not readvaryparam:
            initTempModPPparam = Parameters()  # Initializing an instance of Parameters based on the best-fit template model
            initTempModPPparam.add('norm', tempModPP['norm']['value'], min=tempModPP['norm']['value']/5,
                                   max=tempModPP['norm']['value']*5, vary=True)  # Adding the normalization - this is free to vary

            # Number of components in template model
            nbrComp = len(np.array([ww for harmKey, ww in tempModPP.items() if harmKey.startswith('amp_')]))
            for kk in range(1,
                            nbrComp + 1):  # Adding the amplitudes, centroids, and widths of the components, they are fixed
                initTempModPPparam.add('amp_' + str(kk), tempModPP['amp_' + str(kk)]['value'], vary=False)
                initTempModPPparam.add('cen_' + str(kk), tempModPP['cen_' + str(kk)]['value'], vary=False)
                initTempModPPparam.add('wid_' + str(kk), tempModPP['wid_' + str(kk)]['value'], vary=False)
            initTempModPPparam.add('phShift', 0, vary=True, min=-1.5 * np.pi,
                                   max=1.5 * np.pi, brute_step=0.05)  # Phase shift - parameter of interest
            initTempModPPparam.add('ampShift', 1, vary=False, min=-np.pi, max=np.pi)

            nbrFreeParams = 2  # phshift and model normalization

        elif readvaryparam:
            initTempModPPparam = Parameters()  # Initializing an instance of Parameters based on the best-fit template model
            initTempModPPparam.add('norm', tempModPP['norm']['value'], min=tempModPP['norm']['value']/5,
                                   max=tempModPP['norm']['value']*5, vary=tempModPP['norm']['vary'])  # Adding the normalization - this is free to vary
            # setting up the number of free parameters
            if tempModPP['norm']['vary']:
                nbrFreeParams = 1
            else:
                nbrFreeParams = 0
            # Number of components in template model
            nbrComp = len(np.array([ww for harmKey, ww in tempModPP.items() if harmKey.startswith('amp_')]))
            for kk in range(1, nbrComp + 1):  # Adding the amplitudes, centroids, and widths of the components, they are fixed
                initTempModPPparam.add('amp_' + str(kk), tempModPP['amp_' + str(kk)]['value'],
                                       vary=tempModPP['amp_' + str(kk)]['vary'], min=0,
                                       max=5*tempModPP['amp_' + str(kk)]['value'])
                initTempModPPparam.add('cen_' + str(kk), tempModPP['cen_' + str(kk)]['value'],
                                       vary=tempModPP['cen_' + str(kk)]['vary'],
                                       min=-0.6+tempModPP['cen_' + str(kk)]['value'],
                                       max=0.6+tempModPP['cen_' + str(kk)]['value'], brute_step=0.05)
                initTempModPPparam.add('wid_' + str(kk), tempModPP['wid_' + str(kk)]['value'],
                                       vary=tempModPP['wid_' + str(kk)]['vary'], min=0, max=30*np.pi)  # basically inf

                # properly dealing with number of free parameters
                for key in ('amp_' + str(kk), 'cen_' + str(kk), 'wid_' + str(kk)):
                    if tempModPP[key]['vary']:
                        nbrFreeParams += 1
            initTempModPPparam.add('phShift', 0, vary=True, min=-1.5 * np.pi,
                                   max=1.5 * np.pi, brute_step=0.05)  # Phase shift - parameter of interest
            initTempModPPparam.add('ampShift', 1, vary=False)

    else:
        logger.error('Unknown template model. Only fourier, cauchy, or vonmises are supported')

    return initTempModPPparam, nbrFreeParams


##########################################
# A simple function to measure the phase shift and 1 sigma uncertainty given the Log_Likelihood
def plotLogLikelihood(phaseShifts, logLikeDist, outFile):
    """
    Plot the log-likelihood around the best fit phaseShift
    :param phaseShifts: array of phase shifts
    :type phaseShifts: numpy.ndarray
    :param logLikeDist: log-likelihood of corresponding phase shifts
    :type logLikeDist: numpy.ndarray
    :param outFile: name of the log-likelihood plot
    :type outFile: str
    """
    fig, ax1 = plt.subplots(1, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlabel(r'$\,\mathrm{Phase\,(cycles)}$', fontsize=12)
    ax1.set_ylabel(r'$\,\mathrm{Log(L)}$', fontsize=12)
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.yaxis.offsetText.set_fontsize(12)

    ax1.plot(phaseShifts / (2 * np.pi), logLikeDist, 'k.', lw=3)

    # Finishing touches
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.5)
        ax1.tick_params(width=1.5)

    fig.tight_layout()

    plotName = 'LogL_' + str(outFile) + '.pdf'
    fig.savefig(plotName, format='pdf', dpi=1000)
    plt.close()


##########################################
# Plotting pulse profile with template fit + phase shift
def plotPPofToAs(ppBins, ctRate, ctRateErr, bfModel, initModel, outFile):
    """
    Plot the pulse profile and best fit model of a ToA
    :param ppBins: centroid of pulse profile bins
    :type ppBins: numpy.ndarray
    :param ctRate: countrate of pulse profile bins
    :type ctRate: numpy.ndarray
    :param ctRateErr: uncertainty on count rate of pulse profile bins
    :type ctRateErr: numpy.ndarray
    :param bfModel: model pridicted count rate
    :type bfModel: numpy.ndarray
    :param initModel: initial template model pridicted count rate
    :type initModel: numpy.ndarray
    :param outFile: name of the pulse-profile plot
    :type outFile: str
    """
    # Initiating plot
    fig, ax1 = plt.subplots(1, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k')

    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlabel(r'$\,\mathrm{Phase\,(cycles)}$', fontsize=12)
    ax1.set_ylabel(r'$\,\mathrm{Normalized~rate}$', fontsize=12)
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.yaxis.offsetText.set_fontsize(12)

    if np.max(ppBins) > 1:  # Plotting a second cycle for cauchy or vonmises (cycle = 2*pi)
        secondCycle = 2 * np.pi
    else:  # Plotting a second cycle for fourier (cycle = 1)
        secondCycle = 1

    # Creating two cycles and plotting PP
    ppBins_plt = np.append(ppBins, ppBins + secondCycle)
    ctRate_plt = np.append(ctRate, ctRate)
    ctRateErr_plt = np.append(ctRateErr, ctRateErr)

    ax1.errorbar(ppBins_plt, ctRate_plt, yerr=ctRateErr_plt, fmt='ok', zorder=10)
    ax1.step(ppBins_plt, ctRate_plt, 'k+-', where='mid', zorder=10)

    # Overplotting best fit model
    initModel_plt = np.append(initModel, initModel)
    bfModel_plt = np.append(bfModel, bfModel)

    ax1.plot(ppBins_plt, initModel_plt, 'g-', linewidth=2.0, label='Initial template')
    ax1.plot(ppBins_plt, bfModel_plt, 'r-', linewidth=2.0, label='After fitting for phase-shift')

    ax1.legend()

    ax1.set_xlim(0.0, 2 * secondCycle)

    #############################
    # Finishing touches
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.5)
        ax1.tick_params(width=1.5)

    fig.tight_layout()

    plotName = 'pp_' + str(outFile) + '.pdf'
    fig.savefig(plotName, format='pdf', dpi=1000)
    plt.close()


##########################################
# Plotting Phase residuals of all ToAs
def plotPhaseResiduals(ToAsMJD, phaseShifts, phaseShiftsLL, phaseShiftsUL, outFile=''):
    """
    Plot phase residuals
    :param ToAsMJD: mid-times of the ToA interval (strictly speaking, this is not **pulse** ToA)
    :type ToAsMJD: numpy.ndarray
    :param phaseShifts: array of phase shifts at ToAsMJD
    :type phaseShifts: numpy.ndarray
    :param phaseShiftsLL: 1sigma lower bound on phase shifts
    :type phaseShiftsLL: numpy.ndarray
    :param phaseShiftsUL: 1sigma upper bound on phase shifts
    :type phaseShiftsUL: numpy.ndarray
    :param outFile: name of plot (default='outFile'_phaseResiduals.pdf)
    :type outFile:
    """
    fig, ax1 = plt.subplots(1, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k')

    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlabel(r'$\,\mathrm{Time\,(MJD)}$', fontsize=12)
    ax1.set_ylabel(r'$\,\mathrm{\Delta\phi}\,(Cycles)$', fontsize=12)
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.yaxis.offsetText.set_fontsize(12)

    plt.errorbar(ToAsMJD, phaseShifts / (2 * np.pi), yerr=(phaseShiftsLL / (2 * np.pi), phaseShiftsUL / (2 * np.pi)),
                 fmt='ok', zorder=10)

    # Finishing touches
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.5)
        ax1.tick_params(width=1.5)

    fig.tight_layout()

    plotName = str(outFile) + '_phaseResiduals.pdf'
    fig.savefig(plotName, format='pdf', dpi=1000)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Script to measure ToAs from event file")
    parser.add_argument("evtFile", help="Name of a barycentered event file", type=str)
    parser.add_argument("timMod", help="Timing model, Tempo2 .par file should work", type=str)
    parser.add_argument("tempModPP", help="Parameters of template pulse profile (e.g., built with pulseprofile.py)",
                        type=str)
    parser.add_argument("toagtifile",
                        help="User supplied .txt file with ToA interval information (built with buildtimeintervalsToAs.py)",
                        type=str)
    parser.add_argument("-el", "--enelow", help="Low energy filter in event file, default=0.5", type=float, default=0.5)
    parser.add_argument("-eh", "--enehigh", help="High energy filter in event file, default=10", type=float, default=10)
    parser.add_argument("-ts", "--toaStart", help="Number of ToA to start from, default first ToA", type=int, default=0)
    parser.add_argument("-te", "--toaEnd", help="Number of ToA to end, default full length of ToA list", type=int,
                        default=None)
    parser.add_argument("-pr", "--phShiftRes", help="Phase-shift step resolution, 2*pi/phShiftRes, default = 1000",
                        type=int, default=1000)
    parser.add_argument("-nb", "--nbrBins", help="Number of bins in PP for visualization purposes only, default = 15",
                        type=int, default=15)
    parser.add_argument("-va", "--varyAmps",
                        help="Flag to allow the pulsed fraction of the template pulse profile to vary (not the shape!), default = False",
                        type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-rv", "--readvaryparam",
                        help="Flag to read-in the 'vary' keyword for each parameter in the template from the template "
                             "model. default = False", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-bm", "--brutemin",
                        help="boolean flag to run the global minimizing running the BRUTE method, default = False",
                        type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-pp", "--plotPPs", help="Flag to create pulse profile plots, default = False", type=bool,
                        default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-ll", "--plotLLs", help="Flag to create LogLikelihood plots, default = False", type=bool,
                        default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-tf", "--toaFile", help="name of output ToA file (default = ToAs(.txt))", type=str,
                        default='ToAs')
    parser.add_argument("-mf", "--timFile",
                        help="name of output .tim file compatible with tempo2/PINT (default = None - no .tim file created)",
                        type=str, default=None)
    args = parser.parse_args()

    measureToAs(args.evtFile, args.timMod, args.tempModPP, args.toagtifile, args.enelow, args.enehigh, args.toaStart,
                args.toaEnd, args.phShiftRes, args.nbrBins, args.varyAmps, args.readvaryparam, args.brutemin,
                args.plotPPs, args.plotLLs, args.toaFile, args.timFile)


if __name__ == '__main__':
    main()
