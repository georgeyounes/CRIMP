###############################################################################
# This code calculates ToAs a la Ray et al. 2018. It takes a barycentered
# corrected event file, a timing model (.par file), a template model (.txt file,
# e.g., from pulseprofile.py), and a .txt file defining the ToAs (e.g., ToAs
# start and end times - could be built with buildtimeintervalsToAs.py). The user could
# apply an energy filtering to the event file through eneLow and eneHigh.
# A subset of ToAs could be measured if desired through 'toaStart' and/or 'toaEnd'
# Most of the calculations are done in the template-appropriate functions,
# e.g., "measToA_fourier". The appropriate TIME array is folded using the .par file,
# and fit with the provided template model. For this step, we use the extended MLE
# method, all model parameters are fixed, except for the normalization and a
# phase-shift. The uncertainties on the phase-shift is measured by stepping away
# from the best fit phase-shift in steps of 2pi/phShiftRes (default=1000) and
# calculating the corresponding maximum likelihood; +/-1 sigma uncertainty is when
# the latter drops by +/-0.5 (given that our Likelihood follows a chi2 distribution
# with 1 dof). For debugging/testing/analysis purposes, the user may choose to plot
# the pulse profile (and specify the nbrBins in the plot) and/or the log-likelihood
# around the best fit phase-shift (which should be quite symmetric).
#
# evtFile, timMod, tempModPP, toagtifile, eneLow=0.5, eneHigh=10., toaStart=0, toaEnd=None,
# phShiftRes=1000, nbrBins=15, varyAmps=False, plotPPs=False, plotLLs=False, toaFile='ToAs', timFile=None):
# Input:
# 1- evtFile : barycenter-corrected event file (could be merged along TIME *and* GTIs)
# 2- timMod : *.par timing model (TEMPO2 format is okay)
# 3- tempModPP : *.txt file of the tempolate pulse profile (could be biult with pulseprofile.py)
# 4- toagtifile : *.txt file with ToA properties (could be built with buildtimeintervalsToAs.py)
# 5 eneLow : low energy limit (in keV)
# 6 eneHigh : high energy limit (in keV)
# 7 toaStart : Number of ToA to start from (based on ToAs from toagtifile)
# 8 toaEnd : Number of ToA to end (based on ToAs from toagtifile)
# 9- phShiftRes : Phase-shift step resolution (2*pi/phShiftRes, default = 1000)
# 10- nbrBins : for plotting purposes of pulse profile (default=15)
# 11- varyAmps : vary component amplitude in template pulse profile (i.e., vary pulsed fraction, default=False)
# 12- plotPPs : plot all pulse profiles (only for debugging, default=False)
# 13- plotLLs : plot all logLikelihoods (only for debugging, default=False)
# 14- toaFile : name of output ToA file (default = ToAs(.txt))
# 15- timFile : name of output .tim file compatible with tempo2/PINT - (default = None)
#
# output:
# ToAsTable : pandas table of ToAs properties
#
# To do:
# Add ToA measurement for cauchy and von mises
################################################################################

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
import warnings
import pandas as pd

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
from crimp.phshiftTotimfile import phshiftTotimfile

sys.dont_write_bytecode = True


def measureToAs(evtFile, timMod, tempModPP, toagtifile, eneLow=0.5, eneHigh=10., toaStart=0, toaEnd=None,
                phShiftRes=1000, nbrBins=15, varyAmps=False, plotPPs=False, plotLLs=False, toaFile='ToAs',
                timFile=None):
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
    # Reading and operating on event file
    #####################################
    EF = EvtFileOps(evtFile)
    evtFileKeyWords = EF.readEF()
    # Checking if event file is barycentered 
    if evtFileKeyWords["TIMESYS"] != "TDB":
        warnings.warn(
            "Event file is not barycentered. This script is meant for ToA calculation where all TIMEs should be in TDB. Use with care!",
            stacklevel=2)
    MJDREF = evtFileKeyWords["MJDREF"]

    # Reading TIME column after energy filtering
    dataTP_eneFlt = EF.filtenergyEF(eneLow=eneLow, eneHigh=eneHigh)
    TIME = dataTP_eneFlt['TIME'].to_numpy()
    TIMEMJD = TIME / 86400 + MJDREF

    # Reading ToA intervals from text file
    ######################################
    df_gtiToAParam = pd.read_csv(toagtifile, sep='\s+', comment='#')

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
        BFtempModPP = readPPtemplate(tempModPP)  # BFtempModPP is a dictionary of parameters of best-fit template model

        if BFtempModPP["model"] == 'fourier':
            ToAProp = measureToA_fourier(BFtempModPP, cycleFoldedPhases, ToA_exposure[ii], outFile=ToA_ID,
                                         phShiftRes=phShiftRes, nbrBins=nbrBins, varyAmps=varyAmps, plotPPs=plotPPs,
                                         plotLLs=plotLLs)
        elif BFtempModPP["model"] == 'cauchy':
            cycleFoldedPhases *= (2 * np.pi)
            ToAProp = measureToA_cauchy(BFtempModPP, cycleFoldedPhases, ToA_exposure[ii], outFile=ToA_ID,
                                        phShiftRes=phShiftRes, nbrBins=nbrBins, varyAmps=varyAmps, plotPPs=plotPPs,
                                        plotLLs=plotLLs)
        elif BFtempModPP["model"] == 'vonmises':
            cycleFoldedPhases *= (2 * np.pi)
            ToAProp = measureToA_vonmises(BFtempModPP, cycleFoldedPhases, ToA_exposure[ii], outFile=ToA_ID,
                                          phShiftRes=phShiftRes, nbrBins=nbrBins, varyAmps=varyAmps, plotPPs=plotPPs,
                                          plotLLs=plotLLs)
        else:
            raise Exception(
                'Model {} is not supported yet; fourier, vonmises, cauchy are supported'.format(BFtempModPP["model"]))

        # Calculating Htest power at the epoch frequency
        ################################################
        ephemeridesAtTmid = ephemTmjd(ToA_mid, timMod)
        htestAtTmid = PeriodSearch(TIME_toa * 86400, np.atleast_1d(ephemeridesAtTmid["freqAtTmjd"]), nbrHarm=5)
        htestPowAtTmid = htestAtTmid.htest()[0]

        # Dictionary of ToA properties
        ##############################
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

    # Creating .tim file if specified
    #################################
    if timFile is not None:  # if given convert ToAs.txt to a .tim file
        phshiftTotimfile(toaFile + '.txt', timMod, timFile=timFile, tempModPP=tempModPP, flag='Xray')

    # Plotting Phase residuals of all ToAs
    ######################################
    plotPhaseResiduals(ToA_MJD_all, phShiBF_all, phShiLL_all, phShiUL_all, outFile=toaFile)

    # PANDAS table of ToAs
    ######################
    ToAsTable = pd.read_csv(toaFile + '.txt', sep='\s+', comment='#')

    return ToAsTable


def measureToA_fourier(tempModPP, cycleFoldedPhases, exposureInt, outFile='', phShiftRes=1000, nbrBins=15,
                       varyAmps=False, plotPPs=False, plotLLs=False):
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
    :param plotPPs: boolean flag to plot pulse profile and best fit model (default = False)
    :type plotPPs: bool
    :param plotLLs: boolean flag to plot likelihood curve (default = False)
    :type plotLLs: bool
    :return: ToAPropFourier, a dictionary of ToA properties
    :rtype: dict
    """
    initTempModPPparam = Parameters()  # Initializing an instance of Parameters based on the best-fit template model
    initTempModPPparam.add('norm', tempModPP['norm'], min=0.0, max=np.inf,
                           vary=True)  # Adding the normalization - this is free to vary
    # Number of components in template model
    nbrComp = len(np.array([ww for harmKey, ww in tempModPP.items() if harmKey.startswith('amp_')]))
    for kk in range(1, nbrComp + 1):  # Adding the amplitudes and phases of the harmonics, they are fixed
        initTempModPPparam.add('amp_' + str(kk), tempModPP['amp_' + str(kk)], vary=False)
        initTempModPPparam.add('ph_' + str(kk), tempModPP['ph_' + str(kk)], vary=False)
    initTempModPPparam.add('phShift', 0, vary=True, min=-np.pi, max=np.pi)  # Phase shift - parameter of interest
    initTempModPPparam.add('ampShift', 1, vary=False)

    # Running the extended maximum likelihood
    def unbinnednllfourier(param, xx, exposure):
        return -Fourier(param, xx).loglikelihoodFSnormalized(exposure)

    results_mle_FSNormalized = minimize(unbinnednllfourier, initTempModPPparam, args=(cycleFoldedPhases, exposureInt),
                                        method='nelder', max_nfev=1.0e3, nan_policy='propagate')
    nbrFreeParams = 2
    # In case pulsed fraction should be varied
    if varyAmps is True:
        # We still first fit with fourier amplitudes fixed to 1, this serves to derive a good first guess
        initTempModPPparam.add('ampShift', 1, min=0.0, max=np.inf, vary=True)
        initTempModPPparam.add('phShift', results_mle_FSNormalized.params.valuesdict()['phShift'],
                               vary=True, min=-1.5 * np.pi,
                               max=1.5 * np.pi)  # Phase shift - set to best fit value from above
        initTempModPPparam.add('norm', results_mle_FSNormalized.params.valuesdict()['norm'], min=0.0, max=1000.0,
                               vary=True)  # normalization - set to best fit value from above
        results_mle_FSNormalized = minimize(unbinnednllfourier, initTempModPPparam,
                                            args=(cycleFoldedPhases, exposureInt),
                                            method='nelder', max_nfev=1.0e3, nan_policy='propagate')
        nbrFreeParams = 3  # In this case we varied a third parameters, the harmonics amplitudes

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
        initParam_forErrCalc['phShift'].value = phShiBF - kk * phShiftStep
        initParam_forErrCalc['phShift'].vary = False
        # Performing the fitting
        results_mle_forErrCalc = minimize(unbinnednllfourier, initParam_forErrCalc,
                                          args=(cycleFoldedPhases, exposureInt),
                                          method='nedler', max_nfev=1.0e4, nan_policy='propagate')
        logLikeDist = np.hstack((logLikeDist, -results_mle_forErrCalc.residual))
        phShiftDist = np.hstack((phShiftDist, phShiBF - kk * phShiftStep))
        # New difference 
        chi2diff1sig = LLmax - (-results_mle_forErrCalc.residual)
        # Updating counter
        kk += 1
    phShiBF_LL = (kk * phShiftStep + phShiftStep / 2)

    # Calculating the 1-sigma upper-limit uncertainty
    chi2diff1sig = 0  # Reset initial difference
    kk = 1  # Reset counter for number of steps we shifted phShift
    while chi2diff1sig <= chi2_1sig1dof:
        initParam_forErrCalc['phShift'].value = phShiBF + kk * phShiftStep
        initParam_forErrCalc['phShift'].vary = False
        # Performing the fitting
        results_mle_forErrCalc = minimize(unbinnednllfourier, initParam_forErrCalc,
                                          args=(cycleFoldedPhases, exposureInt),
                                          method='nedler', max_nfev=1.0e4, nan_policy='propagate')
        logLikeDist = np.hstack((logLikeDist, -results_mle_forErrCalc.residual))
        phShiftDist = np.hstack((phShiftDist, phShiBF + kk * phShiftStep))
        # New difference 
        chi2diff1sig = LLmax - (-results_mle_forErrCalc.residual)
        # Updating counter
        kk += 1
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
                      varyAmps=False, plotPPs=False, plotLLs=False):
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
    :param plotPPs: boolean flag to plot pulse profile and best fit model (default = False)
    :type plotPPs: bool
    :param plotLLs: boolean flag to plot likelihood curve (default = False)
    :type plotLLs: bool
    :return: ToAPropFourier, a dictionary of ToA properties
    :rtype: dict
    """
    initTempModPPparam = Parameters()  # Initializing an instance of Parameters based on the best-fit template model

    initTempModPPparam.add('norm', tempModPP['norm'], min=0, max=np.inf,
                           vary=True)  # Adding the normalization - this is free to vary
    # Number of components in template model
    nbrComp = len(np.array([ww for harmKey, ww in tempModPP.items() if harmKey.startswith('amp_')]))
    for kk in range(1, nbrComp + 1):  # Adding the amplitudes, centroids, and widths of the components, they are fixed
        initTempModPPparam.add('amp_' + str(kk), tempModPP['amp_' + str(kk)], vary=False)
        initTempModPPparam.add('cen_' + str(kk), tempModPP['cen_' + str(kk)], vary=False)
        initTempModPPparam.add('wid_' + str(kk), tempModPP['wid_' + str(kk)], vary=False)
    initTempModPPparam.add('phShift', 0, vary=True, min=-1.5*np.pi, max=1.5*np.pi)  # Phase shift - parameter of interest
    initTempModPPparam.add('ampShift', 1, vary=False)

    # Running the extended maximum likelihood
    def unbinnednllcauchy(param, xx, exposure):
        return -WrappedCauchy(param, xx).loglikelihoodCAnormalized(exposure)

    results_mle_CANormalized = minimize(unbinnednllcauchy, initTempModPPparam, args=(cycleFoldedPhases, exposureInt),
                                        method='nelder', max_nfev=1.0e3, nan_policy='propagate')

    nbrFreeParams = 2
    # In case pulsed fraction should be varied
    if varyAmps is True:
        # We still first fit with fourier amplitudes fixed to 1, this serves to derive a good first guess
        initTempModPPparam.add('ampShift', 1, min=0.0, max=np.inf, vary=True)
        initTempModPPparam.add('phShift', results_mle_CANormalized.params.valuesdict()['phShift'],
                               vary=True, min=-1.5 * np.pi,
                               max=1.5 * np.pi)  # Phase shift - set to best fit value from above
        initTempModPPparam.add('norm', results_mle_CANormalized.params.valuesdict()['norm'], min=0.0, max=np.inf,
                               vary=True)  # normalization - set to best fit value from above
        results_mle_CANormalized = minimize(unbinnednllcauchy, initTempModPPparam,
                                            args=(cycleFoldedPhases, exposureInt),
                                            method='nelder', max_nfev=1.0e3, nan_policy='propagate')
        nbrFreeParams = 3  # In this case we varied a third parameters, the harmonics amplitudes

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
                        varyAmps=False, plotPPs=False, plotLLs=False):
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
    :param plotPPs: boolean flag to plot pulse profile and best fit model (default = False)
    :type plotPPs: bool
    :param plotLLs: boolean flag to plot likelihood curve (default = False)
    :type plotLLs: bool
    :return: ToAPropFourier, a dictionary of ToA properties
    :rtype: dict
    """
    initTempModPPparam = Parameters()  # Initializing an instance of Parameters based on the best-fit template model
    initTempModPPparam.add('norm', tempModPP['norm'], min=0.0, max=np.inf,
                           vary=True)  # Adding the normalization - this is free to vary
    # Number of components in template model
    nbrComp = len(np.array([ww for harmKey, ww in tempModPP.items() if harmKey.startswith('amp_')]))
    for kk in range(1, nbrComp + 1):  # Adding the amplitudes, centroids, and widths of the components, they are fixed
        initTempModPPparam.add('amp_' + str(kk), tempModPP['amp_' + str(kk)], vary=False)
        initTempModPPparam.add('cen_' + str(kk), tempModPP['cen_' + str(kk)], vary=False)
        initTempModPPparam.add('wid_' + str(kk), tempModPP['wid_' + str(kk)], vary=False)
    initTempModPPparam.add('phShift', 0, vary=True, min=-1.5*np.pi, max=1.5*np.pi)  # Phase shift - parameter of interest
    initTempModPPparam.add('ampShift', 1, vary=False)

    # Running the extended maximum likelihood
    def unbinnednllvonmises(param, xx, exposure):
        return -VonMises(param, xx).loglikelihoodVMnormalized(exposure)

    results_mle_VMNormalized = minimize(unbinnednllvonmises, initTempModPPparam, args=(cycleFoldedPhases, exposureInt),
                                        method='nelder', max_nfev=1.0e3, nan_policy='propagate')

    nbrFreeParams = 2
    # In case pulsed fraction should be varied
    if varyAmps is True:
        # We still first fit with fourier amplitudes fixed to 1, this serves to derive a good first guess
        initTempModPPparam.add('ampShift', 1, min=0.0, max=np.inf, vary=True)
        initTempModPPparam.add('phShift', results_mle_VMNormalized.params.valuesdict()['phShift'],
                               vary=True, min=-1.5 * np.pi,
                               max=1.5 * np.pi)  # Phase shift - set to best fit value from above
        initTempModPPparam.add('norm', results_mle_VMNormalized.params.valuesdict()['norm'], min=0.0, max=np.inf,
                               vary=True)  # normalization - set to best fit value from above
        results_mle_CANormalized = minimize(unbinnednllvonmises, initTempModPPparam,
                                            args=(cycleFoldedPhases, exposureInt),
                                            method='nelder', max_nfev=1.0e3, nan_policy='propagate')
        nbrFreeParams = 3  # In this case we varied a third parameters, the harmonics amplitudes

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
    parser.add_argument("-el", "--eneLow", help="Low energy filter in event file, default=0.5", type=float, default=0.5)
    parser.add_argument("-eh", "--eneHigh", help="High energy filter in event file, default=10", type=float, default=10)
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

    measureToAs(args.evtFile, args.timMod, args.tempModPP, args.toagtifile, args.eneLow, args.eneHigh, args.toaStart,
                args.toaEnd, args.phShiftRes, args.nbrBins, args.varyAmps, args.plotPPs, args.plotLLs, args.toaFile,
                args.timFile)


if __name__ == '__main__':
    main()
