####################################################################################
# pulseprofile.py is a module that essentially serves to create a template model
# of a high S/N pulse profile. This template model is what is used as anchor to derive
# phase-shifts, i.e., ToAs, from an events fits file (see measureToAs.py).
# The module is composed of a class "PulseProfileFromEventFile" that runs on event files.
# It has two methods, one to simply create a pulse profile of the event file given a
# .par file, and the other to perform the fit. By default, the model is a Fourier series,
# with n=2 harmonics. These are optional arguments, and may be specified by the user.
# So far only a Fourier, a wrapped Gaussian (von Mises), or a wrapped Cauchy (Lorentzian)
# templates are allowed with a generic number of harmonics/components "nbrComp". The
# fitting procedure is done with a maximum likelihood using a gaussian pdf, on a binned
# pulse profile with number of bins = 30 (could be changed as desired). An initial template
# could be provided (default=None) that would serve as initial guess to the fitting
# procedure. The output is a .txt file with the template best fit parameters and
# (optionally) a .pdf file of the profile and the best fit model.
# Lastly when an initial template is provided the user has the option to fix the phases of
# the components, i.e., the peaks of the Gaussian/Lorentzian components or the Fourier
# phases. This is important to maitain absolute timing when deriving ToAs from several
# different instruments (think XTI, XRT, PN, etc.), which require their own template.
#
# Then there is a class called ModelPulseProfile which models a pulse profile that was provided
# as a dictionary with (at least) three keys, 'ppBins', 'countRate', and 'countRateErr';
# numpy arrays with same length.
#
# Finally, there are several other ancilliary functions which calculate pulse properties
# (rms pulsed fraction+), chi2 of best-fit model, and create a plot of pulse profile
#
# Input for 'main' function (ie, PulseProfileFromEventFile.fitPulseProfile):
# 1- evtFile: any event file - could be merged (along **TIME** and **GTI** - the latter is
#                                               used to get an accurate exposure)
# 2- timMod: timing model (.par file)
# 3- eneLow: low energy cutoff in keV (default = 0.5 keV)
# 4- eneHigh: high energy cutoff in keV (default = 10 keV)
# 5- nbrBins: number of bins in pulse profile (default = 30)
# 6- figure: if provided a 'figure'.pdf plot of pulse profile will be created (default=None)
# 7- ppmodel: which model to use (default = fourier, vonmises, cauchy are also allowed)
# 8- nbrComp: number of components in template model (default = 2)
# 9- initTemplateMod: initial template with best-guess for model parameters (default = None)
#                     if this is provided, the "ppmodel" and "nbrComp" inputs will be ignored
#                     and instead read-in from this initial template. The user could simply
#                     run this script with default values to create a template, then modify
#                     it as necessary.
# 10- fixPhases: if True then phases will be fixed to initial value - only applicable if
#                initTemplateMod is provided (default = False)
# 11- templateFile: if provided a 'templateFile'.txt will be created of best fit model parameters (default = None)
# 12- calcPulsedFraction: if True, the pulsed fraction will be calculated.
# 
# output:
# 1- fitResultsDict : dictionary of best fit parameters
# 2- bestFitModel: best fit model as array with same length as pulse profile
# 3- pulsedProperties: RMS pulsed flux and fraction, if calcPulsedFraction=True, None otherwise
#
# To do:
# This module is a bit of a mess, try to simplify
#############################################################

import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import copy

from lmfit import Parameters, minimize

# Custom modules
from crimp.eventfile import EvtFileOps
from crimp.calcphase import calcphase
from crimp.readPPtemplate import readPPtemplate

from crimp.templatemodels import Fourier
from crimp.templatemodels import WrappedCauchy
from crimp.templatemodels import VonMises

from crimp.binphases import binphases

sys.dont_write_bytecode = True


class PulseProfileFromEventFile:
    """
        A class to build and model pulse profile starting from event file

        Attributes
        ----------
        evtFile : str
            name of fits event file
        timMod : str
            timing model, i.e., .par file
        eneLow : float
            lower bound of energy band in keV, default = 0.5
        eneHigh: float
            upper bound of energy band in keV, default = 10
        nbrBins: int
            number of bins in pulse profile, default = 30
        figure: str
            Name of pulse profile plot, default = None

        Methods
        -------
        createPulseProfile():
            create a pulse profile from fits event file given a .par file
        fitPulseProfile(ppmodel = 'fourier', nbrComp = 2, initTemplateMod = None,
                        fixPhases = False, templateFile = None, calcPulsedFraction = False):
            fit pulse profile to model (Fourier, von Mises, Cauchy)
        """

    def __init__(self, evtFile: str, timMod: str, eneLow: float = 0.5, eneHigh: float = 10., nbrBins: int = 30,
                 figure: str = None):
        """
        Constructs all the necessary attributes for the **PulseProfileFromEventFile** object.

        Parameters
        ----------
        evtFile : str
            name of fits event file
        timMod : str
            timing model, i.e., .par file
        eneLow : float
            lower bound of energy band in keV, default = 0.5
        eneHigh: float
            upper bound of energy band in keV, default = 10
        nbrBins: int
            number of bins in pulse profile, default = 30
        figure: str
            Name of pulse profile plot, default = None
        """
        self.evtFile = evtFile
        self.timMod = timMod
        self.eneLow = eneLow
        self.eneHigh = eneHigh
        self.nbrBins = nbrBins
        self.figure = figure

    #################################################################################
    def createpulseprofile(self):
        """
        Method to create a pulse profile
        :return: pulseProfile
        :rtype: dict
        """

        # Reading some event file keywords
        EF = EvtFileOps(self.evtFile)
        evtFileKeyWords = EF.readEF()
        # Checking if event file is barycentered
        if evtFileKeyWords["TIMESYS"] != "TDB":
            raise Warning('Event file is not barycentered, proceed with caution.')
        MJDREF = evtFileKeyWords["MJDREF"]

        # Reading GTIs to calculate an accurate LIVETIME, in case of a merged event file
        gtiList = EF.readGTI()
        LIVETIME = np.sum(gtiList[:, 1] - gtiList[:, 0])

        # Reading TIME column after energy filtering
        dataTP_eneFlt = EF.filtenergyEF(eneLow=self.eneLow, eneHigh=self.eneHigh)
        TIME = dataTP_eneFlt['TIME'].to_numpy()
        timeMJD = TIME / 86400 + MJDREF

        # Calculating PHASE according to timing model
        #############################################
        _, cycleFoldedPhases = calcphase(timeMJD, self.timMod)

        # Creating pulse profile from PHASE
        ###################################
        binnedProfile = binphases(cycleFoldedPhases, self.nbrBins)
        ppBins = binnedProfile["ppBins"]
        ppBinsRange = binnedProfile["ppBinsRange"]
        ctRate = binnedProfile["ctsBins"] / (LIVETIME / self.nbrBins)
        ctRateErr = binnedProfile["ctsBinsErr"] / (LIVETIME / self.nbrBins)
        pulseProfile = {'ppBins': ppBins, 'ppBinsRange': ppBinsRange, 'countRate': ctRate, 'countRateErr': ctRateErr}

        # Creating figure of pulse profile
        ##################################
        if self.figure is not None:
            plotpulseprofile(pulseProfile, outFile=self.figure)

        return pulseProfile

    #################################################################################
    def fitpulseprofile(self, ppmodel: str = 'fourier', nbrComp: int = 2, initTemplateMod: str = None,
                        fixPhases: bool = False, templateFile: str = None, calcPulsedFraction: bool = False):
        """
        Fit pulse profile to a model (Fourier, vonMises, or Cauchy)
        :param ppmodel: name of model to fit data (default = fourier, vonmises, or cauchy)
        :type ppmodel: str
        :param nbrComp: number of components (default = 2)
        :type nbrComp: int
        :param initTemplateMod: name of initial template model for parameter initialization (default = None)
        :type initTemplateMod: str
        :param fixPhases: if **initTemplateMod** is provided, fix phases (default = False)
        :type fixPhases: bool
        :param templateFile: name of output file with best-fit model parameters (default = None)
        :type templateFile: str
        :param calcPulsedFraction: calculate pulsed fraction (default = False)
        :type calcPulsedFraction: bool
        :return: fitResultsDict (dict), bestFitModel (numpy.ndarray), pulsedProperties (dict)
        """
        pulseProfile = self.createpulseprofile()

        if initTemplateMod is None:  # in case an initial template not given, use 'template' keyword to redirect to the appropriate function
            if ppmodel.casefold() == str.casefold('fourier'):
                fitResultsDict, bestFitModel = ModelPulseProfile(pulseProfile, nbrComp).fouriermodel()
            elif ppmodel.casefold() == str.casefold('cauchy'):
                fitResultsDict, bestFitModel = ModelPulseProfile(pulseProfile, nbrComp).cauchymodel()
            elif ppmodel.casefold() == str.casefold('vonmises'):
                fitResultsDict, bestFitModel = ModelPulseProfile(pulseProfile, nbrComp).vonmisesmodel()
            else:
                raise Exception(
                    'Model {} is not supported yet; fourier, vonmises, cauchy are supported'.format(ppmodel))

        else:  # if template is given, continue based on 'model' keyword
            tempModPPparam = readPPtemplate(initTemplateMod)
            if tempModPPparam["model"] == str.casefold('fourier'):
                fitResultsDict, bestFitModel = ModelPulseProfile(pulseProfile, nbrComp, initTemplateMod, fixPhases).fouriermodel()
            elif tempModPPparam["model"] == str.casefold('cauchy'):
                fitResultsDict, bestFitModel = ModelPulseProfile(pulseProfile, nbrComp, initTemplateMod, fixPhases).cauchymodel()
            elif tempModPPparam["model"] == str.casefold('vonmises'):
                fitResultsDict, bestFitModel = ModelPulseProfile(pulseProfile, nbrComp, initTemplateMod, fixPhases).vonmisesmodel()
            else:
                raise Exception('Model {} is not supported yet; fourier, vonmises, cauchy are supported'.format(
                    tempModPPparam["model"]))

        # Write template PP model to .txt file
        ######################################
        if templateFile is not None:
            bestFitTempModPP = templateFile + '.txt'
            # results_mle_FS_dict = results_mle_FS.params.valuesdict()  # converting best-fit results to dictionary

            f = open(bestFitTempModPP, 'w+')
            f.write('model = ' + str(fitResultsDict["model"]) + '\n')
            f.write('norm = ' + str(fitResultsDict["norm"]) + '\n')

            for nn in range(1, nbrComp + 1):
                f.write('amp_' + str(nn) + ' = ' + str(fitResultsDict["amp_" + str(nn)]) + '\n')
                f.write('ph_' + str(nn) + ' = ' + str(fitResultsDict["ph_" + str(nn)]) + '\n')

            f.write('chi2 = ' + str(fitResultsDict["chi2"]) + '\n')
            f.write('dof = ' + str(fitResultsDict["dof"]) + '\n')
            f.write('redchi2 = ' + str(fitResultsDict["redchi2"]) + '\n')
            f.close()

        # Calculating the rms pulsed flux and fraction, along with the harmonics fractions
        ##################################################################################
        if calcPulsedFraction is True:
            pulsedProperties = calcpulseproperties(pulseProfile, nbrComp)
        else:
            pulsedProperties = None

        # Creating figure of pulse profile and best fit model
        #####################################################
        if self.figure is not None:
            plotpulseprofile(pulseProfile, outFile=self.figure, fittedModel=bestFitModel)

        return fitResultsDict, bestFitModel, pulsedProperties


class ModelPulseProfile:
    """
    A class to model a pulse profile, which is provided as a dictionary

    Attributes
    ----------
    pulseProfile : dict
        dictionary of pulse profile; must have 3 keys, ppBins, countRate, and countRateErr
    nbrComp : int
        number of components in model (default = 2)
    initTemplateMod : str
        initial template model name, used as first parameter guess (default = None)
    fixPhases: bool
        Fix phases/peaks to value in initTemplateMod (default = False)

    Methods
    -------
    fitPulseProfileFourier():
        Fit pulse profile to a Fourier series
    fitPulseProfileCauchy():
        Fit pulse profile to a Cauchy curve (wrapped Lorentzian)
    fitPulseProfileVonMises():
        Fit pulse profile to a von Mises curve (wrappged Gaussian)
    """
    def __init__(self, pulseProfile: dict, nbrComp: int = 2, initTemplateMod: str = None, fixPhases: bool = False):
        """
        Constructs all the necessary attributes for the ModelPulseProfile object

        Parameters
        ----------
        pulseProfile : dict
            dictionary of pulse profile; must have 3 keys, ppBins, countRate, and countRateErr
        nbrComp : int
            number of components in model (default = 2)
        initTemplateMod : str
            initial template model name, used as first parameter guess (default = None)
        fixPhases: bool
            Fix phases/peaks to value in initTemplateMod (default = False)
        """
        self.pulseProfile = pulseProfile
        self.nbrComp = nbrComp
        self.initTemplateMod = initTemplateMod
        self.fixPhases = fixPhases

    def fouriermodel(self):
        """
        Function to fit pulse profile to a Fourier series

        returns:
            - fitResultsDict, dictionary of best fit parameters
            - bfModel, numpy array of best fit model at ppBins
        """
        ppBins = self.pulseProfile["ppBins"]
        ctRate = self.pulseProfile["countRate"]
        ctRateErr = self.pulseProfile["countRateErr"]

        # used a few times down below
        template = 'fourier'

        # Fitting pulse profile utilizing mle
        #####################################
        initParams_mle = Parameters()  # Initializing an instance of Parameters
        if self.initTemplateMod is None:  # If no initial template model (i.e., parameter guess) is given, set to default
            # Setting initial guesses to best-guess defaults
            initParams_mle.add('norm', np.mean(ctRate), min=0.0, max=np.inf)
            for kk in range(1, self.nbrComp + 1):
                initParams_mle.add('amp_' + str(kk), 0.1 * np.mean(ctRate))
                initParams_mle.add('ph_' + str(kk), 0)
            initParams_mle.add('phShift', 0,
                               vary=False)  # For consistency, we define a phase shift for the fourier model, though not required here
            initParams_mle.add('ampShift', 1,
                               vary=False)  # For consistency, we define an amplitude shift for the fourier model, though not required here
        else:  # Setting initial guesses to template parameters
            initParams_mle_temp = readPPtemplate(self.initTemplateMod)
            nbrComp = len(np.array([ww for harmKey, ww in initParams_mle_temp.items() if harmKey.startswith('amp_')]))
            initParams_mle.add('norm', initParams_mle_temp['norm'], min=0.0, max=np.inf)
            for kk in range(1, nbrComp + 1):
                initParams_mle.add('amp_' + str(kk), initParams_mle_temp['amp_' + str(kk)])
                initParams_mle.add('ph_' + str(kk), initParams_mle_temp['ph_' + str(kk)])
                if self.fixPhases is True:  # In case component phases should be fixed
                    initParams_mle['ph_' + str(kk)].vary = False
            initParams_mle.add('phShift', 0,
                               vary=False)  # For consistency with our model definition, we define a dummy phase shift
            initParams_mle.add('ampShift', 1,
                               vary=False)  # For consistency with our model definition, we define a dummy phase shift

        # Running the maximum likelihood
        nll = lambda *args: -Fourier.loglikelihoodFS(*args)
        results_mle_FS = minimize(nll, initParams_mle, args=(ppBins, ctRate, ctRateErr), method='nedler',
                                  max_nfev=1.0e6)

        # Calculating the bf Model for the data
        bfModel = Fourier.fourseries(ppBins, results_mle_FS.params)

        # Measuring chi2 and reduced chi2 of template best fit
        ######################################################
        nbrFreeParam = nbrComp * 2 + 1
        chi2Results = measurechi2(self.pulseProfile, bfModel, nbrFreeParam)
        print('Template {} best fit statistics\n chi2 = {} for dof = {}\n Reduced chi2 = {}'.format(template,
                                                                                                    chi2Results["chi2"],
                                                                                                    chi2Results["dof"],
                                                                                                    chi2Results[
                                                                                                        "redchi2"]))

        # Creating dictionary of results
        fitResultsDict = results_mle_FS.params.valuesdict()  # converting best-fit results to dictionary
        fitResultsDict.update(chi2Results, {'model': template})

        return fitResultsDict, bfModel

    def cauchymodel(self):
        """
        Function to fit pulse profile to a Cauchy curve

        returns:
            - fitResultsDict, dictionary of best fit parameters
            - bfModel, numpy array of best fit model at ppBins
        """
        ppBins = self.pulseProfile["ppBins"]
        ctRate = self.pulseProfile["countRate"]
        ctRateErr = self.pulseProfile["countRateErr"]

        # used a few times down below
        template = 'cauchy'

        # Typically the pulse profile is normalized to unity, if that is the case, multiply by 2*pi
        if np.max(ppBins) <= 1:
            ppBins *= 2 * np.pi

        # Fitting pulse profile utilizing mle
        #####################################
        initParams_mle = Parameters()  # Initializing an instance of Parameters
        if self.initTemplateMod is None:  # If no initial template model (i.e., parameter guess) is given, set to default
            # Setting initial guesses to dummy defaults
            initParams_mle.add('norm', np.min(ctRate), min=0.0, max=np.inf)
            for kk in range(1, self.nbrComp + 1):
                initParams_mle.add('amp_' + str(kk), 1.3 * np.min(ctRate), min=0.0, max=np.inf)
                initParams_mle.add('cen_' + str(kk), np.pi, min=0.0, max=2 * np.pi)
                initParams_mle.add('wid_' + str(kk), 1, min=0.0, max=np.inf)
            initParams_mle.add('cenShift', 0,
                               vary=False)  # For consistency with our model definition, we define a dummy phase/centroid shift
            initParams_mle.add('ampShift', 1,
                               vary=False)  # For consistency with our model definition, we define a dummy amplitude shift
        else:  # Setting initial guesses to template parameters
            initParams_mle_temp = readPPtemplate(self.initTemplateMod)
            nbrComp = len(np.array([ww for harmKey, ww in initParams_mle_temp.items() if harmKey.startswith('amp_')]))
            initParams_mle.add('norm', initParams_mle_temp['norm'], min=0.0, max=np.inf)
            for kk in range(1, nbrComp + 1):
                initParams_mle.add('amp_' + str(kk), initParams_mle_temp['amp_' + str(kk)], min=0.0, max=np.inf)
                initParams_mle.add('cen_' + str(kk), initParams_mle_temp['cen_' + str(kk)], min=0.0, max=2 * np.pi)
                initParams_mle.add('wid_' + str(kk), initParams_mle_temp['wid_' + str(kk)], min=0.0, max=np.inf)
                if self.fixPhases is True:  # In case component phases should be fixed
                    initParams_mle['cen_' + str(kk)].vary = False
            initParams_mle.add('cenShift', 0,
                               vary=False)  # For consistency with our model definition, we define a dummy phase/centroid shift
            initParams_mle.add('ampShift', 1,
                               vary=False)  # For consistency with our model definition, we define a dummy amplitude shift

        # Running the maximum likelihood
        nll = lambda *args: -WrappedCauchy.loglikelihoodCA(*args)
        results_mle_CA = minimize(nll, initParams_mle, args=(ppBins, ctRate, ctRateErr), method='nedler',
                                  max_nfev=1.0e6)

        # Calculating the bf Model for the data
        bfModel = WrappedCauchy.wrapcauchy(ppBins, results_mle_CA.params)

        # Measuring chi2 and reduced chi2 of template best fit
        ######################################################
        nbrFreeParam = nbrComp * 2 + 1
        chi2Results = measurechi2(self.pulseProfile, bfModel, nbrFreeParam)
        print('Template {} best fit statistics\n chi2 = {} for dof = {}\n Reduced chi2 = {}'.format(template,
                                                                                                    chi2Results["chi2"],
                                                                                                    chi2Results["dof"],
                                                                                                    chi2Results[
                                                                                                        "redchi2"]))

        # Creating dictionary of results
        fitResultsDict = results_mle_CA.params.valuesdict()  # converting best-fit results to dictionary
        fitResultsDict.update(chi2Results, {'model': template})

        return fitResultsDict, bfModel

    def vonmisesmodel(self):
        """
        Function to fit pulse profile to a von Mises curve

        returns:
            - fitResultsDict, dictionary of best fit parameters
            - bfModel, numpy array of best fit model at ppBins
        """
        ppBins = self.pulseProfile["ppBins"]
        ctRate = self.pulseProfile["countRate"]
        ctRateErr = self.pulseProfile["countRateErr"]

        # used a few times down below
        template = 'vonmises'

        # Typically the pulse profile is normalized to unity, if that is the case, multiply by 2*pi
        if np.max(ppBins) <= 1:
            ppBins *= 2 * np.pi

        # Fitting pulse profile utilizing mle
        #####################################
        initParams_mle = Parameters()  # Initializing an instance of Parameters
        if self.initTemplateMod is None:  # If no initial template model (i.e., parameter guess) is given, set to default
            # Setting initial guesses to dummy defaults
            initParams_mle.add('norm', np.min(ctRate), min=0.0, max=np.inf)
            for kk in range(1, self.nbrComp + 1):
                initParams_mle.add('amp_' + str(kk), 1.3 * np.min(ctRate), min=0.0, max=np.inf)
                initParams_mle.add('cen_' + str(kk), np.pi, min=0.0, max=2 * np.pi)
                initParams_mle.add('wid_' + str(kk), 1, min=0.0, max=np.inf)
            initParams_mle.add('cenShift', 0,
                               vary=False)  # For consistency with our model definition, we define a dummy phase/centroid shift
            initParams_mle.add('ampShift', 1,
                               vary=False)  # For consistency with our model definition, we define a dummy amplitude shift
        else:  # Setting initial guesses to template parameters
            initParams_mle_temp = readPPtemplate(self.initTemplateMod)
            nbrComp = len(np.array([ww for harmKey, ww in initParams_mle_temp.items() if harmKey.startswith('amp_')]))
            initParams_mle.add('norm', initParams_mle_temp['norm'], min=0.0, max=np.inf)
            for kk in range(1, nbrComp + 1):
                initParams_mle.add('amp_' + str(kk), initParams_mle_temp['amp_' + str(kk)], min=0.0, max=np.inf)
                initParams_mle.add('cen_' + str(kk), initParams_mle_temp['cen_' + str(kk)], min=0.0, max=2 * np.pi)
                initParams_mle.add('wid_' + str(kk), initParams_mle_temp['wid_' + str(kk)], min=0.0, max=np.inf)
                if self.fixPhases is True:  # In case component phases should be fixed
                    initParams_mle['cen_' + str(kk)].vary = False
            initParams_mle.add('cenShift', 0,
                               vary=False)  # For consistency with our model definition, we define a dummy phase/centroid shift
            initParams_mle.add('ampShift', 1,
                               vary=False)  # For consistency with our model definition, we define a dummy amplitude shift

        # Running the maximum likelihood
        nll = lambda *args: -VonMises.loglikelihoodVM(*args)
        results_mle_VM = minimize(nll, initParams_mle, args=(ppBins, ctRate, ctRateErr), method='nedler',
                                  max_nfev=1.0e6)

        # Calculating the bf Model for the data
        bfModel = VonMises.vonmises(ppBins, results_mle_VM.params)

        # Measuring chi2 and reduced chi2 of template best fit
        ######################################################
        nbrFreeParam = nbrComp * 2 + 1
        chi2Results = measurechi2(self.pulseProfile, bfModel, nbrFreeParam)
        print('Template {} best fit statistics\n chi2 = {} for dof = {}\n Reduced chi2 = {}'.format(template,
                                                                                                    chi2Results["chi2"],
                                                                                                    chi2Results["dof"],
                                                                                                    chi2Results[
                                                                                                        "redchi2"]))

        # Creating dictionary of results
        fitResultsDict = results_mle_VM.params.valuesdict()  # converting best-fit results to dictionary
        fitResultsDict.update(chi2Results, {'model': template})

        return fitResultsDict, bfModel


#################################################################################
# Measuring chi2 and reduced chi2 of model fit to pulse profile
def measurechi2(pulseProfile, model, nbrFreeParam):
    """
    Method to measure the chi2 of best fit model to pulse profile
    :param pulseProfile: dictionary of pulse profile; must have 3 keys, ppBins, countRate, and countRateErr
    :type pulseProfile: dict
    :param model: array of best fit model at ppBins
    :type model: numpy.ndarray
    :param nbrFreeParam: number of free parameters in the model
    :type nbrFreeParam: int
    :return: chi2Results
    :rtype: dictionary of {'chi2', 'dof', 'redchi2'}
    """
    ctRate = pulseProfile["countRate"]
    ctRateErr = pulseProfile["countRateErr"]

    chi2 = np.sum(((ctRate - model) ** 2) / (ctRateErr ** 2))
    dof = len(ctRate) - nbrFreeParam
    redchi2 = chi2 / dof
    chi2Results = {'chi2': chi2, 'dof': dof, 'redchi2': redchi2}

    return chi2Results


#################################################################################
# Calculating the rms pulsed flux, pulsed fraction, and harmonics pulsed fraction
def calcpulseproperties(pulseProfile, nbrComp):
    """
    Calculate rms pulsed fraction and pulsed flux (total and for individual harmonics)
    :param pulseProfile: dictionary of pulse profile; keys are {ppBins, countRate, countRateErr}
    :type pulseProfile: dict
    :param nbrComp: number of Fourier components
    :type nbrComp: int
    :return: pulseProperties
    :rtype: dict
    """
    ppBins = pulseProfile["ppBins"]
    ctRate = pulseProfile["countRate"]
    ctRateErr = pulseProfile["countRateErr"]

    FrmsHarms = np.zeros(nbrComp)
    for kk in range(1, nbrComp + 1):
        N = len(ppBins)

        ak = (1 / N) * np.sum(ctRate * np.cos(kk * 2 * np.pi * ppBins))
        sak = (1 / N ** 2) * np.sum(ctRateErr ** 2 * np.cos(kk * 2 * np.pi * ppBins) ** 2)
        bk = (1 / N) * np.sum(ctRate * np.sin(kk * 2 * np.pi * ppBins))
        sbk = (1 / N ** 2) * np.sum(ctRateErr ** 2 * np.sin(kk * 2 * np.pi * ppBins) ** 2)

        Frms1harm = (ak ** 2 + bk ** 2) - (sak ** 2 + sbk ** 2)

        FrmsHarms[kk - 1] = Frms1harm

    Frms = np.sqrt(np.sum(FrmsHarms) * 2)
    PFrms = Frms / np.mean(ctRate)

    pulseProperties = {'pulsedFlux': Frms, 'pulsedFraction': PFrms, 'harmonicPulsedFractions': FrmsHarms}
    pulsePropertiesErr = calcuncertaintypulseproperties(pulseProfile, nbrComp)
    pulseProperties.update(pulsePropertiesErr)

    return pulseProperties


def calcuncertaintypulseproperties(pulseProfile, nbrComp):
    """
    Calculate uncertainty on rms pulsed fraction and pulsed flux (total and for individual harmonics)
    In reality this function never need to be invoked directly. It will be called by calcPulseProperties, which returns
    the rms pulsed fraction, pulsed flux, and their uncertainties (total and for individual harmonics)
    :param pulseProfile: dictionary of pulse profile; keys are {ppBins, countRate, countRateErr}
    :type pulseProfile: dict
    :param nbrComp: number of Fourier components
    :type nbrComp: int
    :return: pulsePropertiesErr
    :rtype: dict
    """
    # Error calculation is done through a simple monte carlo simulation
    nbrOfSimulations = 1000  # simulating 1000 pulse profiles
    simulatedPulseProfile = copy.deepcopy(pulseProfile)

    simulatedPulsedFlux = np.zeros(nbrOfSimulations)
    simulatedPulsedFraction = np.zeros(nbrOfSimulations)
    simulatedHarmonicPulsedFraction = np.zeros((nbrOfSimulations, nbrComp))
    for jj in range(nbrOfSimulations):
        simulatedPulseProfile["countRate"] = np.random.normal(pulseProfile["countRate"],
                                                              pulseProfile["countRateErr"], size=None)
        simulatedPulsedProperties = calcpulseproperties(simulatedPulseProfile, nbrComp)
        simulatedPulsedFlux[jj] = simulatedPulsedProperties["pulsedFlux"]
        simulatedPulsedFraction[jj] = simulatedPulsedProperties["pulsedFraction"]
        simulatedHarmonicPulsedFraction[jj, :] = simulatedPulsedProperties["harmonicPulsedFractions"]

    _, FrmsErr = norm.fit(simulatedPulsedFlux)
    _, PFrmsErr = norm.fit(simulatedPulsedFraction)

    FrmsHarmsErr = np.zeros(nbrComp)
    for kk in range(nbrComp):
        _, FrmsHarmsErr[kk] = norm.fit(simulatedHarmonicPulsedFraction[:, kk])

    pulsePropertiesErr = {'pulsedFluxErr': FrmsErr, 'pulsedFractionErr': PFrmsErr,
                          'harmonicPulsedFractionsErr': FrmsHarmsErr}

    return pulsePropertiesErr


def plotpulseprofile(pulseProfile, outFile='pulseprof', fittedModel=None):
    """
    Function to make a plot of pulse profile
    :param pulseProfile: dictionary of pulse profile; keys are {ppBins, countRate, countRateErr}
    :type pulseProfile: dict
    :param outFile: name of plot (default = 'pulseprof'.pdf)
    :type outFile: str
    :param fittedModel: array of best fit model at ppBins
    :type fittedModel: numpy.ndarray
    """
    ppBins = pulseProfile["ppBins"]
    ctRate = pulseProfile["countRate"]
    ctRateErr = pulseProfile["countRateErr"]

    # creating a two-cycles for clarity
    ctRate_plt = np.append(ctRate, ctRate)
    ctRateErr_plt = np.append(ctRateErr, ctRateErr)
    ppBins_plt = np.append(ppBins, ppBins + 1.0)

    fig, ax1 = plt.subplots(1, figsize=(6, 4.0), dpi=80, facecolor='w', edgecolor='k')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlabel(r'$\,\mathrm{Phase\,(cycles)}$', fontsize=12)
    ax1.set_ylabel(r'$\,\mathrm{Rate\,(counts\,s^{-1})}$', fontsize=12)
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.yaxis.offsetText.set_fontsize(12)

    plt.step(ppBins_plt, ctRate_plt, 'k+-', where='mid')
    plt.errorbar(ppBins_plt, ctRate_plt, yerr=ctRateErr_plt, fmt='ok')

    if fittedModel is not None:  # Creating the best fit model for the data
        fittedModel_plt = np.append(fittedModel, fittedModel)
        plt.plot(ppBins_plt, fittedModel_plt, 'r-', linewidth=2.0)

    # saving PP and template best fit
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.5)
        ax1.tick_params(width=1.5)

    fig.tight_layout()

    outPlot = outFile + '.pdf'
    fig.savefig(outPlot, format='pdf', dpi=1000)

    return


def main():
    """
    Main function which could be run as a script, this is PulseProfileFromEventFile.fitpulseprofile
    """
    parser = argparse.ArgumentParser(description="Build and fit pulse profile from event file")
    parser.add_argument("evtFile", help="Event file", type=str)
    parser.add_argument("timMod", help="Timing model (.par file)", type=str)
    parser.add_argument("-el", "--eneLow", help="lower energy cut, default=0.5 keV", type=float, default=0.5)
    parser.add_argument("-eh", "--eneHigh", help="high energy cut, default=10 keV", type=float, default=10)
    parser.add_argument("-nb", "--nbrBins", help="Number of bins for visualization purposes only, default = 15",
                        type=int, default=15)
    parser.add_argument("-fg", "--figure",
                        help="If supplied, a plot of the pulse profile will be produced, 'figure'.pdf", type=str,
                        default=None)
    parser.add_argument("-pm", "--ppmodel",
                        help="Model for fitting (default = 'fourier'; 'vonmises' and 'cauchy' are also supported)",
                        type=str, default='fourier')
    parser.add_argument("-nc", "--nbrComp",
                        help="Number of components in template (nbr of harmonics or nbr of gaussians), default = 2",
                        type=int, default=2)
    parser.add_argument("-it", "--initTemplateMod",
                        help="Initial template model parameters. In this case, keywords template, and nbrComp are ignored",
                        type=str, default=None)
    parser.add_argument("-fp", "--fixPhases",
                        help="Flag to fix phases in input initial template model (initTemplateMod), default = False",
                        type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-tf", "--templateFile", help="'Output' .txt file for best-fit model)", type=str,
                        default='None')
    parser.add_argument("-cp", "--calcPulsedFraction",
                        help="Flag to calculate RMS pulsed fraction of pulse profile, default = False", type=bool,
                        default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    ppfromevt = PulseProfileFromEventFile(args.evtFile, args.timMod, args.eneLow, args.eneHigh, args.nbrBins,
                                          args.figure)
    ppfromevt.fitpulseprofile(args.ppmodel, args.nbrComp, args.initTemplateMod, args.fixPhases, args.templateFile,
                              args.calcPulsedFraction)


##############
# End Script #
##############

if __name__ == '__main__':
    main()