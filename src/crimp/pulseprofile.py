"""
pulseprofile.py is a module that essentially serves to create a template model
of a high S/N pulse profile. This template model is what is used as anchor to derive
phase-shifts, i.e., ToAs, from an events fits file (see measureToAs.py).
The module is composed of a class "PulseProfileFromEventFile" that runs on event files.
It has two methods, one to simply create a pulse profile of the event file given a
.par file, and the other to perform the fit. By default, the model is a Fourier series,
with n=2 harmonics. These are optional arguments, and may be specified by the user.
So far only a Fourier, a wrapped Gaussian (von Mises), or a wrapped Cauchy (Lorentzian)
templates are allowed with a generic number of harmonics/components "nbrComp". The
fitting procedure is done with a maximum likelihood using a gaussian pdf, on a binned
pulse profile with number of bins = 30 (could be changed as desired). An initial template
could be provided (default=None) that would serve as initial guess to the fitting
procedure. The output is a .txt file with the template best fit parameters and
(optionally) a .pdf file of the binned pulse profile and the best fit model.
Lastly when an initial template is provided the user has the option to fix the phases of
the components, i.e., the peaks of the Gaussian/Lorentzian components or the Fourier
phases. This is important to maitain *absolute* timing when deriving ToAs from several
different instruments (think XTI, XRT, PN, etc.), which require their own template.
A .log file is also created summarizing the run.

Best run through the command line as "templatepulseprofile"

To do:
This module is a bit messy, try to simplify
"""

import argparse
import sys
import logging

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

        Methods
        -------
        createPulseProfile():
            create a pulse profile from fits event file given a .par file
        fitPulseProfile(ppmodel = 'fourier', nbrComp = 2, initTemplateMod = None,
                        fixPhases = False, templateFile = None, calcPulsedFraction = False):
            fit pulse profile to model (Fourier, von Mises, Cauchy)
        """

    def __init__(self, evtFile: str, timMod: str, eneLow: float = 0.5, eneHigh: float = 10., nbrBins: int = 30):
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
        """
        self.evtFile = evtFile
        self.timMod = timMod
        self.eneLow = eneLow
        self.eneHigh = eneHigh
        self.nbrBins = nbrBins

    #################################################################################
    def createpulseprofile(self):
        """
        Method to create a pulse profile
        :return: pulseProfile
        :rtype: dict
        """

        # Reading event file and GTIs to calculate an accurate LIVETIME
        ###############################################################
        EF = EvtFileOps(self.evtFile)
        _, gtiList = EF.readGTI()
        LIVETIME = np.sum(gtiList[:, 1] - gtiList[:, 0]) * 86400  # converting to seconds
        # Reading TIME column after energy filtering
        dataTP_eneFlt = EF.build_time_energy_df().filtenergy(eneLow=self.eneLow, eneHigh=self.eneHigh)
        timeMJD = dataTP_eneFlt.time_energy_df['TIME'].to_numpy()

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

        return pulseProfile

    #################################################################################
    def fitpulseprofile(self, ppmodel: str = 'fourier', nbrComp: int = 2, initTemplateMod: str = None,
                        fixPhases: bool = False, figure: str = None, templateFile: str = None,
                        calcPulsedFraction: bool = False):
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
        :param figure: Name of pulse profile plot, default = None
        :type figure: str
        :param templateFile: name of output file with best-fit model parameters (default = None)
        :type templateFile: str
        :param calcPulsedFraction: calculate pulsed fraction (default = False)
        :type calcPulsedFraction: bool
        :return: fitResultsDict (dict), bestFitModel (numpy.ndarray), pulsedProperties (dict)
        """
        logfile = 'logfile' if templateFile is None else templateFile
        fileHandler = logging.FileHandler(logfile + '.log', mode='w')
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(logging.INFO)
        logger.addHandler(fileHandler)

        logger.info('\n Running method fitpulseprofile with input parameters: '
                    '\n evtFile: ' + str(self.evtFile) +
                    '\n Timing model: ' + str(self.timMod) +
                    '\n eneLow: ' + str(self.eneLow) +
                    '\n eneHigh: ' + str(self.eneHigh) +
                    '\n nbrBins: ' + str(self.nbrBins) +
                    '\n ppmodel: ' + str(ppmodel) +
                    '\n nbrComp: ' + str(nbrComp) +
                    '\n initTemplateMod: ' + str(initTemplateMod) +
                    '\n fixPhases: ' + str(fixPhases) +
                    '\n figure: ' + str(figure) + '(.pdf)' +
                    '\n templateFile: ' + str(templateFile) + '(.txt)' +
                    '\n calcPulsedFraction: ' + str(calcPulsedFraction) + '\n')

        pulseProfile = self.createpulseprofile()

        if initTemplateMod is None:  # in case an initial template not given, use 'template' keyword to redirect to the appropriate function
            logger.info('\n No initial template file provided'
                        '\n Fitting to user chosen model : ' + ppmodel +
                        '\n using number of components : ' + str(nbrComp))
            if ppmodel.casefold() == 'fourier':
                fitResultsDict, bestFitModel = ModelPulseProfile(pulseProfile, nbrComp).fouriermodel()
            elif ppmodel.casefold() == 'cauchy':
                pulseProfile["ppBins"] *= 2 * np.pi  # need to be normalized to 2pi
                fitResultsDict, bestFitModel = ModelPulseProfile(pulseProfile, nbrComp).cauchymodel()
            elif ppmodel.casefold() == 'vonmises':
                pulseProfile["ppBins"] *= 2 * np.pi  # need to be normalized to 2pi
                fitResultsDict, bestFitModel = ModelPulseProfile(pulseProfile, nbrComp).vonmisesmodel()
            else:
                logger.error('Model {} is not supported yet; fourier, vonmises, cauchy are supported'.format(ppmodel))

        else:  # if template is given, continue based on 'model' keyword
            logger.info('\n Initial template file provided : ' + initTemplateMod +
                        '\n Using these model parameters as starting point. '
                        '\n Ignoring input keywords ppmodel = ' + ppmodel + ' and nbrComp = ' + str(nbrComp))
            tempModPPparam = readPPtemplate(initTemplateMod)
            ppmodel = tempModPPparam["model"]
            nbrComp = tempModPPparam["nbrComp"]
            if ppmodel.casefold() == 'fourier':
                fitResultsDict, bestFitModel = ModelPulseProfile(pulseProfile, nbrComp, initTemplateMod,
                                                                 fixPhases).fouriermodel()
            elif ppmodel.casefold() == 'cauchy':
                pulseProfile["ppBins"] *= 2 * np.pi  # need to be normalized to 2pi
                fitResultsDict, bestFitModel = ModelPulseProfile(pulseProfile, nbrComp, initTemplateMod,
                                                                 fixPhases).cauchymodel()
            elif ppmodel.casefold() == 'vonmises':
                pulseProfile["ppBins"] *= 2 * np.pi  # need to be normalized to 2pi
                fitResultsDict, bestFitModel = ModelPulseProfile(pulseProfile, nbrComp, initTemplateMod,
                                                                 fixPhases).vonmisesmodel()
            else:
                logger.error('Model {} is not supported yet; fourier, vonmises, cauchy are supported'.format(
                    tempModPPparam["model"]))

        # Write template PP model to .txt file
        ######################################
        if templateFile is not None:

            writetemplatefile(templateFile, fitResultsDict)

            logger.info('\n chi2 = ' + str(fitResultsDict["chi2"]) + '\n dof = ' + str(fitResultsDict["dof"]) +
                        '\n redchi2 = ' + str(fitResultsDict["redchi2"]) + '\n')

            logger.info('\n Created best fit template file : ' + templateFile + '.txt \n')
        else:
            logger.info('\n No template file created: templateFile is None\n')

        # Calculating the rms pulsed flux and fraction, along with the harmonics fractions
        ##################################################################################
        if calcPulsedFraction is True and ppmodel.casefold() == 'fourier':
            pulsedProperties = calcpulseproperties(pulseProfile, nbrComp)
            pulsePropertiesErr = calcuncertaintypulseproperties(pulseProfile, nbrComp)
            pulsedProperties.update(pulsePropertiesErr)
        elif calcPulsedFraction is True and ppmodel.casefold() in ('cauchy', 'vonmises'):
            logger.warning('Cannot calculate rms pulsed fraction for ' + ppmodel.casefold() +
                           '\n Setting pulsedProperties to None')
            pulsedProperties = None
        else:
            pulsedProperties = None

        # Creating figure of pulse profile and best fit model
        #####################################################
        if figure is not None:
            plotpulseprofile(pulseProfile, outFile=figure, fittedModel=bestFitModel)
            logger.info('\n Created figure of pulse profile and best-fit template : ' + figure + '.pdf \n')
        else:
            logger.info('\n No figure file provided/created\n')

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
            initParams_mle.add('norm', np.mean(ctRate), min=0.0, max=1.0e6)
            for kk in range(1, self.nbrComp + 1):
                initParams_mle.add('amp_' + str(kk), 0.1 * np.mean(ctRate))
                initParams_mle.add('ph_' + str(kk), 0)
            initParams_mle.add('phShift', 0,
                               vary=False)  # For consistency, we define a phase shift for the fourier model, though not required here
            initParams_mle.add('ampShift', 1,
                               vary=False)  # For consistency, we define an amplitude shift for the fourier model, though not required here
            # Number of free parameterd
            nbrFreeParams = 2 * self.nbrComp + 1

        else:  # Setting initial guesses to template parameters
            initParams_mle_temp = readPPtemplate(self.initTemplateMod)
            self.nbrComp = initParams_mle_temp["nbrComp"]
            initParams_mle.add('norm', initParams_mle_temp['norm']['value'], min=0.0, max=1.0e6,
                               vary=initParams_mle_temp['norm']['vary'])

            # setting up the number of free parameters
            if initParams_mle_temp['norm']['vary']:
                nbrFreeParams = 1
            else:
                nbrFreeParams = 0

            for kk in range(1, self.nbrComp + 1):
                initParams_mle.add('amp_' + str(kk), initParams_mle_temp['amp_' + str(kk)]['value'],
                                   vary=initParams_mle_temp['amp_' + str(kk)]['vary'])
                initParams_mle.add('ph_' + str(kk), initParams_mle_temp['ph_' + str(kk)]['value'],
                                   vary=initParams_mle_temp['ph_' + str(kk)]['vary'])

                if self.fixPhases is True:  # In case component phases should be fixed
                    initParams_mle['ph_' + str(kk)].vary = False

                # properly dealing with number of free parameters
                for key in ('amp_' + str(kk), 'ph_' + str(kk)):
                    if initParams_mle_temp[key]['vary']:
                        nbrFreeParams += 1
            initParams_mle.add('phShift', 0,
                               vary=False)  # For consistency with our model definition, we define a dummy phase shift
            initParams_mle.add('ampShift', 1,
                               vary=False)  # For consistency with our model definition, we define a dummy amp shift

        # Running the maximum likelihood
        def binnednllfourier(param, xx, yy, yyErr):
            return -Fourier(param, xx).loglikelihoodFS(yy, yyErr)

        results_mle_FS = minimize(binnednllfourier, initParams_mle, args=(ppBins, ctRate, ctRateErr), method='nedler',
                                  max_nfev=1.0e6)

        # Calculating the bf Model for the data
        bfModel = Fourier(results_mle_FS.params, ppBins).fourseries()

        # Measuring chi2 and reduced chi2 of template best fit
        ######################################################
        chi2Results = measurechi2(self.pulseProfile, bfModel, nbrFreeParams)
        print('Template {} best fit statistics\n chi2 = {} for dof = {}\n Reduced chi2 = {}'.format(template,
                                                                                                    chi2Results["chi2"],
                                                                                                    chi2Results["dof"],
                                                                                                    chi2Results[
                                                                                                        "redchi2"]))

        # Creating dictionary of results
        fitResultsDict = results_mle_FS.params.valuesdict()  # converting best-fit results to dictionary
        fitResultsDict.update(chi2Results)
        fitResultsDict.update({'model': template})

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

        # Fitting pulse profile utilizing mle
        #####################################
        initParams_mle = Parameters()  # Initializing an instance of Parameters
        if self.initTemplateMod is None:  # If no initial template model (i.e., parameter guess) is given, set to default
            # Setting initial guesses to dummy defaults
            initParams_mle.add('norm', np.min(ctRate), min=0.0, max=np.max(ctRate))
            for kk in range(1, self.nbrComp + 1):
                initParams_mle.add('amp_' + str(kk), 1.3 * np.min(ctRate), min=0.0, max=np.inf)
                initParams_mle.add('cen_' + str(kk), np.pi, min=0.0, max=2 * np.pi)
                initParams_mle.add('wid_' + str(kk), 1, min=0.0, max=np.inf)
            initParams_mle.add('phShift', 0,
                               vary=False)  # For consistency with our model definition, we define a dummy phase/centroid shift
            initParams_mle.add('ampShift', 1,
                               vary=False)  # For consistency with our model definition, we define a dummy amplitude shift
            # Number of free parameterd
            nbrFreeParams = 2 * self.nbrComp + 1

        else:  # Setting initial guesses to template parameters
            initParams_mle_temp = readPPtemplate(self.initTemplateMod)
            self.nbrComp = initParams_mle_temp["nbrComp"]
            initParams_mle.add('norm', initParams_mle_temp['norm']['value'], min=0.0, max=np.max(ctRate),
                               vary=initParams_mle_temp['norm']['vary'])

            # setting up the number of free parameters
            if initParams_mle_temp['norm']['vary']:
                nbrFreeParams = 1
            else:
                nbrFreeParams = 0

            for kk in range(1, self.nbrComp + 1):
                initParams_mle.add('amp_' + str(kk), initParams_mle_temp['amp_' + str(kk)]['value'], min=0.0, max=np.inf,
                                   vary=initParams_mle_temp['amp_' + str(kk)]['vary'])
                initParams_mle.add('cen_' + str(kk), initParams_mle_temp['cen_' + str(kk)]['value'], min=0.0, max=2 * np.pi,
                                   vary=initParams_mle_temp['cen_' + str(kk)]['vary'])
                initParams_mle.add('wid_' + str(kk), initParams_mle_temp['wid_' + str(kk)]['value'], min=0.0, max=np.inf,
                                   vary=initParams_mle_temp['wid_' + str(kk)]['vary'])

                if self.fixPhases is True:  # In case component phases should be fixed
                    initParams_mle['cen_' + str(kk)].vary = False

                # properly dealing with number of free parameters
                for key in ('amp_' + str(kk), 'cen_' + str(kk), 'wid_' + str(kk)):
                    if initParams_mle_temp[key]['vary']:
                        nbrFreeParams += 1

            initParams_mle.add('phShift', 0,
                               vary=False)  # For consistency with our model definition, we define a dummy phase/centroid shift
            initParams_mle.add('ampShift', 1,
                               vary=False)  # For consistency with our model definition, we define a dummy amplitude shift

        # Running the maximum likelihood
        def binnednllcauchy(param, xx, yy, yyErr):
            return -WrappedCauchy(param, xx).loglikelihoodCA(yy, yyErr)

        results_mle_CA = minimize(binnednllcauchy, initParams_mle, args=(ppBins, ctRate, ctRateErr),
                                  method='nedler', max_nfev=1.0e6, nan_policy='propagate')

        # Calculating the bf Model for the data
        bfModel = WrappedCauchy(results_mle_CA.params, ppBins).wrapcauchy()

        # Measuring chi2 and reduced chi2 of template best fit
        ######################################################
        chi2Results = measurechi2(self.pulseProfile, bfModel, nbrFreeParams)
        print('Template {} best fit statistics\n chi2 = {} for dof = {}\n Reduced chi2 = {}'.format(template,
                                                                                                    chi2Results["chi2"],
                                                                                                    chi2Results["dof"],
                                                                                                    chi2Results[
                                                                                                        "redchi2"]))

        # Creating dictionary of results
        fitResultsDict = results_mle_CA.params.valuesdict()  # converting best-fit results to dictionary
        fitResultsDict.update(chi2Results)
        fitResultsDict.update({'model': template})

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

        # Fitting pulse profile utilizing mle
        #####################################
        initParams_mle = Parameters()  # Initializing an instance of Parameters
        if self.initTemplateMod is None:  # If no initial template model (i.e., parameter guess) is given, set to default
            # Setting initial guesses to dummy defaults
            initParams_mle.add('norm', np.min(ctRate), min=0, max=np.max(ctRate), vary=True)
            for kk in range(1, self.nbrComp + 1):
                initParams_mle.add('amp_' + str(kk), 1.3 * np.min(ctRate), min=0.0, max=np.inf)
                initParams_mle.add('cen_' + str(kk), np.pi, min=0.0, max=2 * np.pi)
                initParams_mle.add('wid_' + str(kk), 1, min=0.0, max=np.inf)
            initParams_mle.add('phShift', 0,
                               vary=False)  # For consistency with our model definition, we define a dummy phase/centroid shift
            initParams_mle.add('ampShift', 1,
                               vary=False)  # For consistency with our model definition, we define a dummy amplitude shift
            # Number of free parameterd
            nbrFreeParams = 2 * self.nbrComp + 1

        else:  # Setting initial guesses to template parameters
            initParams_mle_temp = readPPtemplate(self.initTemplateMod)
            self.nbrComp = initParams_mle_temp["nbrComp"]
            initParams_mle.add('norm', initParams_mle_temp['norm']['value'], min=0.0, max=np.max(ctRate),
                               vary=initParams_mle_temp['norm']['vary'])
            # setting up the number of free parameters
            if initParams_mle_temp['norm']['vary']:
                nbrFreeParams = 1
            else:
                nbrFreeParams = 0

            for kk in range(1, self.nbrComp + 1):
                initParams_mle.add('amp_' + str(kk), initParams_mle_temp['amp_' + str(kk)]['value'], min=0.0,
                                   max=np.inf,
                                   vary=initParams_mle_temp['amp_' + str(kk)]['vary'])
                initParams_mle.add('cen_' + str(kk), initParams_mle_temp['cen_' + str(kk)]['value'], min=0.0,
                                   max=2 * np.pi,
                                   vary=initParams_mle_temp['cen_' + str(kk)]['vary'])
                initParams_mle.add('wid_' + str(kk), initParams_mle_temp['wid_' + str(kk)]['value'], min=0.0,
                                   max=np.inf,
                                   vary=initParams_mle_temp['wid_' + str(kk)]['vary'])

                if self.fixPhases is True:  # In case component phases should be fixed
                    initParams_mle['cen_' + str(kk)].vary = False

                # properly dealing with number of free parameters
                for key in ('amp_' + str(kk), 'cen_' + str(kk), 'wid_' + str(kk)):
                    if initParams_mle_temp[key]['vary']:
                        nbrFreeParams += 1

            initParams_mle.add('phShift', 0,
                               vary=False)  # For consistency with our model definition, we define a dummy phase/centroid shift
            initParams_mle.add('ampShift', 1,
                               vary=False)  # For consistency with our model definition, we define a dummy amplitude shift

        def binnednllvonmises(param, xx, yy, yyErr):
            return -VonMises(param, xx).loglikelihoodVM(yy, yyErr)

        # Running the maximum likelihood
        results_mle_VM = minimize(binnednllvonmises, initParams_mle, args=(ppBins, ctRate, ctRateErr),
                                  method='nedler', max_nfev=1.0e6, nan_policy='propagate')

        # Calculating the bf Model for the data
        bfModel = VonMises(results_mle_VM.params, ppBins).vonmises()

        # Measuring chi2 and reduced chi2 of template best fit
        ######################################################
        chi2Results = measurechi2(self.pulseProfile, bfModel, nbrFreeParams)
        print('Template {} best fit statistics\n chi2 = {} for dof = {}\n Reduced chi2 = {}'.format(template,
                                                                                                    chi2Results["chi2"],
                                                                                                    chi2Results["dof"],
                                                                                                    chi2Results[
                                                                                                        "redchi2"]))

        # Creating dictionary of results
        fitResultsDict = results_mle_VM.params.valuesdict()  # converting best-fit results to dictionary
        fitResultsDict.update(chi2Results)
        fitResultsDict.update({'model': template})

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

    return pulseProperties


def calcuncertaintypulseproperties(pulseProfile, nbrComp):
    """
    Calculate uncertainty on rms pulsed fraction and pulsed flux (total and for individual harmonics)
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

    if np.max(pulseProfile["ppBins"]) > 1:  # Plotting a second cycle for cauchy or vonmises (cycle = 2*pi)
        secondCycle = 2 * np.pi
    else:  # Plotting a second cycle for fourier (cycle = 1)
        secondCycle = 1

    # creating a two-cycles for clarity
    ctRate_plt = np.append(ctRate, ctRate)
    ctRateErr_plt = np.append(ctRateErr, ctRateErr)
    ppBins_plt = np.append(ppBins, ppBins + secondCycle)

    fig, ax1 = plt.subplots(1, figsize=(6, 4.0), dpi=80, facecolor='w', edgecolor='k')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlabel(r'$\,\mathrm{Phase\,(cycles)}$', fontsize=12)
    ax1.set_ylabel(r'$\,\mathrm{Rate\,(counts\,s^{-1})}$', fontsize=12)
    ax1.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.yaxis.offsetText.set_fontsize(12)

    ax1.step(ppBins_plt, ctRate_plt, 'k+-', where='mid')
    ax1.errorbar(ppBins_plt, ctRate_plt, yerr=ctRateErr_plt, fmt='ok')

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


def writetemplatefile(templateFile, fitResultsDict):
    """
    Write the results of a template fit to a .txt file
    :param templateFile:
    :type templateFile: str
    :param fitResultsDict:
    :type fitResultsDict: dict
    """
    bestFitTempModPP = templateFile + '.txt'
    nbrComp = len(np.array([ww for modelKey, ww in fitResultsDict.items() if modelKey.startswith('amp_')]))
    ppmodel = fitResultsDict["model"]

    f = open(bestFitTempModPP, 'w+')
    f.write('model ' + str(fitResultsDict["model"]) + '\n')
    f.write('norm ' + str(fitResultsDict["norm"]) + ' vary True \n')

    for nn in range(1, nbrComp + 1):
        f.write('amp_' + str(nn) + ' ' + str(fitResultsDict["amp_" + str(nn)]) + ' vary True \n')
        if ppmodel.casefold() == 'fourier':
            f.write('ph_' + str(nn) + ' ' + str(fitResultsDict["ph_" + str(nn)]) + ' vary True \n')
        if ppmodel.casefold() == 'vonmises' or ppmodel.casefold() == 'cauchy':
            f.write('cen_' + str(nn) + ' ' + str(fitResultsDict["cen_" + str(nn)]) + ' vary True \n')
            f.write('wid_' + str(nn) + ' ' + str(fitResultsDict["wid_" + str(nn)]) + ' vary True \n')

    f.write('chi2 ' + str(fitResultsDict["chi2"]) + '\n')
    f.write('dof ' + str(fitResultsDict["dof"]) + '\n')
    f.write('redchi2 ' + str(fitResultsDict["redchi2"]) + '\n')
    f.close()

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
    parser.add_argument("-fg", "--figure",
                        help="If supplied, a plot of the pulse profile will be produced, 'figure'.pdf", type=str,
                        default=None)
    parser.add_argument("-tf", "--templateFile", help="'Output' .txt file for best-fit model)", type=str,
                        default=None)
    parser.add_argument("-cp", "--calcPulsedFraction",
                        help="Flag to calculate RMS pulsed fraction of pulse profile, default = False", type=bool,
                        default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    ppfromevt = PulseProfileFromEventFile(args.evtFile, args.timMod, args.eneLow, args.eneHigh, args.nbrBins)
    ppfromevt.fitpulseprofile(args.ppmodel, args.nbrComp, args.initTemplateMod, args.fixPhases, args.figure,
                              args.templateFile, args.calcPulsedFraction)


if __name__ == '__main__':
    main()
