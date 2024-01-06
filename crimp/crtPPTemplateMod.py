####################################################################################
# crtPPTemplateMod.py is a module to create a template model of a high S/N pulse
# profile. This template model is what is used as anchor to derive phase-shifts, i.e.,
# ToAs, from an events fits file (see measToAs.py).
# The function "crtPPTemplateMod" runs on an event file and requires a timing model
# (.par file). By default, the event file will be filtered in the energy range 0.5-10 keV.
# The template model is a Fourier series, with n=2 harmonics. These are optional arguments,
# and may be specified by the user. So far only a Fourier, a wrapped Gaussian (von Mises),
# or a wrapped Cauchy (Lorentzian) templates are allowed with a generic number of
# harmonics/components "nbrComp". The fitting procedure is done with a maximum likelihood
# using a gaussian pdf, on a binned pulse profile with number of bins = 30 (may be
# changed as desired). An initial template could be provided (default=None) that would
# serve as initial guess to the fitting procedure. The output is a .txt file with the
# template best fit parameters and a .pdf file of the profile and the best fit model.
# The default output filenames prefix is 'ppTemplateMod', though it could also be
# specified by the user.
# Lastly when an initial template is provided the user has the option to fix the phases of
# the components, i.e., the peaks of the Gaussian/Lorentzian components or the Fourier
# phases. This is important to maitain absolute timing when deriving ToAs from several
# different instruments (think XRT, XTI, PN, etc.), which require their own template.
#
# Note: The function "crtPPTemplateMod" redirects to one of three other functions,
#       "crtPPTemplateModFour", "crtPPTemplateModCauchy", and "crtPPTemplateModVonmises",
#       each could be run separately though on an already created pulse profile. They take
#       as input ppBins, ctRate, ctRateErr arrays (of same length obviously) and optionally,
#       nbrComp(=2), initTemplateMod(=None), outFile(='ppTemplateMod').
#
#
# Input for crtPPTemplateMod:
# 1- evtFile: any event file - could be merged (but for both TIME and GTI - the latter is
#                                               used to get an accurate exposure)
# 2- timMod: timing model, e.g., .par file
# 3- eneLow: low energy cutoff in keV (default = 0.5 keV)
# 4- eneHigh: high energy cutoff in keV (default = 10 keV)
# 5- template: which template model to use (default = fourier, vonmises, cauchy are also allowed)
# 6- nbrComp: number of components in template model (default = 2)
# 7- nbrBins: number of bins in pulse profile (default = 30)
# 8- initTemplateMod: initial template with best-guess for model parameters (default = None)
#                     if this parameter is provided, the "template" and "nbrComp" parameters
#                     will be ignored and instead read-in from this initial template
#                     The user could simply run this script with default values to create
#                     a template, then modify it as necessary
# 9- fixPhases: if True then phases will be fixed to initial value - only applicable if
#               initTemplateMod is provided (default = False)
# 10- outFile: Name of output files 'outfile'.txt and 'outfile'.pdf (default=ppTemplateMod)
# 
# output:
# 1- 'ppTemplateMod'.pdf: a plot of the template best fit
# 2- 'ppTemplateMod'.txt: a text file with the best fit parameters
#                         'ppTemplateMod' could be specified with input parameter 'outFile'
#############################################################

import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits

from lmfit import Parameters, minimize

# Custom modules
from evtFileOps import EvtFileOps
from calcPhase import calcPhase
from crtPP import crtPP
from readPPTemp import readPPTempAny
from templateModels import fourSeries, logLikelihoodFS, wrapCauchy, logLikelihoodCA, vonmises, logLikelihoodVM

sys.dont_write_bytecode = True

###########################################################################
## Script that creates template pulse profile for use in ToA calculation ##
###########################################################################


#################################################################
# Creating a template pulse profile model from event file - for command line
def crtPPTemplateMod(evtFile, timMod, eneLow=0.5, eneHigh=10., template='fourier', nbrComp=2, nbrBins=30, initTemplateMod=None, fixPhases=False, outFile='ppTemplateMod'):
    # Reading some event file keywords
    EF = EvtFileOps(evtFile)
    evtFileKeyWords = EF.readEF()
    # Checking if event file is barycentered 
    if evtFileKeyWords["TIMESYS"] != "TDB":
        raise Exception('Event file is not barycentered. Cannot create template pulse profile')
    MJDREF = evtFileKeyWords["MJDREF"]

    # Reading GTIs to calculate an accurate LIVETIME, in case of a merged event file
    gtiList = EF.readGTI()
    LIVETIME = np.sum(gtiList[:,1]-gtiList[:,0])

    # Reading TIME column after energy filtering
    dataTP_eneFlt = EF.filtEneEF(eneLow=eneLow, eneHigh=eneHigh)
    TIME = dataTP_eneFlt['TIME'].to_numpy()
    timeMJD = TIME/86400 + MJDREF
        
    # Calculating PHASE according to timing model
    #############################################
    _, cycleFoldedPhases = calcPhase(timeMJD, timMod)

    # Creating pulse profile from PHASE
    ###################################
    pulseProfile = crtPP(cycleFoldedPhases,nbrBins)
    ppBins = pulseProfile["ppBins"]
    ctRate = pulseProfile["ctsBins"]/(LIVETIME/nbrBins)
    ctRateErr = pulseProfile["ctsBinsErr"]/(LIVETIME/nbrBins)
    
    # Ftting the pulse profile and building the template
    ####################################################
    if initTemplateMod is None: # in case an initial template not given, use 'template' keyword to redirect to the appropriate function
        if template.casefold()==str.casefold('fourier'):
            bestFitTempModPP = crtPPTemplateModFour(ppBins, ctRate, ctRateErr, nbrComp=nbrComp, outFile=outFile)
        elif template.casefold()==str.casefold('cauchy'):
            bestFitTempModPP = crtPPTemplateModCauchy(ppBins, ctRate, ctRateErr, nbrComp=nbrComp, outFile=outFile)
        elif template.casefold()==str.casefold('vonmises'):
            bestFitTempModPP = crtPPTemplateModVonmises(ppBins, ctRate, ctRateErr, nbrComp=nbrComp, outFile=outFile)
        else:
            raise Exception('Model {} is not supported yet; fourier, vonmises, cauchy are supported'.format(template))
        
    else: # if template is given, continue based on 'model' keyword
        tempModPPparam = readPPTempAny(initTemplateMod)
        if tempModPPparam["model"]==str.casefold('fourier'):
            bestFitTempModPP = crtPPTemplateModFour(ppBins, ctRate, ctRateErr, nbrComp=nbrComp, initTemplateMod=initTemplateMod, fixPhases=fixPhases, outFile=outFile)
        elif tempModPPparam["model"]==str.casefold('cauchy'):
            bestFitTempModPP = crtPPTemplateModCauchy(ppBins, ctRate, ctRateErr, nbrComp=nbrComp, initTemplateMod=initTemplateMod, fixPhases=fixPhases, outFile=outFile)
        elif tempModPPparam["model"]==str.casefold('vonmises'):
            bestFitTempModPP = crtPPTemplateModVonmises(ppBins, ctRate, ctRateErr, nbrComp=nbrComp, initTemplateMod=initTemplateMod, fixPhases=fixPhases, outFile=outFile)
        else:
            raise Exception('Model {} is not supported yet; fourier, vonmises, cauchy are supported'.format(tempModPPparam["model"]))   

    return bestFitTempModPP

#################################################################
# Fourier series fit to pulse profile
# These functions could also be useful if profile is built in a non-conventional way (e.g., no event file is readily available)
def crtPPTemplateModFour(ppBins, ctRate, ctRateErr, nbrComp=2, initTemplateMod=None, fixPhases=False, outFile='ppTemplateMod'):
    # used a few times down below
    template = 'fourier'
    nbrBins = len(ppBins) # number of bins in pulse profile
    
    # Fitting pulse profile utilizing mle
    #####################################
    initParams_mle = Parameters() # Initializing an instance of Parameters
    if initTemplateMod is None: # If no initial template model (i.e., parameter guess) is given, set to default
        # Setting initial guesses to dummy defaults
        initParams_mle.add('norm', 10, min=0.0, max=np.inf)
        for kk in range (1, nbrComp+1):
            initParams_mle.add('amp_'+str(kk), 1)
            initParams_mle.add('ph_'+str(kk), 0)
        initParams_mle.add('phShift', 0, vary=False) # For consistency we define a phase shift for the fourier model, though not required here
        initParams_mle.add('ampShift', 1, vary=False) # For consistency we define an amplitude shift for the fourier model, though not required here
    else: # Setting initial guesses to template parameters
        initParams_mle_temp = readPPTempAny(initTemplateMod)
        nbrComp = len(np.array([ww for harmKey, ww in initParams_mle_temp.items() if harmKey.startswith('amp_')]))
        initParams_mle.add('norm', initParams_mle_temp['norm'], min=0.0, max=np.inf)
        for kk in range (1, nbrComp+1):
            initParams_mle.add('amp_'+str(kk), initParams_mle_temp['amp_'+str(kk)])
            initParams_mle.add('ph_'+str(kk), initParams_mle_temp['ph_'+str(kk)])
            if fixPhases is True: # In case component phases should be fixed
                initParams_mle['ph_'+str(kk)].vary = False
        initParams_mle.add('phShift', 0, vary=False) # For consistency we define a phase shift for the fourier model, though not required here
        initParams_mle.add('ampShift', 1, vary=False) # For consistency we define an amplitude shift for the fourier model, though not required here
        
    # Running the maximum likelihood
    nll = lambda *args: -logLikelihoodFS(*args)
    results_mle_FS = minimize(nll, initParams_mle, args=(ppBins, ctRate, ctRateErr), method='nedler', max_nfev=1.0e6)
    
    # Calculating the bf Model for the data
    bfModelFS = fourSeries(ppBins, results_mle_FS.params)

    # Create plot of best fit model
    ###############################
    # creating a two-cycles for clarity
    ctRate_plt = np.append(ctRate,ctRate)
    ctRateErr_plt = np.append(ctRateErr,ctRateErr)
    ppBins_plt = np.append(ppBins,ppBins+1.0)
    
    fig, ax1 = plt.subplots(1, figsize=(6, 4.0), dpi=80, facecolor='w', edgecolor='k')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlabel(r'$\,\mathrm{Phase\,(cycles)}$', fontsize=12)
    ax1.set_ylabel(r'$\,\mathrm{Rate\,(counts\,s^{-1})}$', fontsize=12)
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.yaxis.offsetText.set_fontsize(12)
    
    plt.step(ppBins_plt, ctRate_plt,'k+-',where='mid')
    plt.errorbar(ppBins_plt, ctRate_plt, yerr=ctRateErr_plt, fmt='ok')
    
    # Creating the best fit model for the data
    bfModelFS_plt = np.append(bfModelFS,bfModelFS)
    plt.plot(ppBins_plt, bfModelFS_plt, 'r-', linewidth=2.0)
    
    # saving PP and template best fit
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(1.5)
        ax1.tick_params(width=1.5)
        
    fig.tight_layout()
    
    outPlot = outFile + '.pdf'
    fig.savefig(outPlot, format='pdf', dpi=1000)

    # Measuring chi2 and reduced chi2 of template best fit
    ######################################################
    chi2 = np.sum(((ctRate-bfModelFS)**2) / (ctRateErr**2))
    nbrFreeParam = nbrComp*2 + 1
    dof = nbrBins-nbrFreeParam
    redchi2 = chi2 / dof
    print('Template {} best fit statistics\n chi2 = {} for dof = {}\n Reduced chi2 = {}'.format(template, chi2, dof, redchi2))
        
    # Write template PP model to .txt file
    ######################################
    bestFitTempModPP = outFile + '.txt'
    results_mle_FS_dict = results_mle_FS.params.valuesdict() # converting best-fit results to dictionary

    f= open(bestFitTempModPP,'w+')
    f.write('model = '+template+'\n')
    f.write('norm = '+str(results_mle_FS_dict["norm"])+'\n')
    
    for nn in range (1,nbrComp+1):
        f.write('amp_'+str(nn)+' = '+str(results_mle_FS_dict["amp_"+str(nn)])+'\n')
        f.write('ph_'+str(nn)+' = '+str(results_mle_FS_dict["ph_"+str(nn)])+'\n')

    f.write('chi2 = '+str(chi2)+'\n')
    f.write('dof = '+str(dof)+'\n')
    f.write('redchi2 = '+str(redchi2)+'\n')
    f.close()
    
    return bestFitTempModPP


#################################################################
# Cauchy model fit to pulse profile
def crtPPTemplateModCauchy(ppBins, ctRate, ctRateErr, nbrComp=2, initTemplateMod=None, outFile='ppTemplateMod'):
    # used a few times down below
    template = 'cauchy'
    nbrBins = len(ppBins) # number of bins in pulse profile
    
    # Typically the pulse profile is normalized to unity, if that is the case, multiply by 2*pi
    if np.max(ppBins)<=1:
        ppBins *= 2*np.pi

    # Fitting pulse profile utilizing mle
    #####################################
    initParams_mle = Parameters() # Initializing an instance of Parameters
    if initTemplateMod is None: # If no initial template model (i.e., parameter guess) is given, set to default
        # Setting initial guesses to dummy defaults
        initParams_mle.add('norm', 10, min=0.0, max=np.inf)
        for kk in range (1, nbrComp+1):
            initParams_mle.add('amp_'+str(kk), 1, min=0.0, max=np.inf)
            initParams_mle.add('cen_'+str(kk), np.pi, min=0.0, max=2*np.pi)
            initParams_mle.add('wid_'+str(kk), 1, min=0.0, max=np.inf)
    else: # Setting initial guesses to template parameters
        initParams_mle_temp = readPPTempAny(initTemplateMod)
        nbrComp = len(np.array([ww for harmKey, ww in initParams_mle_temp.items() if harmKey.startswith('amp_')]))
        initParams_mle.add('norm', initParams_mle_temp['norm'], min=0.0, max=np.inf)
        for kk in range (1, nbrComp+1):
            initParams_mle.add('amp_'+str(kk), initParams_mle_temp['amp_'+str(kk)], min=0.0, max=np.inf)
            initParams_mle.add('cen_'+str(kk), initParams_mle_temp['cen_'+str(kk)], min=0.0, max=2*np.pi)
            initParams_mle.add('wid_'+str(kk), initParams_mle_temp['wid_'+str(kk)], min=0.0, max=np.inf)
            
    # Running the maximum likelihood
    nll = lambda *args: -logLikelihoodCA(*args)
    results_mle_CA = minimize(nll, initParams_mle, args=(ppBins, ctRate, ctRateErr), method='nedler', max_nfev=1.0e6)
    
    # Calculating the bf Model for the data
    bfModelCA = wrapCauchy(ppBins, results_mle_CA.params)
    
    # Create plot of best fit model
    ###############################
    # creating a two-cycles for clarity
    ctRate_plt = np.append(ctRate,ctRate)
    ctRateErr_plt = np.append(ctRateErr,ctRateErr)
    ppBins_plt = np.append(ppBins,ppBins+2*np.pi)
    
    fig, ax1 = plt.subplots(1, figsize=(6, 4.0), dpi=80, facecolor='w', edgecolor='k')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlabel(r'$\,\mathrm{Phase\,(cycles)}$', fontsize=12)
    ax1.set_ylabel(r'$\,\mathrm{Rate\,(counts\,s^{-1})}$', fontsize=12)
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.yaxis.offsetText.set_fontsize(12)
    
    plt.step(ppBins_plt, ctRate_plt,'k+-',where='mid')
    plt.errorbar(ppBins_plt, ctRate_plt, yerr=ctRateErr_plt, fmt='ok')
    
    # Creating the best fit model for the data
    bfModelCA_plt = np.append(bfModelCA,bfModelCA)
    plt.plot(ppBins_plt, bfModelCA_plt, 'r-', linewidth=2.0)
    
    # saving PP and template best fit
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(1.5)
        ax1.tick_params(width=1.5)
        
    fig.tight_layout()
    
    outPlot = outFile + '.pdf'
    fig.savefig(outPlot, format='pdf', dpi=1000)
        
    # Write template PP model to .txt file
    ######################################
    bestFitTempModPP = outFile + '.txt'
    results_mle_CA_dict = results_mle_CA.params.valuesdict() # converting best-fit results to dictionary

    f= open(bestFitTempModPP,'w+')
    f.write('model = '+template+'\n')
    f.write('norm = '+str(results_mle_CA_dict["norm"])+'\n')
    
    for nn in range (1,nbrComp+1):
        f.write('amp_'+str(nn)+' = '+str(results_mle_CA_dict["amp_"+str(nn)])+'\n')
        f.write('cen_'+str(nn)+' = '+str(results_mle_CA_dict["cen_"+str(nn)])+'\n')
        f.write('wid_'+str(nn)+' = '+str(results_mle_CA_dict["wid_"+str(nn)])+'\n')
            
    f.close()

    # Measuring chi2 and reduced chi2 of template best fit
    ######################################################
    chi2 = np.sum(((ctRate-bfModelCA)**2) / (ctRateErr**2))
    nbrFreeParam = nbrComp*3 + 1
    dof = nbrBins-nbrFreeParam
    redchi2 = chi2 / dof
    print('Template {} best fit statistics\n chi2 = {} for dof = {}\n Reduced chi2 = {}'.format(template, chi2, dof, redchi2))
    
    return bestFitTempModPP


#################################################################
# von Mises model fit to pulse profile
def crtPPTemplateModVonmises(ppBins, ctRate, ctRateErr, nbrComp=2, initTemplateMod=None, outFile='ppTemplateMod'):
    # used a few times down below
    template = 'vonmises'
    nbrBins = len(ppBins) # number of bins in pulse profile
    
    # Typically the pulse profile is normalized to unity, if that is the case, multiply by 2*pi
    if np.max(ppBins)<=1:
        ppBins *= 2*np.pi
        
    # Fitting pulse profile utilizing mle
    #####################################
    initParams_mle = Parameters() # Initializing an instance of Parameters
    if initTemplateMod is None: # If no initial template model (i.e., parameter guess) is given, set to default
        # Setting initial guesses to dummy defaults
        initParams_mle.add('norm', 10, min=0.0, max=np.inf)
        for kk in range (1, nbrComp+1):
            initParams_mle.add('amp_'+str(kk), 1, min=0.0, max=np.inf)
            initParams_mle.add('cen_'+str(kk), np.pi, min=0.0, max=2*np.pi)
            initParams_mle.add('wid_'+str(kk), 1, min=0.0, max=np.inf)
    else: # Setting initial guesses to template parameters
        initParams_mle_temp = readPPTempAny(initTemplateMod)
        nbrComp = len(np.array([ww for harmKey, ww in initParams_mle_temp.items() if harmKey.startswith('amp_')]))
        initParams_mle.add('norm', initParams_mle_temp['norm'], min=0.0, max=np.inf)
        for kk in range (1, nbrComp+1):
            initParams_mle.add('amp_'+str(kk), initParams_mle_temp['amp_'+str(kk)], min=0.0, max=np.inf)
            initParams_mle.add('cen_'+str(kk), initParams_mle_temp['cen_'+str(kk)], min=0.0, max=2*np.pi)
            initParams_mle.add('wid_'+str(kk), initParams_mle_temp['wid_'+str(kk)], min=0.0, max=np.inf)

    # Running the maximum likelihood
    nll = lambda *args: -logLikelihoodVM(*args)
    results_mle_VM = minimize(nll, initParams_mle, args=(ppBins, ctRate, ctRateErr), method='least_squares')
    
    # Calculating the bf Model for the data
    bfModelVM = wrapCauchy(ppBins, results_mle_VM.params)
        
    # Create plot of best fit model
    ###############################
    # creating a two-cycles for clarity
    ctRate_plt = np.append(ctRate,ctRate)
    ctRateErr_plt = np.append(ctRateErr,ctRateErr)
    ppBins_plt = np.append(ppBins,ppBins+2*np.pi)
    
    fig, ax1 = plt.subplots(1, figsize=(6, 4.0), dpi=80, facecolor='w', edgecolor='k')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlabel(r'$\,\mathrm{Phase\,(cycles)}$', fontsize=12)
    ax1.set_ylabel(r'$\,\mathrm{Rate\,(counts\,s^{-1})}$', fontsize=12)
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.yaxis.offsetText.set_fontsize(12)
    
    plt.step(ppBins_plt, ctRate_plt,'k+-',where='mid')
    plt.errorbar(ppBins_plt, ctRate_plt, yerr=ctRateErr_plt, fmt='ok')
    
    # Creating the best fit model for the data
    bfModelVM_plt = np.append(bfModelVM,bfModelVM)
    plt.plot(ppBins_plt, bfModelVM_plt, 'r-', linewidth=2.0)
    
    # saving PP and template best fit
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(1.5)
        ax1.tick_params(width=1.5)
        
    fig.tight_layout()
    
    outPlot = outFile + '.pdf'
    fig.savefig(outPlot, format='pdf', dpi=1000)
        
    # Write template PP model to .txt file
    ######################################
    bestFitTempModPP = outFile + '.txt'
    results_mle_VM_dict = results_mle_VM.params.valuesdict() # converting best-fit results to dictionary

    f= open(bestFitTempModPP,'w+')
    f.write('model = '+template+'\n')
    f.write('norm = '+str(results_mle_VM_dict["norm"])+'\n')
    
    for nn in range (1,nbrComp+1):
        f.write('amp_'+str(nn)+' = '+str(results_mle_VM_dict["amp_"+str(nn)])+'\n')
        f.write('cen_'+str(nn)+' = '+str(results_mle_VM_dict["cen_"+str(nn)])+'\n')
        f.write('wid_'+str(nn)+' = '+str(results_mle_VM_dict["wid_"+str(nn)])+'\n')
            
    f.close()

    # Measuring chi2 and reduced chi2 of template best fit
    ######################################################
    chi2 = np.sum(((ctRate-bfModelVM)**2) / (ctRateErr**2))
    nbrFreeParam = nbrComp*3 + 1
    dof = nbrBins-nbrFreeParam
    redchi2 = chi2 / dof
    print('Template {} best fit statistics\n chi2 = {} for dof = {}\n Reduced chi2 = {}'.format(template, chi2, dof, redchi2))
    
    return bestFitTempModPP


##############
# End Script #
##############


######################################
######################################
if __name__ == '__main__':

    ##############################
    ## Parsing input parameters ##
    ##############################
    parser = argparse.ArgumentParser(description="Fold phases to create a pulse profile")
    parser.add_argument("evtFile", help="Event file", type=str)
    parser.add_argument("timMod", help="Timing model in text format. A tempo2 .par file should work.", type=str)
    parser.add_argument("-el","--eneLow", help="lower energy cut, default=0.5", type=float, default=0.5)
    parser.add_argument("-eh","--eneHigh", help="high energy cut, default=10", type=float, default=10)
    parser.add_argument("-tp", "--template", help="Template model to use. default = 'fourier', 'vonmises' and 'cauchy' are also supported", type=str, default='fourier')
    parser.add_argument("-nc", "--nbrComp", help="Number of components in template (nbr of harmonics or nbr of gaussians), default = 2", type=int, default=2)
    parser.add_argument("-nb", "--nbrBins", help="Number of bins for visualization purposes only, default = 15", type=int, default=15)
    parser.add_argument("-it", "--initTemplateMod", help="Initial template model parameters. In this case, keywords template, and nbrComp are ignored", type=str, default=None)
    parser.add_argument("-fp","--fixPhases", help="Flag to fix phases in input initial template model (initTemplateMod), default = False", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-of", "--outFile", help="Output files for best-fit template model plot (appended with .pdf) and text file (appended with .txt)", type=str, default='ppTemplateMod')
    args = parser.parse_args()

    crtPPTemplateMod(args.evtFile, args.timMod, args.eneLow, args.eneHigh, args.template, args.nbrComp, args.nbrBins, args.initTemplateMod, args.fixPhases, args.outFile)
