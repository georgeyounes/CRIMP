"""
A simple module to create a light curve from a TIME column. The GTIs
must also be provided and a user defined start and stop time if desired
Two functions are available (1) lightcurve, which specifically builds
the light curve and (2) plotlightcurve, to plot the lightcurve

Input:
1- TIME: TIME column from an event file, i.e., a time-tagged event list
2- GTI: good time intervals, e.g., from an event file
3- timebin: light curve binning length in seconds
4- tstart: tstart of light curve; if None, use GTI START columns
5- tend: tend of light curve; if None, use GTI STOP columns

output:
1- binnedLC: dataframe of light curve; columns are {'lcBins', 'lcBinsRange', 'ctrate', 'ctrateErr', 'ctsbin'}

Future ideas:
1- Deadtime correction
2- add exceptions if tstart > tend, tstart > end of GTI and tend < start
   of GTI
3- For NICER, a correction for FPM_SEL

Warning:
The timbin is limited to the length of each GTI, i.e., the light curve
will not be binned across GTI

Script createlightcurve:
createlightcurve is a script that runs this module on an event file
and optionally perform a burst search as well
Run "createlightcurve -h" for usage and options
"""

import sys
import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt
import pandas as pd

from crimp.eventfile import EvtFileOps
from crimp.bursts import burstsearch

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


def lightcurve(TIME, GTI, timebin=100., tstart=None, tend=None, outputFile=None):
    """
    Function to create a light curve from a TIME array and a GTI array
    :param TIME: time array in seconds
    :type TIME: numpy.ndarray
    :param GTI: gtiList
    :type GTI: numpy.ndarray
    :param timebin: time binsize of light curve
    :type timebin: float
    :param tstart: Start of light curve in seconds
    :type tstart: float
    :param tend: End of light curve in seconds
    :type tend: float
    :param outputFile: Name of output .log and .pdf file
    :type outputFile: str
    :return: binnedLC, dataframe of light curve; columns are {'lcBins', 'lcBinsRange', 'ctrate', 'ctrateErr', 'ctsbin'}
    :rtype: pandas.DataFrame
    :return: GTI, a nx2 array of GTI that corresponds to the light curve
    :rtype: pandas.DataFrame
    """

    logfile = 'lc_logfile' if outputFile is None else outputFile
    fileHandler = logging.FileHandler(logfile + '.log', mode='w')
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.INFO)
    logger.addHandler(fileHandler)

    logger.info('\n Running lightcurve module with input parameters : \n '
                'timebin (s) : ' + str(timebin) +
                '\n tstart (MET, s) : ' + str(tstart) +
                '\n tend (MET, s) : ' + str(tend) +
                '\n outputFile : ' + str(outputFile) + '\n')

    logger.info('\n For NICER only : number of selected FPMs is not taken into account.'
                'This plot should correspond to the nicerl3-lc task with parameter detnormtype = NONE')

    # Restructuring GTI if tstart and tend are provided
    if tstart is None and tend is None:
        logger.info('\n No tstart or tend provided. Using full GTI table instead\n')

    elif tstart is not None and tend is None:
        logger.info('\n Fixing GTI start time to match user defined tstart')
        includegti_idx = (GTI[:, 1] > tstart)
        GTI = GTI[:][includegti_idx]
        if tstart > GTI[0, 0]:
            GTI[0, 0] = tstart

    elif tstart is None and tend is not None:
        logger.info('\n Fixing GTI end time to match user defined tend')
        includegti_idx = (GTI[:, 0] < tend)
        GTI = GTI[:][includegti_idx]
        if tend < GTI[-1, -1]:
            GTI[-1, -1] = tend

    elif tstart is not None and tend is not None:
        logger.info('\n Fixing GTI start and end time to match user defined tstart and tend')
        # Fixing tstart
        includegti_idx = (GTI[:, 1] > tstart)
        GTI = GTI[:][includegti_idx]
        if tstart > GTI[0, 0]:
            GTI[0, 0] = tstart
        # Fixing tend
        includegti_idx = (GTI[:, 0] < tend)
        GTI = GTI[:][includegti_idx]
        if tend < GTI[-1, -1]:
            GTI[-1, -1] = tend

    # Creating light curve
    binnedLC_list = [[] for i in range(5)]  # empty lightcurve list
    for kk, startstop in enumerate(GTI):
        # Defining edges for the histogram - note last bin could have <= timebin resolution
        expEachGTI = startstop[1] - startstop[0]
        fractional_part, nbrBins_pergti = np.modf(expEachGTI / timebin)
        nbrBins_pergti = int(nbrBins_pergti)
        if fractional_part==0:
            lcBins_pergti = np.linspace(startstop[0], startstop[0] + nbrBins_pergti * timebin,
                                                   (nbrBins_pergti + 1))
        else:
            lcBins_pergti = np.hstack((np.linspace(startstop[0], startstop[0]+nbrBins_pergti*timebin,
                                                   (nbrBins_pergti+1)), startstop[1]))

        # Filter TIME array according to GTI
        TIME_idx = (TIME[:] >= startstop[0]) & (TIME[:] <= startstop[1])
        TIME_pergti = TIME[TIME_idx]

        # This should not be necessary, it is a placeholder for when GTI is not accurate, in such a case
        # you may have 0 counts within a certain time interval, not because the source turned off, but because you do
        # not have any livetime during the said time interval. The user can remove those 0 counts by hand from the
        # output of this function, i.e., binnedLC. Warning that if GTI is not accurate, things will get messy!
        # if TIME_pergti.size == 0:
        #    continue

        # Exposure per bin, should be timebin, except for last bin
        lcBinsRange_pergti = lcBins_pergti[1:] - lcBins_pergti[:-1]
        # Midpoint of each bin
        lcBins_pergti_midpoint = (lcBins_pergti[1:] + lcBins_pergti[:-1]) / 2

        # Performing the binning
        ctsbin_pergti = np.histogram(TIME_pergti, bins=lcBins_pergti)[0]
        ctrate_pergti = ctsbin_pergti / lcBinsRange_pergti
        ctrateErr_pergti = np.sqrt(ctsbin_pergti) / lcBinsRange_pergti

        # Appending light curve list with each gti section
        binnedLC_list[0].append(lcBins_pergti_midpoint)
        binnedLC_list[1].append(lcBinsRange_pergti)
        binnedLC_list[2].append(ctrate_pergti)
        binnedLC_list[3].append(ctrateErr_pergti)
        binnedLC_list[4].append(ctsbin_pergti)

    binnedLC = {'lcBins': np.concatenate(binnedLC_list[0]).ravel(),
                'lcBinsRange': np.concatenate(binnedLC_list[1]).ravel(),
                'ctrate': np.concatenate(binnedLC_list[2]).ravel(),
                'ctrateErr': np.concatenate(binnedLC_list[3]).ravel(),
                'ctsbin': np.concatenate(binnedLC_list[4]).ravel()}

    binnedLC = pd.DataFrame.from_dict(binnedLC)

    if outputFile is not None:
        plotlightcurve(binnedLC, outputFile=outputFile)
        logger.info('\n Created figure of light curve : ' + outputFile + '.pdf \n')
    else:
        logger.info('\n No figure file is created\n')

    return binnedLC, GTI


def plotlightcurve(binnedLC, outputFile='lc_plot'):
    """
    Function to make a plot of a light curve given a lightcurve dict
    :param binnedLC: dataframe of light curve; columns are {'lcBins', 'lcBinsRange', 'ctrate', 'ctrateErr'}
    :type binnedLC: dict
    :param outputFile: name of plot (default = 'lc_plot'.pdf)
    :type outputFile: str
    """
    lcBins = binnedLC['lcBins']
    lcBinsRange = binnedLC['lcBinsRange']
    ctrate = binnedLC['ctrate']
    ctrateErr = binnedLC['ctrateErr']

    fig, ax1 = plt.subplots(1, figsize=(6, 4.0), dpi=80, facecolor='w', edgecolor='k')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlabel(r'$\,\mathrm{Time\,(MET,\,s)}$', fontsize=12)
    ax1.set_ylabel(r'$\,\mathrm{Rate\,(counts/\,s)}$', fontsize=12)
    ax1.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.yaxis.offsetText.set_fontsize(12)

    ax1.errorbar(lcBins, ctrate, xerr=lcBinsRange/2, yerr=ctrateErr, fmt='ok')

    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.5)
        ax1.tick_params(width=1.5)

    fig.tight_layout()

    outPlot = outputFile + '.pdf'
    fig.savefig(outPlot, format='pdf', dpi=200)

    return


def main():
    parser = argparse.ArgumentParser(description="Creating a light curve")
    parser.add_argument("evtFile", help="Name of (X-ray) fits event file", type=str)
    parser.add_argument("-el", "--eneLow", help="Low energy filter in event file, default=0.5",
                        type=float, default=0.5)
    parser.add_argument("-eh", "--eneHigh", help="High energy filter in event file, default=10",
                        type=float, default=10)
    parser.add_argument("-tb", "--timebin", help="Time resolution of light curve in seconds",
                        type=float, default=100)
    parser.add_argument("-ts", "--tstart", help="Time start of interval to create light curve, "
                                                "(MET seconds), default=None", type=float, default=None)
    parser.add_argument("-te", "--tend", help="Time end of interval to create light curve, "
                                              "(MET seconds), default=None", type=float, default=None)
    parser.add_argument("-of", "--outputFile",
                        help="name of .pdf light curve and .log file. If None, log file will be named lc_logfile.log",
                        type=str, default=None)
    parser.add_argument("-rb", "--runburstsearch", help="Flag to run a simple burst search algorithm, "
                                                        "default = False", type=bool, default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("-pb", "--problim", help="Probability mass function limit for burst search, "
                                                 "default = 0.01", type=float, default=0.01)

    args = parser.parse_args()

    EVENT_file = EvtFileOps(args.evtFile)
    _, GTI = EVENT_file.readGTI()
    TIME = EVENT_file.filtenergy(eneLow=args.eneLow, eneHigh=args.eneHigh)["TIME"].to_numpy()

    binnedLC, _ = lightcurve(TIME, GTI, args.timebin, args.tstart, args.tend, args.outputFile)

    if args.runburstsearch:
        burstsearch(binnedLC, probLim=args.problim, outputfile=args.outputFile)


if __name__ == '__main__':
    main()
