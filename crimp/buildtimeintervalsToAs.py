#############################################################
# A code to create START and END times that will define each
# TOA. It takes as input a (barycenter-corrected) event file,
# the desired number of counts in each ToA (totCtsEachToA),
# and a desired max waiting time between GTIs (waitTimeCutoff).
# In essence, if the waiting time between two GTIs in an event
# file is larger than "waitTimeCutoff", force it to start a
# new ToA. This is important because we typically work with
# merged event files over years time-scales that encompass a
# full monitoring program. The code also allows for an energy
# filtering through the parameters enelow and enehigh.
# 
# The outputs are two text files: (1) "outFile"_bunches.txt,
# an intermediary file (that can be largely ignored), and (2)
# a "outFile".txt file that summarizes the T_start and T_end
# of each ToA_GTI, the length of the time_interval to
# accumulate totCtsEachToA, the exact exposure (livetime) of
# each ToA, the total number of counts in each ToA, and the
# exact count rate.
# 
# Input:
# 1- evtFile: event file (barycenter-corrected)
# 2- totCtsEachToA: desired number of counts in each ToA_GTI (default = 1000)
#                   (last bin within a valid time interval might not provide exactly totCtsEachToA)
# 3- waitTimeCutoff: GTI gap cutoff to start new ToA (in days, default = 1)
# 4- eneLow: low energy limit (in keV, default = 0.5)
# 5- eneHigh: high energy limit (in keV, default = 10)
# 6- outputFile: name of output file (default = "timIntToAs")
#
# Return: None
#
# output:
# 1- timIntForBunchToAs.txt: intermediate file for housekeeping
# 2- timIntToAs.txt: time info relevant for ToA measurement
#
# Warning:
# In the case of fragmented observations, say, hypothetically,
# you have few tens of seconds of exposure (livetime) on your
# source every day for 2 weeks, if your "waitTimeCutoff" is >1 day,
# you may continuously accumulate counts until you reach the desired
# "totCtsEachToA", e.g., say 2 weeks (which also assumes low source
# count rate and/or large number of desired counts to derive a ToA).
# This is because "waitTimeCutoff" is based on GTI separation, and
# not "time-passed". This might not be ideal for noisy pulsars,
# or when you are trying to find glitches. This script does not deal
# with such monitoring instances, simply since they are very rare
# (I have not encoutered such a case). Yet, the "ToA_lenInt" column
# in the timIntToAs.txt file will provide this information, and if
# the interval length for the ToA is larger than your desired length,
# you could simply comment it out with "#" or just delete it. If you
# can afford it, lowering your "totCtsEachToA" might fix this problem.
# I have thought of a few ways to deal with this, though none is ideal
# for various reasons that I won't go into here.
# Again, hopefully, you will never have to worry about this.
#############################################################

import argparse
import logging

import numpy as np

# Custom modules
from crimp.eventfile import EvtFileOps

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


def timeintervalsToAs(evtFile, totCtsEachToA=1000, waitTimeCutoff=1.0, eneLow=0.5, eneHigh=10,
                      outputFile="timIntToAs"):
    """
    Calculates START and END times that will define each TOA

    :param evtFile:
    :type evtFile: str
    :param totCtsEachToA: default = 1000 (counts)
    :type totCtsEachToA: int
    :param waitTimeCutoff: default = 1 (day)
    :type waitTimeCutoff: float
    :param eneLow: default = 0.5 (keV)
    :type eneLow: float
    :param eneHigh: default = 10 (keV)
    :type eneHigh: float
    :param outputFile: default = "timIntToAs"
    :type outputFile: str
    """

    # Log to a file
    ###############
    fileHandler = logging.FileHandler(outputFile + '.log', mode='w')
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.INFO)
    logger.addHandler(fileHandler)

    logger.info('\n Running timeintervalsToAs with input parameters: \n '
                'evtFile: ' + str(evtFile) +
                '\n totCtsEachToA: ' + str(totCtsEachToA) +
                '\n waitTimeCutoff: ' + str(waitTimeCutoff) +
                '\n eneLow: ' + str(eneLow) +
                '\n eneHigh: ' + str(eneHigh) +
                '\n outputFile: ' + str(outputFile) + '\n')

    # Reading data and filtering for energy
    #######################################
    EF = EvtFileOps(evtFile)
    evtFileKeyWords, gtiList = EF.readGTI()
    MJDREF = evtFileKeyWords["MJDREF"]

    # Reading TIME column after energy filtering
    dataTP_eneFlt = EF.filtenergyEF(eneLow=eneLow, eneHigh=eneHigh)
    TIME = dataTP_eneFlt['TIME'].to_numpy()

    # Calculating the wait time until next GTI and exposure of each GTI
    ###################################################################
    waitForNextGTI = gtiList[1:, 0] - gtiList[:-1, 1]
    waitForNextGTI = np.append(waitForNextGTI, 0)
    expEachGTI = gtiList[:, 1] - gtiList[:, 0]

    # Creating an array combining start and end of GTIs with above
    alltimes = np.vstack((waitForNextGTI, expEachGTI)).T
    alltimes = np.hstack((gtiList, alltimes))

    # Creating an array with the start and stop indeces according to wait time until next GTI
    ind_gtiTiming = []

    for jj in range(0, np.shape(alltimes)[0]):
        if alltimes[jj, 2] > waitTimeCutoff * 86400:
            ind_gtiTiming = np.append(ind_gtiTiming, jj + 1)

    if np.size(ind_gtiTiming) == 0:
        ind_gtiTiming = np.hstack((0, np.shape(alltimes)[0]))
    else:
        ind_gtiTiming = np.hstack((0, ind_gtiTiming.astype(int), np.shape(alltimes)[0]))

    # Creating intermediary array (and text files)
    # This is basically done so that no GTI extends beyond waitTimeCutoff
    ######################################################################
    ToAs_timeInfo = np.zeros((0, 4))

    for ll in range(0, np.size(ind_gtiTiming) - 1):
        alltimes_tmp = alltimes[ind_gtiTiming[ll]:ind_gtiTiming[ll + 1]]

        ToAs_timeInfo_tmp = np.hstack((alltimes_tmp[0, 0], alltimes_tmp[-1, 1], np.sum(alltimes_tmp[:, 3])))
        lenInt_ToABunch = alltimes_tmp[-1, 1] - alltimes_tmp[0, 0]

        ToAs_timeInfo_tmp = np.hstack((ToAs_timeInfo_tmp, lenInt_ToABunch))

        ToAs_timeInfo = np.vstack((ToAs_timeInfo, ToAs_timeInfo_tmp))

    # Writing above results to .txt file.
    f = open(outputFile + "_bunches.txt", "w+")
    nbrGTIs = np.shape(ToAs_timeInfo)[0]

    f.write('ToABunch_tstart \t ToABunch_tend \t ToABunch_exp \t ToABunch_lenInt\n')
    for ll in range(0, nbrGTIs):
        f.write(str(ToAs_timeInfo[ll][0]) + '\t' + str(ToAs_timeInfo[ll][1]) + '\t' + str(
            ToAs_timeInfo[ll][2]) + '\t' + str(ToAs_timeInfo[ll][3]) + '\n')

    f.close()

    # Utilizing the above to create final GTI .txt file that can be used to derive ToAs
    ###################################################################################
    f = open(outputFile + ".txt", "w+")
    f.write('ToA \t ToA_tstart \t ToA_tend \t ToA_lenInt \t ToA_exposure \t Events \t ct_rate\n')

    nbrToATOT = 0

    for kk in range(0, nbrGTIs):

        timeToA_tmp = (TIME[:] >= ToAs_timeInfo[kk][0]) & (TIME[:] <= ToAs_timeInfo[kk][1])
        timeToA_tmp = TIME[timeToA_tmp]

        nbrToAs = np.int64(np.ceil(len(timeToA_tmp) / totCtsEachToA))

        if nbrToAs == 1:  # In case 1 ToA is possible within the desired WT interval
            timeToA = timeToA_tmp[0:len(timeToA_tmp)]
            # See below for explanation
            tsToA_4exp = (gtiList[:, 1] > timeToA[0])
            gtiList_tsToA_tmp = gtiList[tsToA_4exp, :]
            # See below for explanation
            teToA_4exp = (gtiList_tsToA_tmp[:, 0] < timeToA[-1])
            gtiList_fullToA_tmp = gtiList_tsToA_tmp[teToA_4exp, :]
            # See below for explanation
            gtiList_fullToA_tmp[0, 0] = timeToA[0]
            # See below for explanation
            gtiList_fullToA_tmp[-1, -1] = timeToA[-1]
            # See below for explanation
            expToA_final = np.sum(gtiList_fullToA_tmp[:, -1] - gtiList_fullToA_tmp[:, 0])
            f.write(
                str('{}'.format(nbrToATOT)) + '\t' + str('{:0.9f}'.format(timeToA[0] / 86400 + MJDREF)) + '\t' + str(
                    '{:0.9f}'.format(timeToA[-1] / 86400 + MJDREF)) + '\t' + str(
                    '{:0.9f}'.format(timeToA[-1] - timeToA[0])) + '\t' + str(
                    '{:0.9f}'.format(expToA_final)) + '\t' + str(len(timeToA)) + '\t' + str(
                    '{:0.9f}'.format(len(timeToA) / expToA_final)) + '\n')
            nbrToATOT += 1
            continue

        elif nbrToAs == 0:  # This may occur in rare occasions, e.g., one small GTI exists and 0 counts were registered
            continue

        # In case many ToAs are possible within the desired WT interval
        for ll in range(0, nbrToAs):

            if ll == (nbrToAs - 1):
                timeToA = timeToA_tmp[ll * totCtsEachToA:len(timeToA_tmp)]
                # See below for explanation
                tsToA_4exp = (gtiList[:, 1] > timeToA[0])
                gtiList_tsToA_tmp = gtiList[tsToA_4exp, :]
                # See below for explanation
                teToA_4exp = (gtiList_tsToA_tmp[:, 0] < timeToA[-1])
                gtiList_fullToA_tmp = gtiList_tsToA_tmp[teToA_4exp, :]
                # See below for explanation
                gtiList_fullToA_tmp[0, 0] = timeToA[0]
                # See below for explanation
                gtiList_fullToA_tmp[-1, -1] = timeToA[-1]
                # See below for explanation
                expToA_final = np.sum(gtiList_fullToA_tmp[:, -1] - gtiList_fullToA_tmp[:, 0])
                f.write(str('{}'.format(nbrToATOT)) + '\t' + str(
                    '{:0.9f}'.format(timeToA[0] / 86400 + MJDREF)) + '\t' + str(
                    '{:0.9f}'.format(timeToA[-1] / 86400 + MJDREF)) + '\t' + str(
                    '{:0.9f}'.format(timeToA[-1] - timeToA[0])) + '\t' + str(
                    '{:0.9f}'.format(expToA_final)) + '\t' + str(len(timeToA)) + '\t' + str(
                    '{:0.9f}'.format(len(timeToA) / expToA_final)) + '\n')
                nbrToATOT += 1

            else:
                timeToA = timeToA_tmp[ll * totCtsEachToA:(ll + 1) * totCtsEachToA]
                # Measure exact exposure per ToA
                # First cut all GTIs that do not encompass Tstart of ToA
                tsToA_4exp = (gtiList[:, 1] > timeToA[0])
                gtiList_tsToA_tmp = gtiList[tsToA_4exp, :]
                # and Tend of ToA
                teToA_4exp = (gtiList_tsToA_tmp[:, 0] < timeToA[-1])
                gtiList_fullToA_tmp = gtiList_tsToA_tmp[teToA_4exp, :]
                # Now change tstart of first GTI column to match ToA tstart
                gtiList_fullToA_tmp[0, 0] = timeToA[0]
                # and tend of last GTI column to match ToA end
                gtiList_fullToA_tmp[-1, -1] = timeToA[-1]
                # Sum all ToA GTI columns
                expToA_final = np.sum(gtiList_fullToA_tmp[:, 1] - gtiList_fullToA_tmp[:, 0])
                f.write(str('{}'.format(nbrToATOT)) + '\t' + str(
                    '{:0.9f}'.format(timeToA[0] / 86400 + MJDREF)) + '\t' + str(
                    '{:0.9f}'.format(timeToA[-1] / 86400 + MJDREF)) + '\t' + str(
                    '{:0.9f}'.format(timeToA[-1] - timeToA[0])) + '\t' + str(
                    '{:0.9f}'.format(expToA_final)) + '\t' + str(len(timeToA)) + '\t' + str(
                    '{:0.9f}'.format(len(timeToA) / expToA_final)) + '\n')
                nbrToATOT += 1

    f.close()

    print('Total number of time intervals for ToA calculation: {}'.format(nbrToATOT))

    logger.info('\n End of timeintervalsToAs run. Created the following output files:\n '
                + str(outputFile) + '_bunches.txt - this could be largely ignored\n '
                + str(outputFile) + '.txt\n')

    return


def main():
    parser = argparse.ArgumentParser(
        description="Creating time intervals for individual ToAs - saving info to .txt file")
    parser.add_argument("evtFile", help="Fits event file", type=str)
    parser.add_argument("-tc", "--totCtsEachToA", help="Desired number of counts per ToA", type=int,
                        default=1000)
    parser.add_argument("-wt", "--waitTimeCutoff", help="Do not allow any gap in GTI larger than this (in days)",
                        type=float, default=1)
    parser.add_argument("-el", "--eneLow", help="Low energy filter in event file, default=0.5",
                        type=float, default=0.5)
    parser.add_argument("-eh", "--eneHigh", help="High energy filter in event file, default=10",
                        type=float, default=10)
    parser.add_argument("-of", "--outputFile",
                        help="name of .txt output GTI file that defines ToAs. Also, name of .lo file (default = timIntToAs)",
                        type=str,
                        default='timIntToAs')
    args = parser.parse_args()

    timeintervalsToAs(args.evtFile, args.totCtsEachToA, args.waitTimeCutoff, args.eneLow, args.eneHigh, args.outputFile)


if __name__ == '__main__':
    main()
