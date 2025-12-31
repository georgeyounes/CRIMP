"""
A code to create START and END times that will define each
TOA. It takes as input a (barycenter-corrected) event file,
the desired number of counts in each ToA (totCtsEachToA),
and a desired max waiting time between GTIs (waitTimeCutoff).
In essence, if the waiting time between two GTIs in an event
file is larger than "waitTimeCutoff", force it to start a
new ToA. This is important because we typically work with
merged event files over years time-scales that encompass a
long-term monitoring program. The code also allows for an
energy filtering through the parameters enelow and enehigh.

The outputs are two text files: (1) "outFile"_bunches.txt,
an intermediary file (that can be largely ignored), and (2)
a "outFile".txt file that summarizes the T_start and T_end
of each ToA_GTI, the length of the time_interval to
accumulate totCtsEachToA, the exact exposure (livetime) of
each ToA, the total number of counts in each ToA, and the
exact count rate. A log file "outFile".log is also cretead.

The code will clean the output file slightly in the sense that
it will merge any time intervals that have counts < min_counts
**and** delta_t < max_wait with the previous one. By default,
min_counts = totCtsEachToA/2 and max_wait = waitTimeCutoff

The code can be run from the command line as "timeintervalsfortoas"

Warning:
In the case of fragmented observations, say, hypothetically, you
have few tens of seconds of exposure (livetime) on your source every
day for 2 weeks, if your "waitTimeCutoff" is >1 day, say 2 days,
you will continue to accumulate counts beyond 2 days (this also
assumes low source count rate and/or large number of desired counts
to derive a ToA). This is because "waitTimeCutoff" is based on GTI
separation, and not "time-passed". This might not be ideal for noisy
pulsars, or when you are trying to find glitches. This script does
not deal with such monitoring instances, simply since they are very
rare (I have not encoutered such a case). Yet, the "ToA_lenInt" column
in the timIntToAs.txt file will provide this information (length of
time from start to end of each ToA), and if the interval length for
the ToA is larger than your desired length, you could simply comment
it out with "#" or just delete it. If you can afford it, lowering
your "totCtsEachToA" might fix this problem. I have thought of a few
ways to deal with this, though none is ideal for various reasons that
I won't go into here, and in fact, will raise other issues.
Again, hopefully, you will never have to worry about this.
"""

import argparse

import numpy as np
import pandas as pd

# Custom modules
from crimp.eventfile import EvtFileOps

# Log config
############
from crimp.logging_utils import get_logger, configure_logging

logger = get_logger(__name__)


def timeintervalsToAs(evtFile, totCtsEachToA=1000, waitTimeCutoff=1.0, eneLow=0.5, eneHigh=10,
                      min_counts: int | None = None, max_wait: float | None = None,
                      outputFile="timIntToAs", correxposure=False):
    """
    Calculates START and END times that will define each TOA
    :param evtFile: name of the fits event file
    :type evtFile: str
    :param totCtsEachToA: max number of counts in each ToA (default = 1000)
    :type totCtsEachToA: int
    :param waitTimeCutoff: max waiting time in days between two GTIs (default = 1)
    :type waitTimeCutoff: float
    :param eneLow: default = 0.5 (keV)
    :type eneLow: float
    :param eneHigh: default = 10 (keV)
    :type eneHigh: float
    :param min_counts: min counts < which merge time interval with previous ones (default = totCtsEachToA/2)
    :type min_counts: int
    :param max_wait: max wait < which merge time interval with previous ones (default = waitTimeCutoff)
    :type max_wait: float
    :param outputFile: default = "timIntToAs"
    :type outputFile: str
    :param correxposure: flag to correct exposure, hence, count rate, according to selected FPMs, default = False)
    :type correxposure: Bool
    :return: dataframe of TOA properties
    :rtype: pandas.DataFrame
    """
    # Establishing min_counts and max_wait for TOA interval cleaning
    if min_counts is None:
        min_counts = int(totCtsEachToA / 2)
    if max_wait is None:
        max_wait = waitTimeCutoff

    logger.info('\n Running timeintervalsToAs with input parameters: \n '
                'evtFile: ' + str(evtFile) +
                '\n totCtsEachToA: ' + str(totCtsEachToA) +
                '\n waitTimeCutoff: ' + str(waitTimeCutoff) +
                '\n eneLow: ' + str(eneLow) +
                '\n eneHigh: ' + str(eneHigh) +
                '\n min_counts: ' + str(min_counts) +
                '\n max_wait: ' + str(max_wait) +
                '\n outputFile: ' + str(outputFile) + '\n')

    # Reading data and filtering for energy
    #######################################
    EF = EvtFileOps(evtFile)
    evtFileKeyWords, gtiList = EF.readGTI()
    # Reading TIME column after energy filtering
    dataTP_eneFlt = EF.build_time_energy_df().filtenergy(eneLow=eneLow, eneHigh=eneHigh)
    TIME = dataTP_eneFlt.time_energy_df['TIME'].to_numpy()

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
        if alltimes[jj, 2] > waitTimeCutoff:
            ind_gtiTiming = np.append(ind_gtiTiming, jj + 1)

    if np.size(ind_gtiTiming) == 0:
        ind_gtiTiming = np.hstack((0, np.shape(alltimes)[0]))
    else:
        ind_gtiTiming = np.hstack((0, ind_gtiTiming.astype(int), np.shape(alltimes)[0]))

    # Creating intermediary array (and text files)
    # This is basically done so that no GTI extends beyond waitTimeCutoff
    ######################################################################
    ToAs_timeInfo = np.zeros((0, 4))

    for kk in range(0, np.size(ind_gtiTiming) - 1):
        alltimes_tmp = alltimes[ind_gtiTiming[kk]:ind_gtiTiming[kk + 1]]

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
            ToAs_timeInfo[ll][2] * 86400) + '\t' + str(ToAs_timeInfo[ll][3]) + '\n')

    f.close()

    # Utilizing the above to create final GTI .txt file that can be used to derive ToAs
    ###################################################################################
    f = open(outputFile + ".txt", "w+")
    f.write('ToA_tstart \t ToA_tend \t ToA_lenInt \t ToA_exposure \t Events \t ct_rate\n')

    for mm in range(0, nbrGTIs):

        timeToA_tmp = (TIME[:] >= ToAs_timeInfo[mm][0]) & (TIME[:] <= ToAs_timeInfo[mm][1])
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
            if expToA_final == 0:
                logger.warning(f"At {timeToA[0]} MJD: exposure = 0 likely caused by a single timestamp "
                               f"in interval - skipping")
                continue
            f.write(
                str('{:0.9f}'.format(timeToA[0])) + '\t' + str(
                    '{:0.9f}'.format(timeToA[-1])) + '\t' + str(
                    '{:0.9f}'.format(timeToA[-1] - timeToA[0])) + '\t' + str(
                    '{:0.9f}'.format(expToA_final * 86400)) + '\t' + str(len(timeToA)) + '\t' + str(
                    '{:0.9f}'.format(len(timeToA) / (expToA_final * 86400))) + '\n')
            continue

        elif nbrToAs == 0:  # This may occur in rare occasions, e.g., one small GTI exists and 0 counts were registered
            continue

        # In case many ToAs are possible within the desired WT interval
        for nn in range(0, nbrToAs):

            if nn == (nbrToAs - 1):
                timeToA = timeToA_tmp[nn * totCtsEachToA:len(timeToA_tmp)]
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
                if expToA_final == 0:
                    logger.warning(f"At {timeToA[0]} MJD: exposure = 0 likely caused by a single timestamp "
                                   f"in interval - skipping")
                    continue
                f.write(str('{:0.9f}'.format(timeToA[0])) + '\t' + str(
                    '{:0.9f}'.format(timeToA[-1])) + '\t' + str(
                    '{:0.9f}'.format(timeToA[-1] - timeToA[0])) + '\t' + str(
                    '{:0.9f}'.format(expToA_final * 86400)) + '\t' + str(len(timeToA)) + '\t' + str(
                    '{:0.9f}'.format(len(timeToA) / (expToA_final * 86400))) + '\n')

            else:
                timeToA = timeToA_tmp[nn * totCtsEachToA:(nn + 1) * totCtsEachToA]
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
                if expToA_final == 0:
                    logger.warning(f"At {timeToA[0]} MJD: exposure = 0 likely caused by a single timestamp "
                                   f"in interval - skipping")
                    continue
                f.write(str('{:0.9f}'.format(timeToA[0])) + '\t' + str(
                    '{:0.9f}'.format(timeToA[-1])) + '\t' + str(
                    '{:0.9f}'.format(timeToA[-1] - timeToA[0])) + '\t' + str(
                    '{:0.9f}'.format(expToA_final * 86400)) + '\t' + str(len(timeToA)) + '\t' + str(
                    '{:0.9f}'.format(len(timeToA) / (expToA_final * 86400))) + '\n')

    f.close()

    # reading the text file we just created as pandas table (i.e., time intervals that define each ToA)
    timInt_toas = pd.read_csv(outputFile + ".txt", sep=r'\s+')

    # Row cleaning, i.e., merge TOA time intervals with counts < min_counts and waitime < max_wait
    timInt_toas = merge_adjacent_intervals(timInt_toas, min_counts, max_wait)

    # Total number of TOA - this should be final after cleaning
    nbrToATOT = len(timInt_toas)

    # Correcting for NICER exposure according to number of detector on. valid for heasoft 6.32+
    if evtFileKeyWords["TELESCOPE"] == 'NICER':
        logger.warning("\n If NICER event files were generaed with HEASOFT version 6.32+,\n"
                       " it is advisable to correct for the number of selected FPMs with the flag -ce for accurate\n"
                       " measurement of count rates\n")
        if correxposure is True:
            logger.info('\n Assuming HEASOFT 6.32+ was used to reduce NICER data\n'
                        ' Getting the number of FPM_SEL during each ToA interval\n')

            # Reading the selected FPM per 1-second interval from event file
            _, FPMSEL_table_condensed = EF.read_fpmsel()

            for pp in range(nbrToATOT):
                toa_start = timInt_toas['ToA_tstart'][pp]
                toa_end = timInt_toas['ToA_tend'][pp]
                # filtering FPM selection table to within each ToA
                mkf_toa_filtered = FPMSEL_table_condensed.loc[
                    ((FPMSEL_table_condensed['TIME'] >= toa_start) & (FPMSEL_table_condensed['TIME'] <= toa_end))]
                # summing all detectors
                nbr_sel_det = np.sum(mkf_toa_filtered['TOTFPMSEL'])
                # total number of detectors if 52 were operating
                exp_nbr_det = 52 * timInt_toas['ToA_exposure'][pp]
                # Measuring correction factor
                correction_factor = exp_nbr_det / nbr_sel_det
                # Changing rate values with corrected ones
                timInt_toas.at[pp, 'ct_rate'] *= correction_factor

        else:
            logger.info('\n No correction of exposure according to number of detectors_selected per ToA interval\n'
                        ' This should not be an issue assuming HEASOFT 6.31- was used to reduce NICER data\n')

    elif evtFileKeyWords["TELESCOPE"] == 'NuSTAR':
        logger.warning("\n If NuSTAR event files are merged for detectors FPMA and FPMB, then resulting count rates "
                       " will be a factor of 2 smaller.\n")

    print('Total number of time intervals that define the TOAs: {}'.format(nbrToATOT))

    # Dumping pandas dataFrame to same output text file with row cleaning and rate correction applied file
    timInt_toas.to_csv(outputFile + ".txt", sep='\t', index=True, index_label='ToA')

    logger.info('\n End of timeintervalsToAs run'
                '\n Total number of time intervals that define each ToA: {}'
                '\n Created the following output files: '
                '\n {}_bunches.txt - this could be largely ignored '
                '\n {}.txt \n and the current {}.log file'.format(nbrToATOT, outputFile, outputFile, outputFile))

    return timInt_toas


def merge_adjacent_intervals(
        df: pd.DataFrame,
        events_max: int,
        dtstart_max_days: float,
) -> pd.DataFrame:
    """
    Merge row jj into the previous row jj-1 if Events[jj] < events_max *and*
    ToA_tstart[jj] - ToA_tstart[jj-1] < dtstart_max_days
    :param df: interval TOAs with columns ['ToA_tstart','ToA_tend','ToA_lenInt','ToA_exposure','Events','ct_rate'], can be built with timeintervalsToAs()
    :type df: pandas.DataFrame
    :param events_max: if row events < events_max attempt merging if dtstart_max_days condition is also true
    :type events_max: int
    :param dtstart_max_days: if diff time (days) between consecutive rows < dtstart_max_days (days) merge if events_max condition is also true
    :type dtstart_max_days: float
    :return: dataframe of cleaned TOAs
    :rtype: pandas.DataFrame
    """
    cols = ['ToA_tstart', 'ToA_tend', 'ToA_lenInt', 'ToA_exposure', 'Events', 'ct_rate']
    if df.empty:
        return pd.DataFrame(columns=cols)

    out = []
    cur = df.iloc[0].copy()  # current merged segment

    for i in range(1, len(df)):
        row = df.iloc[i]
        cond_events = row['Events'] < events_max
        cond_dtstart = (row['ToA_tstart'] - cur['ToA_tend']) < dtstart_max_days

        if cond_events and cond_dtstart:
            # merge into current segment
            new_tstart = cur['ToA_tstart']
            new_tend = row['ToA_tend']
            new_expo = cur['ToA_exposure'] + row['ToA_exposure']
            new_events = cur['Events'] + row['Events']
            new_len = new_tend - new_tstart
            new_rate = new_events / new_expo if new_expo != 0 else float('nan')

            cur['ToA_tstart'] = new_tstart
            cur['ToA_tend'] = new_tend
            cur['ToA_lenInt'] = new_len
            cur['ToA_exposure'] = new_expo
            cur['Events'] = new_events
            cur['ct_rate'] = new_rate
        else:
            # finalize current segment and start a new one
            out.append(cur[cols].copy())
            cur = row.copy()

    out.append(cur[cols].copy())
    return pd.DataFrame(out, columns=cols).reset_index(drop=True)


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
    parser.add_argument("-mc", "--min_counts",
                        help="min counts < which merge time interval with previous ones, default = totCtsEachToA / 2",
                        type=int, default=None)
    parser.add_argument("-mw", "--max_wait",
                        help="max wait < which merge time interval with previous ones, default = waitTimeCutoff",
                        type=float, default=None)
    parser.add_argument("-of", "--outputFile",
                        help="name of .txt output file that defines ToAs. Also name of .log file (default = timIntToAs)",
                        type=str,
                        default='timIntToAs')
    parser.add_argument("-ce", "--correxposure",
                        help="Flag to correct exposure/rate according to selected FPMs, default = False",
                        default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="WARNING if absent, -v: INFO, -vv: DEBUG")
    args = parser.parse_args()

    # Configure the log-file
    v = min(args.verbose, 2)  # cap -vv
    console_level = ("WARNING", "INFO", "DEBUG")[v]  # WARNING if --verbose is absent, INFO if -v, DEBUG if -vv

    log_file = f"{args.outputFile}.log"
    configure_logging(console_level=console_level, file_path=log_file, file_level="INFO", force=True)

    cli_logger = get_logger(__name__)
    cli_logger.info("\nCLI starting")

    timeintervalsToAs(args.evtFile, args.totCtsEachToA, args.waitTimeCutoff, args.eneLow, args.eneHigh,
                      args.min_counts, args.max_wait, args.outputFile, args.correxposure)


if __name__ == '__main__':
    main()
