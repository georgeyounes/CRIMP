"""
Module consisting of two functions:
(1) burstsearch: a simple and efficient script that will search for
magnetar-like bursts using a Poisson probability mass function. It calculates
the average of the full light curve and flags the time bins for which the
counts/bin cannot be explanied by random Poisson fluctuations around mean.
Keeps iterating until no more bursts are found
(2) mergesamebursts: Merges bins flagged as bursts if time between them is
less than desired (user-defined through "tsameburst")
"""

import numpy as np
from scipy.stats import poisson
import pandas as pd


def burstsearch(binnedLC, probLim=0.01, outputfile=None):
    """
    Module to run a simple burst search algorithm on a binned light curve
    :param binnedLC: dataframe of light curve; columns are {'lcBins', 'lcBinsRange', 'ctrate', 'ctrateErr', 'ctsbin'}
    :type binnedLC: pandas.DataFrame
    :param probLim: probability mass function below hich to consider a bin part of a burst
    :type probLim: float
    :param outputfile: Name of file to save the burst search
    :type outputfile: str
    :return: burstsearch_result, dictionary with keys {'bin_bursts', 'bins_nobursts', 'average_rate_nobursts'}
    :rtype: dict
    """
    average_rate = np.mean(binnedLC["ctrate"])
    nbrbins = len(binnedLC["ctrate"])

    # Initial data
    prob = poisson.pmf(binnedLC["ctsbin"], average_rate*binnedLC["lcBinsRange"])
    binnedLC['poissprob'] = prob
    lc_withprob = binnedLC

    # Flag for continuing the burst search until no more found
    foundbursts = True

    # Initialize empty dict
    bin_bursts = [[] for i in range(6)]  # empty lightcurve list
    while foundbursts:
        # Initial arrays that are considered burst-candidates
        # Rarely (e.g., very small GTI) counts/bin is 0 and that flags it as burst, last equality is to avoid this
        bins_bursts_tmp = lc_withprob[(lc_withprob["poissprob"] < (probLim / nbrbins)) &
                                      (lc_withprob["ctsbin"] > average_rate*lc_withprob["lcBinsRange"])]

        if bins_bursts_tmp.empty:
            foundbursts = False
            average_rate_nobursts = np.mean(lc_withprob["ctrate"])
            bins_nobursts = lc_withprob
            continue
        else:
            # First let's append to bin_bursts dictionary - cannot seem to find a better solution
            bin_bursts[0].extend(bins_bursts_tmp["lcBins"].values)
            bin_bursts[1].extend(bins_bursts_tmp["lcBinsRange"].values)
            bin_bursts[2].extend(bins_bursts_tmp["ctrate"].values)
            bin_bursts[3].extend(bins_bursts_tmp["ctrateErr"].values)
            bin_bursts[4].extend(bins_bursts_tmp["ctsbin"].values)
            bin_bursts[5].extend(bins_bursts_tmp["poissprob"].values)

            # Recalculating after removing "bursts" see above for last equality
            bins_nobursts = lc_withprob[(lc_withprob["poissprob"] > (probLim / nbrbins)) |
                                        (lc_withprob["ctsbin"] < average_rate*lc_withprob["lcBinsRange"])]
            average_rate = np.mean(bins_nobursts["ctrate"])
            prob = poisson.pmf(bins_nobursts["ctsbin"], average_rate*bins_nobursts["lcBinsRange"])
            lc_withprob = bins_nobursts.assign(**{'poissprob': prob})

    # Switching from list to pandas dataframe
    bin_bursts = {'lcBins': bin_bursts[0],
                  'lcBinsRange': bin_bursts[1],
                  'ctrate': bin_bursts[2],
                  'ctrateErr': bin_bursts[3],
                  'ctsbin': bin_bursts[4],
                  'poissprob': bin_bursts[5]}
    bin_bursts = pd.DataFrame.from_dict(bin_bursts)
    bin_bursts = bin_bursts.sort_values(by=['lcBins'])
    bin_bursts = bin_bursts.reset_index(drop=True)

    if outputfile is not None:
        bin_bursts.to_csv(outputfile + '.txt', index=False)

    burstsearch_result = {'bin_bursts': bin_bursts, 'bins_nobursts': bins_nobursts,
                          'average_rate_nobursts': average_rate_nobursts}

    return burstsearch_result


def mergesamebursts(burstbins, tsameburst):
    """
    Merge flagged bins separated by time tsameburst or less
    :param burstbins: Dataframe of light curve bins that were flagged as bursts, output from function burstsearch
    :type burstbins: pandas.DataFrame
    :param tsameburst: time in seconds that should separate consecutive bins
    :type tsameburst: float
    :return: mergedbursts, dictionary of dataframes for each merged, individual burst
    :rtype: dict
    """
    burstbins['tdiffflares'] = burstbins['lcBins'].diff()

    counter = 0  # number of individual bursts
    temp_burst = np.array([burstbins.iloc[0].tolist()])
    mergedbursts = {}
    for index, row in burstbins.iloc[1:].iterrows():

        if row['tdiffflares'] <= tsameburst:
            temp_burst_0 = np.array([burstbins.iloc[index].tolist()])
            temp_burst = np.vstack((temp_burst, temp_burst_0))

        else:
            bursts_dfs = pd.DataFrame(temp_burst, columns=burstbins.columns)
            temp_burst = np.array([burstbins.iloc[index].tolist()])
            mergedbursts.update({'burst' + str(counter): bursts_dfs})
            counter += 1

    # Append the last dataframe if not empty
    if temp_burst.size != 0:
        bursts_dfs = pd.DataFrame(temp_burst, columns=burstbins.columns)
        mergedbursts.update({'burst' + str(counter): bursts_dfs})

    return mergedbursts