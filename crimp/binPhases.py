####################################################################################
# Function to bin a "phases" array to create a pulse profile. Phases are assumed
# to be between [0, 1), i.e., cycle folded, e.g., from calcPhase.py
####################################################################################

import numpy as np


def binPhases(phases, nbrBins=15):
    """
    Function to bin an array of phases to create a counts pulse profile, given a number of bins
    :param phases: array of cycle-folded phases, i.e., in the [0,1) range
    :type phases: numpy.ndarray
    :param nbrBins: number of bins in pulse profile
    :type nbrBins: int
    :return: binnedProfile, a dictionary of the pulse profile properties
    :rtype: dict
    """
    # Defining the mid-points of the folded profile bins - this is good for plotting
    ppBins = (np.linspace(0, 1, nbrBins, endpoint=False)) + (1 / nbrBins) / 2
    ppBinsRange = (1 / nbrBins) / 2  # Range that the bins cover, i.e., ppBins +/- ppBinsRange

    # Here we define the edges for the histogram, which we use for binning purposes.
    # Note that this is slightly different from the above since for histogram, you define the start and end of bins
    ppBinsForHist = (np.linspace(0, 1, nbrBins + 1, endpoint=True))
    ctsBins = np.histogram(phases, bins=ppBinsForHist)[0]
    ctsBinsErr = np.sqrt(ctsBins)

    binnedProfile = {'ppBins': ppBins, 'ppBinsRange': ppBinsRange, 'ctsBins': ctsBins, 'ctsBinsErr': ctsBinsErr}

    return binnedProfile
