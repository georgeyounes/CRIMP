#################################################################################
import numpy as np


def foldPhases(phases, nbrBins=15):
    """ Function to fold an array of phases with a given number of bins
"""
    # Defining the mid-points of the folded profile bins - this is good for plotting
    ppBins = (np.linspace(0, 1, nbrBins, endpoint=False)) + (1 / nbrBins) / 2
    ppBinsRange = (1 / nbrBins) / 2  # Range that the bins cover, i.e., ppBins +/- ppBinsRange

    # Here we define the edges for the histogram, which we use for binning purposes.
    # Note that this is slightly different from the above since for histogram, you define the start and end of bins
    ppBinsForHist = (np.linspace(0, 1, nbrBins + 1, endpoint=True))
    ctsBins = np.histogram(phases, bins=ppBinsForHist)[0]
    ctsBinsErr = np.sqrt(ctsBins)

    foldedProfile = {'ppBins': ppBins, 'ppBinsRange': ppBinsRange, 'ctsBins': ctsBins, 'ctsBinsErr': ctsBinsErr}

    return foldedProfile
