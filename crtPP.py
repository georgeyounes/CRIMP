####################################################################################
# A script to create a pulse profile given a PHASE array. The number of bins in the
# pulse profile could be specified through the parameter 'nbrBins', default = 15
# 
# Input:
# 1- phase: PHASE array
# 2- nbrBins: nbr of bins used to build the pulse profile (default = 15)
# 
# output:
# 1- pulseProfile: a dictionary of four arrays, 'ppBins' and 'ppBinsRange' are the bins
#                  centeroids and their range, and 'ctsBins' and 'ctsBinsErr' are the
#                  counts in each bin and their uncertainties
####################################################################################

import sys
import numpy as np

sys.dont_write_bytecode = True

############################################################
## Script that creates a pulse profile from a phase array ##
############################################################

def crtPP(phases, nbrBins=15):

    # Defining the mid points of the pulse profile bins - this is good for plotting
    ppBins = (np.linspace(0, 1, nbrBins, endpoint=False))+(1/nbrBins)/2
    ppBinsRange = (1/nbrBins)/2 # Range that the bins cover, i.e., ppBins +/- ppBinsRange

    # Here we define the edges for the histogram, which we use for binning purposes.
    # Note that this is slightly different from the above since for histogram, you define the start and end of bins
    ppBinsForHist = (np.linspace(0, 1, nbrBins+1, endpoint=True))
    ctsBins = np.histogram(phases, bins=ppBinsForHist)[0]
    ctsBinsErr = np.sqrt(ctsBins)

    pulseProfile = {'ppBins':ppBins, 'ppBinsRange':ppBinsRange, 'ctsBins':ctsBins, 'ctsBinsErr':ctsBinsErr}

    return pulseProfile

################
## End Script ##
################

if __name__ == '__main__':

    ##############################
    ## Parsing input parameters ##
    ##############################
    
    parser = argparse.ArgumentParser(description="Fold phases to create a pulse profile")
    parser.add_argument("phases", help="Array of phases", type=float)
    parser.add_argument("-nb","--nbrBins", help="Number of bins, default = 15", type=int, default=15)
    args = parser.parse_args()

    crtPP(args.phase, args.nbrBins)
