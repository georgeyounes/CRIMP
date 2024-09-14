############################################################
# A rudimentary simulation of a time series which follows a
# simple sine wave. It only requires an input frequency. The
# user can specify the source count rate, exposure, rms
# pulsed fraction of the signal, the background count rate,
# an instrument resolution or the total number of phase bins
# in intial wave function to draw counts from. The output
# is a dictionary of two TIME arrays, one includes background
# (assigned_t_wBgr) and one without (assigned_t_nobgr)
#
# Input:
# 1- freq : frequency of input signal
# 2- srcrate : source count rate (default = 1 cts/s)
# 3- exposure : desired exposure in seconds (default = 10000)
# 4- pulsedfraction : desired rms pulsed fraction (default=0.2) 
# 5- bgrRate : background count rate (default = 0.05)
# 6- resolution : resolution of instrument (default = 0.073)
# 7- nbrPhaseBins : Number of phase bins to draw source counts from (default = None, i.e., (1/(resolution*freq))
# 
# output:
# 1- assigned_t_wBgr : time of events including background
# 2- assigned_t_nobgr : time of events excluding background
############################################################

import numpy as np
import sys
import argparse

sys.dont_write_bytecode = True


def simulatemodulatedlc(freq, srcrate=1, exposure=10000, pulsedfraction=0.2, bgrrate=0.05, resolution=0.073,
                        nbrPhaseBins=None):
    nbrRotPh = int(exposure * freq)
    # here there should be a warning or even an error if nbrRotPh is not > 100, i.e., exposure > 100*P
    # if not, and the desirved exposure is not a perfect integer multiplied by period, this approximation will lead to inaccurate results
    amp = np.sqrt(2) * pulsedfraction * srcrate

    # Establishing the number of phase bins in pulse profile to draw counts from
    if nbrPhaseBins is None:
        nbrPhaseBins = int(np.floor(1 / (resolution * freq)))

    sim_p = np.linspace(0, 1, nbrPhaseBins, endpoint=False)
    sim_cts = (srcrate + (amp * np.cos(2 * np.pi * sim_p + np.pi))) * (
            exposure / nbrPhaseBins)  # Peak is in the middle of a cycle

    # Simulated light curve
    # Producing the phases of all events in initial pulse profile
    phaseBin = 0
    assigned_phase_all = []

    for k in range(0, nbrPhaseBins):
        nbrRot = np.random.uniform(0, nbrRotPh, int(sim_cts[k]))
        assigned_phase = nbrRot.astype(int) + np.random.uniform(phaseBin, phaseBin + (1 / nbrPhaseBins),
                                                                int(sim_cts[k]))
        assigned_phase_all = np.append(assigned_phase_all, assigned_phase)
        phaseBin += (1 / nbrPhaseBins)
    assigned_phase_all.sort()

    # Assigning times to the simulated phases above
    assigned_t_nobgr = assigned_phase_all / freq
    assigned_t_nobgr = np.sort(assigned_t_nobgr)

    # Adding background to Simulated light curve
    # Calculating total number of backgrond counts
    totBackCts = int(bgrrate * exposure)

    # Assigning times to background counts
    assignedBgr_t_tmp = np.cumsum(
        np.random.exponential(1 / bgrrate, totBackCts))  # Simulate waiting times for a poisson process
    assignedBgr_t = assignedBgr_t_tmp[assignedBgr_t_tmp < exposure]

    # Adding the background counts to the source counts and sorting
    assigned_t_wBgr = np.sort(np.hstack((assigned_t_nobgr, assignedBgr_t)))

    assigned_t = {'assigned_t_wBgr': assigned_t_wBgr, 'assigned_t_nobgr': assigned_t_nobgr}

    return assigned_t


#################
## End Program ##
#################

if __name__ == '__main__':
    #############################
    ## Parsing input arguments ##
    #############################

    parser = argparse.ArgumentParser(description="Simulating an event file following a sinusoidal function")
    parser.add_argument("freq", help="Frequency of signal", type=float)
    parser.add_argument("-sr", "--srcrate", metavar='', help="Source count rate /s, default=1.", type=float, default=1.)
    parser.add_argument("-ex", "--exposure", metavar='', help="Exposure in seconds, default=10000.", type=float,
                        default=10000.)
    parser.add_argument("-pf", "--pulsedfraction", metavar='', help="RMS pulsed fraction of signal, default=0.2.",
                        type=float, default=0.2)
    parser.add_argument("-bg", "--bgrrate", metavar='', type=float, default=0.05,
                        help="background count rate, default=0.05")
    parser.add_argument("-rs", "--resolution", metavar='', help="Resolution of instrument in seconds", type=float,
                        default=0.073)
    parser.add_argument("-nb", "--nbrPhaseBins", metavar='', help="Number of phase bins to draw source counts from",
                        type=int, default=None)
    args = parser.parse_args()

    simulatemodulatedlc(args.freq, args.srcrate, args.exposure, args.pulsedfraction, args.bgrrate, args.resolution,
                        args.nbrPhaseBins)
