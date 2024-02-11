############################################################
# This code simulates an event file which follows a simple sine function
# The simulation is based on a given initial pulse profile. The number of counts in
# each bin of the pulse profile are simulated and then randomly distributed in time
# at the specific rotational phase.
# It takes as inputs the frequency of the signal, the amplitude of the signal (default=50)
# and a given normalization (default=100), a phaseShift (default=1.2), number of
# rotational phases, i.e., the length of the light curve = nbrRotPhases*P, and number
# of bins in pulse profile (default=15).
# The shape of the sine function : cts(phase) = norm + amp * cos(2*pi*freq*phase+phaseShift)
#
# Input:
# 1- freq : frequency of input signal
# 2- amp : amplitude of sine function (default = 50)
# 3- norm : a normalization constant (default = 100)
# 4- phSh : phase shift (default = 1.2 radians)
# 5- bgrRate : background count rate (default = 0.05)
# 6- nbrRotPh: number of rotational phases (default = 1000, exposure=1000*Period)
# 7- nbrBins: nbr of bins in pulse profile (default = 15)
#
# output:
# 1- assign_t : time of events
# 2- freq : frequency of input signal
# 3- amp : amplitude of sine function (default = 50)
# 4- norm : a normalization constant (default = 100)
# 5- phSh : phase shift (default = 1.2 radians)
#
#
# Written by George A. Younes 2017 July 22
#
# Future ideas:
# 1- Needs more testing
#
#############################################################

import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from crtPP import crtPP

sys.dont_write_bytecode = True


####################
# Start of script #
####################

def simSinFundEvtFile(freq, norm=100, amp=50, phSh=1.2, amp2=20, phSh2=2.2, bgrRate=0.05, nbrRotPh=1000, nbrBins=15):
    ########################################################
    #####  Creating pulse profile with given parameters ####
    ########################################################    
    sim_p = np.linspace(0, 1, nbrBins)
    sim_cts = norm + (np.multiply(amp, np.cos(np.multiply(1 * 2 * np.pi, sim_p) + phSh))) + (
        np.multiply(amp2, np.cos(np.multiply(2 * 2 * np.pi, sim_p) + phSh2)))
    sim_cts_err = np.sqrt(sim_cts)

    # For plotting
    # sim_cts_plt = np.append(sim_cts,sim_cts)
    # sim_p_plt = np.append(sim_p,sim_p+1)
    # sim_cts_err_plt = np.append(sim_cts_err,sim_cts_err)

    # Plotting the results
    # plt.figure(1, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')

    # H = plt.step(sim_p_plt, sim_cts_plt,'k+-',where='mid')
    # plt.errorbar(sim_p_plt, sim_cts_plt, yerr=sim_cts_err_plt, fmt='ok')

    # plt.ylabel('Number of counts', fontsize=28)
    # plt.xlabel('Phase', fontsize=28)
    # plt.title('Initial pulse profile', fontsize=20)
    # plt.tick_params(axis='both', labelsize=24)

    # saving initial pulse profile
    # name = "PP_initial_nobgr.pdf"
    # plt.savefig(name)
    # plt.close()

    #################################
    #####  Simulated light curve ####
    #################################

    # Producing the phases of all events in initial pulse profile
    phaseBin = 0.
    sizeBin = 1. / nbrBins
    assigned_phase_all = []

    for k in range(0, nbrBins):
        nbrRot = np.random.uniform(1, nbrRotPh, int(sim_cts[k]))
        assigned_phase = nbrRot.astype(int) + np.random.uniform(phaseBin, phaseBin + sizeBin, int(sim_cts[k]))
        phaseBin = phaseBin + sizeBin
        assigned_phase_all = np.append(assigned_phase_all, assigned_phase)

    # Assigning times to the different light curves
    t0 = 0
    assigned_t_nobgr = ((assigned_phase_all - phSh) / freq) + t0
    assigned_t_nobgr = np.sort(assigned_t_nobgr)

    # plt.figure(1, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')

    # H = plt.hist(assigned_t_nobgr,int(nbrRotPh*(1/freq)))

    # plt.tick_params(axis='both', labelsize=24)
    # plt.ylabel('Number of counts', fontsize=28)
    # plt.xlabel('Time (seconds)', fontsize=28)
    # plt.title('Simulated Events Time of Arrival', fontsize=20)

    # saving initial simulated light curve
    # name = "simulatedEventFile_nobgr.pdf"
    # plt.savefig(name)
    # plt.close()

    #################################################################
    #### Recreating the pulse profile from simulated light curve ####
    #################################################################
    phases = phSh + ((assigned_t_nobgr - t0) * freq)
    phases = phases - np.floor(phases)
    phases = np.sort(phases)

    # Creating pulse profile
    pulseProfile_nobgr = crtPP(phases, nbrBins=nbrBins)
    Pbinned_nobgr = pulseProfile_nobgr["ppBins"]
    cts_phase_nobgr = pulseProfile_nobgr["ctsBins"]
    cts_phase_err_nobgr = pulseProfile_nobgr["ctsBinsErr"]

    # Plotting pulse profile
    # cts_phase_plt = np.append(cts_phase_nobgr,cts_phase_nobgr)
    # Pbinned_plt = np.append(Pbinned_nobgr,Pbinned_nobgr+1.0)
    # cts_phase_err_plt = np.append(cts_phase_err_nobgr,cts_phase_err_nobgr)

    # plt.figure(1, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k')

    # H = plt.step(Pbinned_plt, cts_phase_plt,'k+-', where='mid')
    # plt.errorbar(Pbinned_plt, cts_phase_plt, xerr=sizeBin/2, yerr=cts_phase_err_plt, fmt='ok')

    # plt.ylabel('Number of counts', fontsize=12)
    # plt.xlabel('Phase', fontsize=12)
    # plt.title('Pulse Profile of Simulated Event File', fontsize=12)
    # plt.tick_params(axis='both', labelsize=12)
    # plt.xlim(0.0, 2.0)

    # saving pulse profile created from simulated light curve
    # name = "PP_simulatedEventFile_nobgr.pdf"
    # plt.savefig(name)
    # plt.close()

    ######################################################
    #####  Adding background to Simulated light curve ####
    ######################################################

    # Calculating total number of backgrond counts
    exposure = nbrRotPh * (1 / freq)
    totBackCts = int(bgrRate * exposure)

    # Assigning times to background counts
    assignedBgr_t_tmp = np.random.exponential(1 / bgrRate, totBackCts)  # Simulate waiting times for a poisson process
    assignedBgr_t = np.cumsum(assignedBgr_t_tmp)

    # Adding the background counts to the source counts and sorting
    assigned_t_wBgr = np.hstack((assigned_t_nobgr, assignedBgr_t))
    assigned_t_wBgr = np.sort(assigned_t_wBgr)

    print('Source count rate = {}, background count rate = {}, source+background count rate = {}'.format(
        np.size(assigned_t_nobgr) / exposure, np.size(assignedBgr_t) / exposure, np.size(assigned_t_wBgr) / exposure))

    ########################################################################################
    #### Recreating pulse profile from simulated light curve after including background ####
    ########################################################################################

    phases = phSh + ((assigned_t_wBgr - t0) * freq)
    phases = phases - np.floor(phases)
    phases = np.sort(phases)

    # Creating pulse profile
    pulseProfile_wbgr = crtPP(phases, nbrBins=nbrBins)
    Pbinned_wbgr = pulseProfile_wbgr["ppBins"]
    cts_phase_wbgr = pulseProfile_wbgr["ctsBins"]
    cts_phase_err_wbgr = pulseProfile_wbgr["ctsBinsErr"]

    # Plotting pulse profile
    cts_phase_plt = np.append(cts_phase_wbgr, cts_phase_wbgr)
    Pbinned_plt = np.append(Pbinned_wbgr, Pbinned_wbgr + 1.0)
    cts_phase_err_plt = np.append(cts_phase_err_wbgr, cts_phase_err_wbgr)

    plt.figure(1, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k')

    H = plt.step(Pbinned_plt, cts_phase_plt, 'k+-', where='mid')
    plt.errorbar(Pbinned_plt, cts_phase_plt, xerr=sizeBin / 2, yerr=cts_phase_err_plt, fmt='ok')
    plt.ylabel('Number of counts', fontsize=12)
    plt.xlabel('Phase', fontsize=12)
    plt.title('Pulse Profile of Simulated Event File With Background', fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.xlim(0.0, 2.0)

    # saving pulse profile of simulated light curve after including background
    name = "PP_simulatedEventFile_wbgr.pdf"
    plt.savefig(name)
    plt.close()

    ###################
    #####  Returns ####
    ###################

    return assigned_t_wBgr, freq, amp, norm, phSh, bgrRate, nbrRotPh, nbrBins, Pbinned_wbgr, cts_phase_wbgr, cts_phase_err_wbgr, cts_phase_nobgr, cts_phase_err_nobgr


#################
## End Program ##
#################

if __name__ == '__main__':
    #############################
    ## Parsing input arguments ##
    #############################

    parser = argparse.ArgumentParser(description="Simulating an event file following a sinusoidal function")
    parser.add_argument("freq", help="Frequency of signal", type=float)
    parser.add_argument("-n", "--norm", metavar='', help="Normalization constant, default=100.", type=float,
                        default=100.)
    parser.add_argument("-a", "--amp", metavar='', help="Amplitude of sinusoidal function, default=50.", type=float,
                        default=50.)
    parser.add_argument("-p", "--phSh", metavar='', help="Phase shift, default=1.2", type=float, default=1.2)
    parser.add_argument("-a2", "--amp2", metavar='', help="Amplitude of second harmonic, default=20.", type=float,
                        default=20.)
    parser.add_argument("-p2", "--phSh2", metavar='', help="Phase shift of second harmonic, default=2.2", type=float,
                        default=2.2)
    parser.add_argument("-bg", "--bgrRate", metavar='', type=float, default=0.05,
                        help="background count rate, default=0.05")
    parser.add_argument("-nr", "--nbrRotPh", metavar='',
                        help="Number of rotational phases to simulate, i.e., exposure=nbrRotPh*period, default=1000",
                        type=float, default=1000.)
    parser.add_argument("-nb", "--nbrBins", metavar='', help="Number of bins in pulse profile, default=15", type=int,
                        default=15)
    args = parser.parse_args()

    simSinFundEvtFile(args.freq, args.norm, args.amp, args.phSh, args.amp2, args.phSh2, args.bgrRate, args.nbrRotPh,
                      args.nbrBins)
