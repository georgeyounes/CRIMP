####################################################################################
# calcPhase.py is a script that calculates phases of an array of TIME (MJD) instances
# using a .par file. It reads the .par file with the readTimMod scripts. As a reminder,
# this script can manage glitches, wave functions and the frequency and its derivatives
# up to F12; it does not accomodate IFUNC, binary systems, proper motion, or parallax.
# These will be implemented in future versions, likely in that respective order
####################################################################################

import sys
import numpy as np
import argparse

from math import factorial

# Custom modules
from readTimMod import readTimMod

sys.dont_write_bytecode = True

################################################################################################
## Script that calculates pulsar rotational phases using a time array and a .par timing model ##
################################################################################################

def calcPhase(timeMJD, timMod):
    
    timModParam = readTimMod(timMod)
    phase0 = 0 # reference phase - does not mean much in this context
    t0mjd = timModParam["PEPOCH"]
    
    #############################
    # Taking into account the taylor expansion terms to the frequency evolution
    phases_te = phase0
    for nn in range(1, 14):
        phases_te += (1/factorial(nn)) * timModParam["F"+str(nn-1)] * ((timeMJD-t0mjd)*86400)**nn
        
    #############################
    # Taking into account the glitches
    nbrGlitches = len([gg for glKey, gg in timModParam.items() if glKey.startswith('GLEP_')])
    phases_gl_all = 0 # initializing the jump in phase due to all glitches combined

    for jj in range(1,nbrGlitches+1):
        
        glep = timModParam["GLEP_"+str(jj)]
        # Creating boolean list based on whether any times are after glitch
        timesAfterGlitch = (timeMJD>=glep)
        
        # If any time instance in the array timeMJD is after the glitch,
        # calculate phase jumps according to the glitch model
        if timesAfterGlitch.any():
            glph = timModParam["GLPH_"+str(jj)]
            glf0 = timModParam["GLF0_"+str(jj)]
            glf1 = timModParam["GLF1_"+str(jj)]
            glf2 = timModParam["GLF2_"+str(jj)]
            glf0d = timModParam["GLF0D_"+str(jj)]
            gltd = timModParam["GLTD_"+str(jj)]
            # Calculating phases
            # Here we calculate phase jumps according to each glitch model for all time column,
            # then we multiply by 0 if times are < glep and 1 if times are > glep using the boolean list created above
            phases_gl = (glph + (glf0*((timeMJD-glep)*86400)) + (0.5*glf1*((timeMJD-glep)*86400)**2) + ((1/6)*glf2*((timeMJD-glep)*86400)**3) +
                             (glf0d*(gltd*86400)*(1-np.exp(-((timeMJD-glep)*86400)/(gltd*86400))))) * timesAfterGlitch

            phases_gl_all += phases_gl
            
    ###############################
    # Taking into account the waves
    nbrWaves = np.array([ww for wvKey, ww in timModParam.items() if wvKey.startswith('WAVE')])
    phases_waves_all = 0 # initializing the noise in phase due to all waves combined
    
    if nbrWaves.size:
        waveEpoch = timModParam["WAVEEPOCH"]
        waveFreq = timModParam["WAVE_OM"]
        
        for jj in range(1, len(nbrWaves)-1):
            phases_waves_all += (timModParam["WAVE"+str(jj)]["A"]*np.sin(jj*waveFreq*(timeMJD-waveEpoch))) + (timModParam["WAVE"+str(jj)]["B"]*np.cos(jj*waveFreq*(timeMJD-waveEpoch)))
    
    ###################################
    # Adding all phase-shifts together
    phases = phases_te + phases_gl_all + phases_waves_all
    cycleFoldedPhases = phases-np.floor(phases)

    return phases, cycleFoldedPhases

################
## End Script ##
################

if __name__ == '__main__':

    #############################
    ## Parsing input arguments ##
    #############################

    parser = argparse.ArgumentParser(description="Calculate phase timeMJD column and a timing model")
    parser.add_argument("timeMJD", help="Time column from an event file",type=float)
    parser.add_argument("timMod", help="Timing model in text format. A tempo2 .par file should work.",type=str)
    
    args = parser.parse_args()

    calcPhase(args.timeMJD, args.timMod)
