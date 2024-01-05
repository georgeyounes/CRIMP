####################################################################################
# readTimMod.py is a script that reads in a .par file. It is made to be as compatible
# with tempo2 as possible. It reads in all glitches, frequency derivatives, and any 
# wave functions, which are typically used to align pulses in noisy systems, i.e.,  
# magnetars and isolated neutron stars. Returns a dictionary 'timModParam' of
# parameters. As a reminder this tool does not yet accomodate binary systems, or deal
# with proper motion, parallax, or IFUNC
####################################################################################

import os
import sys
import argparse
import numpy as np

sys.dont_write_bytecode = True

##################################
# Script that reads a .par file ##
##################################

# Reading timing model
def readTimMod(timMod):

    ###############################
    # Parsing the .par timing model
    data_file = open(timMod,'r+')
    
    ###################
    # Ephemerides epoch
    blockt0 = ""
    
    #######################################
    # Reading frequency and its derivatives
    blockF0 = ""
    blockF1 = ""
    blockF2 = ""
    blockF3 = ""
    blockF4 = ""
    blockF5 = ""
    blockF6 = ""
    blockF7 = ""
    blockF8 = ""
    blockF9 = ""
    blockF10 = ""
    blockF11 = ""
    blockF11 = ""
    blockF12 = ""

    # Reading lines in the .par file
    for line in data_file:
        # Stripping lines vertically to create a list of characters
        li=line.lstrip()
        
        if li.startswith("F0"):
            blockF0 = line
            F0 = np.float64((" ".join(blockF0.split('F0')[1].split())).split(' ')[0])

        elif li.startswith("PEPOCH"):
            blockt0 = line
            PEPOCH = np.float64((" ".join(blockt0.split('PEPOCH')[1].split())).split(' ')[0])
            
        elif li.startswith(("F1 ","F1\t")): # Necessary to be more specific to distinguish it from F10+
            blockF1 = line
            
        elif li.startswith("F2"):
            blockF2 = line

        elif li.startswith("F3"):
            blockF3 = line

        elif li.startswith("F4"):
            blockF4 = line

        elif li.startswith("F5"):
            blockF5 = line
            
        elif li.startswith("F6"):
            blockF6 = line
            
        elif li.startswith("F7"):
            blockF7 = line
            
        elif li.startswith("F8"):
            blockF8 = line
            
        elif li.startswith("F9"):
            blockF9 = line
            
        elif li.startswith("F10"):
            blockF10 = line
            
        elif li.startswith("F11"):
            blockF11 = line

        elif li.startswith("F12"):
            blockF12 = line

    # These parameters are not mandatory, and if they are not in the .par file, set to 0
    if not blockF1:
        F1 = 0
    else:
        F1 = np.float64((" ".join(blockF1.split('F1')[1].split())).split(' ')[0])
        
    if not blockF2:
        F2 = 0
    else:
        F2 = np.float64((" ".join(blockF2.split('F2')[1].split())).split(' ')[0])

    if not blockF3:
        F3 = 0
    else:
        F3 = np.float64((" ".join(blockF3.split('F3')[1].split())).split(' ')[0])

    if not blockF4:
        F4 = 0
    else:
        F4 = np.float64((" ".join(blockF4.split('F4')[1].split())).split(' ')[0])

    if not blockF5:
        F5 = 0
    else:
        F5 = np.float64((" ".join(blockF5.split('F5')[1].split())).split(' ')[0])

    if not blockF6:
        F6 = 0
    else:
        F6 = np.float64((" ".join(blockF6.split('F6')[1].split())).split(' ')[0])

    if not blockF7:
        F7 = 0
    else:
        F7 = np.float64((" ".join(blockF7.split('F7')[1].split())).split(' ')[0])

    if not blockF8:
        F8 = 0
    else:
        F8 = np.float64((" ".join(blockF8.split('F8')[1].split())).split(' ')[0])

    if not blockF9:
        F9 = 0
    else:
        F9 = np.float64((" ".join(blockF9.split('F9')[1].split())).split(' ')[0])

    if not blockF10:
        F10 = 0
    else:
        F10 = np.float64((" ".join(blockF10.split('F10')[1].split())).split(' ')[0])

    if not blockF11:
        F11 = 0
    else:
        F11 = np.float64((" ".join(blockF11.split('F11')[1].split())).split(' ')[0])

    if not blockF12:
        F12 = 0
    else:
        F12 = np.float64((" ".join(blockF12.split('F12')[1].split())).split(' ')[0])
            
    timModParam = {'PEPOCH':PEPOCH, 'F0':F0, 'F1':F1, 'F2':F2, 'F3':F3, 'F4':F4,
                       'F5':F5, 'F6':F6, 'F7':F7, 'F8':F8, 'F9':F9,
                       'F10':F10, 'F11':F11, 'F12':F12}
        
    data_file.seek(0)
    
    ###############################
    # Now reading glitch parameters
    # First let's determine how many glitches we have
    nmbrOfGlitches = np.array([])
    for line in data_file:
        # Stripping lines vertically to create a list of characters
        li=line.lstrip()
        # Glitch parameters
        if li.startswith("GLEP"):
            blockglep = line
            nmbrOfGlitches = np.append(nmbrOfGlitches, blockglep.split('_')[1].split(' ')[0])
    
    # We now loop over all glitches and add them to the dictionary
    for jj in nmbrOfGlitches:
        data_file.seek(0)

        # Glitch parameters
        blockglep = ""
        blockglph = ""
        blockglf0 = ""
        blockglf1 = ""
        blockglf2 = ""
        blockglf0d = ""
        blockgltd = ""
        
        for line in data_file:
            
            li=line.lstrip()
            # Glitch parameters
            if li.startswith("GLEP_"+jj):
                blockglep = line
                glep = np.float64((" ".join(blockglep.split("GLEP_"+jj)[1].split())).split(' ')[0])

            elif li.startswith("GLPH_"+jj):
                blockglph = line
                glph = np.float64((" ".join(blockglph.split("GLPH_"+jj)[1].split())).split(' ')[0])
                
            elif li.startswith("GLF0_"+jj):
                blockglf0 = line
                glf0 = np.float64((" ".join(blockglf0.split("GLF0_"+jj)[1].split())).split(' ')[0])

            elif li.startswith("GLF1_"+jj):
                blockglf1 = line
                glf1 = np.float64((" ".join(blockglf1.split("GLF1_"+jj)[1].split())).split(' ')[0])

            elif li.startswith("GLF2_"+jj):
                blockglf2 = line
                glf2 = np.float64((" ".join(blockglf2.split("GLF2_"+jj)[1].split())).split(' ')[0])

            elif li.startswith("GLF0D_"+jj):
                blockglf0d = line
                glf0d = np.float64((" ".join(blockglf0d.split("GLF0D_"+jj)[1].split())).split(' ')[0])

            elif li.startswith("GLTD_"+jj):
                blockgltd = line
                gltd = np.float64((" ".join(blockgltd.split("GLTD_"+jj)[1].split())).split(' ')[0])

            # Setting to 0 glitch parameters that are not mandatory, and are not in the glitch model in the .par file
            if not blockglf1:
                glf1 = 0

            if not blockglf2:
                glf2 = 0

            if not blockglf0d:
                glf0d = 0

            if not blockgltd:
                gltd = 1 # to avoid a devide by 0

        timModParam["GLEP_"+jj] = glep
        timModParam["GLPH_"+jj] = glph
        timModParam["GLF0_"+jj] = glf0
        timModParam["GLF1_"+jj] = glf1
        timModParam["GLF2_"+jj] = glf2
        timModParam["GLF0D_"+jj] = glf0d
        timModParam["GLTD_"+jj] = gltd
            
    data_file.seek(0)

    #######################################################
    # Now reading wave parameters - deals with timing noise
    # First let's determine how many harmonics were added
    waveHarms = np.array([])
    for line in data_file:
        # Stripping lines vertically to create a list of characters
        li=line.lstrip()
        # Wave parameters
        if li.startswith("WAVE"):
            waveHarms = np.append(waveHarms, line.split('WAVE')[1].split(' ')[0])
    data_file.seek(0)

    # If waves exist in the .par file, we first read the epoch and frequency
    if waveHarms.size:
        for line in data_file:
            li=line.lstrip()
            # Wave parameters
            if li.startswith("WAVEEPOCH "):
                waveEpoch = np.float64((" ".join(line.split("WAVEEPOCH ")[1].split())).split(' ')[0])
                
            elif li.startswith("WAVE_OM "):
                waveFreq = np.float64((" ".join(line.split("WAVE_OM ")[1].split())).split(' ')[0])

        timModParam["WAVEEPOCH"] = waveEpoch
        timModParam["WAVE_OM"] = waveFreq
                
        # Now we loop through the rest of the wave parameters, i.e., the harmonics
        for jj in range (1, len(waveHarms)-1):
            data_file.seek(0)
        
            for line in data_file:
                li=line.lstrip()
                # Wave parameters
                if li.startswith(("WAVE"+str(jj)+" ","WAVE"+str(jj)+"\t")):
                    waveHarmA = np.float64((" ".join(line.split("WAVE"+str(jj))[1].split())).split(' ')[0])
                    waveHarmB = np.float64((" ".join(line.split("WAVE"+str(jj))[1].split())).split(' ')[1])
                    waveHarmAmp = {"A" : waveHarmA, "B" : waveHarmB}
                    timModParam["WAVE"+str(jj)] = waveHarmAmp


    # Returning timing model parameters
    # Reminder that this scripts reads in taylor series expansion up to F12 (beyond which you start losing accuracy),
                    # glitch parameters for any number of glitches, and wave parameters for timing noise
                    # Things that are not read in and not taken into account are IFUNC, binary modulation, proper motion, parallax, 
                    # Non-relativistic binary modulation, proper motion, parallax, and (especially) IFUNC might be added in later versions
    return timModParam    
            
################
## End Script ##
################

if __name__ == '__main__':

    #############################
    ## Parsing input arguments ##
    #############################
    
    parser = argparse.ArgumentParser(description="Simple script to read in a .par timing model (compatible with TEMPO2)")
    parser.add_argument("timMod", help="Timing model in text format. A tempo2 .par file should work.",type=str)
    args = parser.parse_args()

    readTimMod(args.timMod)
