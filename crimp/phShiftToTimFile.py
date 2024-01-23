####################################################################
# A simple script to convert phase shifts to a .tim file compatible
# with Tempo2 and PINT. The phase shifts are read-in from a "ToAs.txt"
# file which could be created with measToAs.py. The .par file that
# was used to build the "ToAs.txt" file must also be supplied.
# 
# Inputs:
# 1- ToAs: text file containing phase-shifts (created with measToAs.py)
# 2- timMod: timing model (e.g. .par file)
# 3- timFile: output .tim file (default = residuals(.tim))
# 4- tempModPP: name of tempolate pulse profile used to measure the phase shifts (default = ppTemplateMod)
# 5- flag: a flag inserted into the .timFile to give it a sort of identity (default = Xray)
# 
# outputs:
# 1- .tim file as pandas table
####################################################################
import argparse

import numpy as np
import pandas as pd

# Custom scripts
from crimp import ephemeridesAtTmjd


######################################################################################
# Script that converts phaseShift .txt file to .tim file compatible with Tempo2/PINT #
######################################################################################

def phShiftToTimFile(ToAs, timMod, timFile='residuals', tempModPP='ppTemplateMod', flag='Xray'):
    df_phShs = pd.read_csv(ToAs, sep='\s+', comment='#')
    tToA_MJD = df_phShs['ToA_mid'].to_numpy()
    dph = df_phShs['phShift'].to_numpy() / (2 * np.pi)
    dph_err = np.hypot(df_phShs['phShift_LL'].to_numpy() / (2 * np.pi),
                       df_phShs['phShift_UL'].to_numpy() / (2 * np.pi)) / np.sqrt(2)  # converting to cycles

    # Initializing some parameters
    nbrToAs = len(tToA_MJD)

    # Writing .tim file
    freqInst = 700  # X-rays
    f = open(timFile + '.tim', "w+")
    f.write('FORMAT 1\n')

    for ii in range(nbrToAs):
        ephemerides = ephemeridesAtTmjd(tToA_MJD[ii], timMod)
        ephemerides_intRot = ephemeridesAtTmjd(ephemerides["Tmjd_intRotation"], timMod)

        # Time corresponding to phase shift 
        deltaT = dph[ii] * (1 / ephemerides_intRot["freqAtTmjd"])
        deltaT_err = dph_err[ii] * (1 / ephemerides_intRot["freqAtTmjd"])

        ToATim = ephemerides["Tmjd_intRotation"] + deltaT / 86400
        ToATim_err_mus = deltaT_err * 1.0e6

        f.write(' {0} {1} {2} {3} @ {4} {5}\n'.format(tempModPP, str(freqInst), str(round(ToATim, 12)),
                                                      str(round(ToATim_err_mus, 5)), str('-flag'), flag))

    f.close()

    # Pandas table of .tim file
    ToAs_Tim = pd.read_csv(timFile + '.tim', sep='\s+', comment='#', skiprows=1, header=None)

    return ToAs_Tim


#################
# End of script #
#################

if __name__ == '__main__':
    ############################
    # Parsing input parameters #
    ############################

    parser = argparse.ArgumentParser(description="Fold phases to create a pulse profile")
    parser.add_argument("ToAs", help=".txt file of phase shifts created with measToAs.py, e.g., ToAs.txt", type=str)
    parser.add_argument("timMod", help="Timing model in text format. A tempo2 .par file should work.", type=str)
    parser.add_argument("-tf", "--timfile", help="output .tim file, default = residuals(.tim)", type=str,
                        default="residuals")
    parser.add_argument("-tp", "--tempModPP",
                        help="Name of best-fit template model used to measure ToAs, default = ppTemplateMod", type=str,
                        default='ppTemplateMod')
    parser.add_argument("-fg", "--flag", help="Flag in the .timFile, default = Xray", type=str, default='Xray')
    args = parser.parse_args()

    phShiftToTimFile(args.ToAs, args.timMod, args.timfile, args.tempModPP, args.flag)
