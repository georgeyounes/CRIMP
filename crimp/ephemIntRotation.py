#################################################################
# A simple script to derive the ephemerides (frequency and
# rotational phase) at a given MJD based on a timing solution
# For convenience it also provides the closest MJD that results
# in an integer number of rotational phases according to the
# timing solution.
# Reminder that currently this script deals with taylor
# expansion of the rotation evolution, a random number of
# glitches, and waves - binary motion is not included
#
# Input:
# 1- Tmjd: time at which to derive frequency
# 2- timeMod: timing model
# 3- Logging
#
# output:
# 1- freqAtTmjd: Frequency at timeMJD
# 2- phAtTmjd: Rotational phase at timeMJD (from reference epoch)
#################################################################

import argparse
import sys
import numpy as np
from math import factorial
import logging

# Custom modules
from crimp.ephemeridesAtTmjd import ephemeridesAtTmjd
from crimp.calcPhase import calcPhase

sys.dont_write_bytecode = True


##########################################################################
# Function that creates a light curve given a time column and a GTI list #
##########################################################################

def ephemIntRotation(Tmjd, timMod, loglevel='warning'):
    freqAtTmjd = ephemeridesAtTmjd(Tmjd, timMod)["freqAtTmjd"]

    # Phases that correspond to Tmjd according to timing model
    phAtTmjd, _ = calcPhase(Tmjd, timMod)

    # Deriving the closest MJD and spin frequency with an integer number of rotations
    phAtTmjd_Frac = phAtTmjd % 1
    FracTFromIntRotation = ((1 - phAtTmjd_Frac) / freqAtTmjd) / 86400

    Tmjd_intRotation = Tmjd + FracTFromIntRotation
    freq_intRotation = ephemeridesAtTmjd(Tmjd_intRotation, timMod)["freqAtTmjd"]
    ph_intRotation, _ = calcPhase(Tmjd_intRotation, timMod)

    logging.basicConfig(level=loglevel.upper())
    logging.info(
        ' Tmjd corresponding to an integer number of rotation = {}\n Frequency at this MJD is {}\n Number of rotations passed = {}'.format(
            Tmjd_intRotation, freq_intRotation, ph_intRotation))

    ephemerides_intRotation = {'Tmjd_intRotation': Tmjd_intRotation, 'freq_intRotation': freq_intRotation,
                               'ph_intRotation': ph_intRotation}

    return ephemerides_intRotation


def main():
    parser = argparse.ArgumentParser(description="Calculate closes MJD (and corresponding spin frequency and "
                                                 "rotational phase) that results in an integer number of rotation")
    parser.add_argument("tMJD", help="Time in MJD at which to derive frequency and rotational phase", type=float)
    parser.add_argument("timMod", help="Timing model in text format. A tempo2 .par file should work.", type=str)
    parser.add_argument('-ll', '--loglevel', help='Provide logging level. default, default=warning', default='warning')
    args = parser.parse_args()

    ephemIntRotation(args.tMJD, args.timMod, args.loglevel)


if __name__ == '__main__':
    main()
