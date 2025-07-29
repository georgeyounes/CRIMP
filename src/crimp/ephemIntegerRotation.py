#################################################################
# A simple script that, given an MJD and a .par file, will provide
# the earliest MJD that results in an integer number of rotational
# phases from PEPOCH (epoch of timing solution) and corresponding
# ephemerides (currently only spin frequency)
#
# Reminder that this script deals with taylor expansion of the phase
# evolution and a random number of glitches. Binary motion is not
# included
#
# Input:
# 1- Tmjd: desired time for measurement of ephemerides that result
#          in integer number of rotational phases.
# 2- timeMod: timing model (.par file)
# 3- printOutput: flag to print output to screen (default=False)
#
# output:
# 1- ephemerides_intRotation: dictionary of Tmjd_intRotation and
#                             corresponding rotational frequency
#                             and phase
#################################################################

import argparse
import sys

import numpy as np

# Custom modules
from crimp.ephemTmjd import ephemTmjd
from crimp.calcphase import calcphase

sys.dont_write_bytecode = True


def ephemIntegerRotation(Tmjd, timMod, printOutput=False):
    """
    Function that provides the earliest MJD and corresponding spin frequency
    to input MJD, which, according to the input .par file, would result in an
    integer number of rotational phases from PEPOCH

    :param Tmjd: time for measurement of ephemerides that result
                 in integer number of rotational phases.
    :type Tmjd: float
    :param timMod: .par file
    :type timMod: str
    :param printOutput: Print output (default=Flase)
    :type printOutput: bool
    :return: ephemerides_intRotation
    :rtype: dict
    """

    # Phase and frequency that correspond to Tmjd according to timing model
    ph_Tmjd, _ = calcphase([Tmjd], timMod)
    freq_Tmjd = ephemTmjd(Tmjd, timMod)["freqAtTmjd"]

    # Deriving the closest MJD and spin frequency with an integer number of rotations
    ph_Tmjd_Frac = ph_Tmjd % 1
    FracTFromIntRotation = (ph_Tmjd_Frac / freq_Tmjd) / 86400

    Tmjd_intRotation = Tmjd - FracTFromIntRotation
    freq_Tmjd_intRotation = ephemTmjd(Tmjd_intRotation, timMod)["freqAtTmjd"]
    ph_intRotation, _ = calcphase([Tmjd_intRotation], timMod)
    ph_intRotation = np.round(ph_intRotation)

    if printOutput is True:
        print('Input Tmjd = {} days. Corresponding spin frequency = {} Hz. Corresponding phase = {} \n'
              'Earliest Tmjd with integer number of rotation = {}. Corresponding '
              'frequency = {}. Corresponding phase = {}'.format(Tmjd, freq_Tmjd, ph_Tmjd, Tmjd_intRotation, freq_Tmjd_intRotation, ph_intRotation))

    ephemerides_intRotation = {'Tmjd_intRotation': Tmjd_intRotation, 'freq_intRotation': freq_Tmjd_intRotation,
                               'ph_intRotation': ph_intRotation}

    return ephemerides_intRotation


def main():
    parser = argparse.ArgumentParser(description="Calculate earliest MJD (and corresponding spin frequency and "
                                                 "rotational phase) that results in an integer number of rotations")
    parser.add_argument("tMJD", help="Time in MJD at which to derive frequency and rotational phase", type=float)
    parser.add_argument("timMod", help="Timing model in text format. A tempo2 .par file should work.", type=str)
    parser.add_argument('-po', '--printOutput', help='Print output.', type=bool,
                        default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    ephemIntegerRotation(args.tMJD, args.timMod, args.printOutput)


if __name__ == '__main__':
    main()
