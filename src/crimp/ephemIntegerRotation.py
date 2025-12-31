"""
A simple module that, given an input MJD and a .par file, will provide
the earliest MJD from the input MJD that results in an integer number
of rotational phases from PEPOCH according to the input timing solution.
It will also provide the ephemerides at that MJD, i.e., F0, and F1.

Reminder that this script deals with taylor expansion of the phase
evolution and a random number of glitches. Binary motion is not
included

Can be run from command line via "ephemintegerrotation"
"""

import argparse
import sys

# Custom modules
from crimp.ephemTmjd import ephemTmjd
from crimp.calcphase import Phases

sys.dont_write_bytecode = True


def ephemIntegerRotation(Tmjd, timMod, printOutput=False):
    """
    Function that provides the earliest MJD to input MJD, which,
    according to the input .par file, would result in an integer
    number of rotational phases from PEPOCH. It also provides the
    corresponding F0 and F1 to the inferred MJD.

    :param Tmjd: time
    :type Tmjd: float
    :param timMod: .par file
    :type timMod: str
    :param printOutput: Print output (default=Flase)
    :type printOutput: bool
    :return: ephemerides_intRotation
    :rtype: dict
    """

    # Phase and frequency that correspond to Tmjd according to timing model
    # (only consider taylor expansion and glitches, ignore waves)
    phases = Phases(Tmjd, timMod)
    ph_Tmjd = phases.taylorexpansion() + phases.glitches()

    # Frequency at Tmjd
    freq_Tmjd = ephemTmjd(Tmjd, timMod)["freqAtTmjd"]

    # Deriving the closest MJD and spin frequency with an integer number of rotations
    ph_Tmjd_Frac = ph_Tmjd % 1
    FracTFromIntRotation = (ph_Tmjd_Frac / freq_Tmjd) / 86400

    Tmjd_intRotation = Tmjd - FracTFromIntRotation
    freq_Tmjd_intRotation = ephemTmjd(Tmjd_intRotation, timMod)["freqAtTmjd"]
    freqdot_Tmjd_intRotation = ephemTmjd(Tmjd_intRotation, timMod)["freqdotAtTmjd"]

    # Phase at interger rotation
    phases_intRotation = Phases(Tmjd_intRotation, timMod)
    ph_intRotation = phases_intRotation.taylorexpansion() + phases_intRotation.glitches()

    if printOutput is True:
        print(f"Input Tmjd = {Tmjd} days. Corresponding spin frequency = {freq_Tmjd} Hz. "
              f"Corresponding phase = {ph_Tmjd} \n Earliest Tmjd with integer number of rotation = {Tmjd_intRotation}. "
              f"Corresponding frequency = {freq_Tmjd_intRotation}. Corresponding phase = {ph_intRotation}")

    ephemerides_intRotation = {'Tmjd_intRotation': Tmjd_intRotation, 'freq_intRotation': freq_Tmjd_intRotation,
                               'freqdot_intRotation': freqdot_Tmjd_intRotation, 'ph_intRotation': ph_intRotation}

    return ephemerides_intRotation


def main():
    parser = argparse.ArgumentParser(description="Calculate earliest MJD (and corresponding spin frequency and "
                                                 "rotational phase) that results in an integer number of rotations")
    parser.add_argument("tMJD", help="Time in MJD at which to derive frequency and rotational phase", type=float)
    parser.add_argument("timMod", help="Timing model in text format. A tempo2 .par file should work", type=str)
    parser.add_argument('-po', '--printOutput', help='Print output',
                        default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    ephemIntegerRotation(args.tMJD, args.timMod, args.printOutput)


if __name__ == '__main__':
    main()
