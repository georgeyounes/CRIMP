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
import numpy as np

# Custom modules
from crimp.ephemTmjd import ephemTmjd
from crimp.calcphase import Phases

sys.dont_write_bytecode = True


def ephemIntegerRotation(Tmjd, timMod, printOutput=False, tol_phase=1e-10, max_iter=10):
    """
    Function that provides the earliest MJD to input MJD, which,
    according to the input .par file, would result in an integer
    number of rotational phases from PEPOCH. It also provides the
    corresponding F0 and F1 to the inferred MJD. Caveat, only takes
    into account Taylor+glitch components.

    :param Tmjd: time
    :type Tmjd: float
    :param timMod: .par file
    :type timMod: str
    :param printOutput: Print output (default=Flase)
    :type printOutput: bool
    :param tol_phase: tolerance level for phase correction (default=1.0e-10)
    :type tol_phase: float
    :param max_iter: maximum number of iterations (default=10)
    :type max_iter: int
    :return: ephemerides_intRotation
    :rtype: dict
    """

    ph_Tmjd = phase_no_waves(Tmjd, timMod)
    target_phase = np.floor(ph_Tmjd)

    t = Tmjd

    for _ in range(max_iter):
        ph = phase_no_waves(t, timMod)
        phase_err = ph - target_phase
        if abs(phase_err) < tol_phase:
            break

        freq = ephemTmjd(t, timMod)["freqAtTmjd"]
        t -= (phase_err / freq) / 86400.0

    Tmjd_intRotation = t
    eph = ephemTmjd(Tmjd_intRotation, timMod)

    ph_intRotation = phase_no_waves(Tmjd_intRotation, timMod)

    ephemerides_intRotation = {
        "Tmjd_intRotation": Tmjd_intRotation,
        "freq_intRotation": eph["freqAtTmjd"],
        "freqdot_intRotation": eph["freqdotAtTmjd"],
        "ph_intRotation": ph_intRotation,
        "phase_residual_from_integer": ph_intRotation - np.round(ph_intRotation),
    }

    if printOutput:
        print(f"Input Tmjd = {Tmjd} days. Corresponding phase = {ph_Tmjd}"
              f"\n Earliest Tmjd with integer number of rotation = {Tmjd_intRotation}. "
              f"Corresponding frequency = {eph['freqAtTmjd']}. Corresponding phase = {ph_intRotation}"
              f"\n Phase residual from integer = {ephemerides_intRotation['phase_residual_from_integer']}")

    return ephemerides_intRotation


def phase_no_waves(t, timMod):
    phases = Phases(t, timMod)
    ph = phases.taylorexpansion() + phases.glitches()
    return float(np.atleast_1d(ph)[0])


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
