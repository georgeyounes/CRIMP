"""
calcphase.py is a module that calculates phases of an array of TIME (MJD) instances
using a .par file. It reads the .par file with the readTimMod scripts. As a reminder,
this script can manage glitches, wave functions and the frequency and its derivatives
up to F12; it does not accomodate IFUNC, proper motion, binary systems, or parallax.
These *may* be implemented in future versions, likely in that respective order
"""

import sys
import numpy as np
import os
from math import factorial

# Custom modules
from crimp.readtimingmodel import ReadTimingModel, get_parameter_value

sys.dont_write_bytecode = True


class Phases:
    """
        A class to calcualte the phases of a time array

        Attributes
        ----------
        timeMJD : float
            time array in modified julian day
        timMod : str | dict
            timing model, i.e., .par file or a dictionary

        Methods
        -------
        taylorExpansion():
            calculates the phases according to the taylor expansion of the phase evolution (F0, F1, F2, etc.)
        glitches():
            calculates the phases according to glitches
        waves():
            calculates the phases according to "waves" used to whiten noise and align ToAs
        """

    def __init__(self, timeMJD, timMod):
        """
        Constructs all the necessary attributes for the Phases object.

        Parameters
        ----------
            timeMJD : float
                time array in modified julian day
            timMod : str | dict
                timing model, i.e., .par file or a dictionary
        """
        # original shape to restore later
        self._orig_shape = np.shape(timeMJD)
        # work internally with a flat 1-D float array
        self.timeMJD = np.atleast_1d(timeMJD).astype(float).reshape(-1)
        # accept either .par path or dict
        if isinstance(timMod, dict):
            self.timModParam = self._normalize_timdict(timMod)
        elif isinstance(timMod, (str, os.PathLike)):
            self.timModParam = ReadTimingModel(str(timMod)).readfulltimingmodel()[0]
        else:
            raise TypeError("timMod must be a dict or path to a .par file")

    @staticmethod
    def _normalize_timdict(d):
        """
        Make sure the dict has the expected keys and numeric types
        - Converts numeric values to float
        - If a sub-dict contains both {'value', 'flag'}, replaces it with float(value), for ease of phase calculation
        """
        return {k: get_parameter_value(v) for k, v in d.items()}

    def taylorexpansion(self):
        """
        Calculates phases from the taylor expansion of the phase evolution

        :return: phases_te
        :rtype: numpy.ndarray
        """
        t0mjd = self.timModParam["PEPOCH"]
        phases_te = 0.0
        dt = (self.timeMJD - t0mjd) * 86400.0
        for nn in range(1, 14):
            phases_te += (1.0 / factorial(nn)) * self.timModParam[f"F{nn - 1}"] * (dt ** nn)
        return phases_te  # 1-D array

    def glitches(self):
        """
        Calculates phases from glitches in spin evolution

        :return: phases_gl_all
        :rtype: numpy.ndarray
        """
        nbrGlitches = sum(1 for k in self.timModParam if k.startswith('GLEP_'))
        phases_gl_all = np.zeros(self.timeMJD.size, dtype=float)
        time = self.timeMJD  # alias

        for jj in range(1, nbrGlitches + 1):
            glep = float(self.timModParam[f"GLEP_{jj}"])
            mask = (time >= glep)  # 1-D boolean mask
            if not np.any(mask):
                continue

            # safe .get with zeros if some terms are absent
            glph = float(self.timModParam.get(f"GLPH_{jj}", 0.0))
            glf0 = float(self.timModParam.get(f"GLF0_{jj}", 0.0))
            glf1 = float(self.timModParam.get(f"GLF1_{jj}", 0.0))
            glf2 = float(self.timModParam.get(f"GLF2_{jj}", 0.0))
            glf0d = float(self.timModParam.get(f"GLF0D_{jj}", 0.0))
            gltd = float(self.timModParam.get(f"GLTD_{jj}", 0.0))

            t_after = time[mask]
            dt_sec = (t_after - glep) * 86400.0
            # avoid division by zero in exp term
            exp_term = 0.0 if gltd == 0.0 else (gltd * 86400.0) * (1.0 - np.exp(-(t_after - glep) / gltd))

            phases_gl = (
                    glph
                    + glf0 * dt_sec
                    + 0.5 * glf1 * dt_sec ** 2
                    + (1.0 / 6.0) * glf2 * dt_sec ** 3
                    + glf0d * exp_term
            )
            phases_gl_all[mask] += phases_gl

        return phases_gl_all  # 1-D array

    def waves(self):
        """
        Calculates phases from "waves", used to whiten noise and align ToAs

        :return: phases_waves_all
        :rtype: numpy.ndarray
        """
        nbrWaves = np.array([ww for wvKey, ww in self.timModParam.items() if wvKey.startswith('WAVE')])
        phases_waves_all = 0  # initializing the noise in phase due to all waves combined

        if nbrWaves.size:
            waveEpoch = self.timModParam["WAVEEPOCH"]
            waveFreq = self.timModParam["WAVE_OM"]

            for jj in range(1, len(nbrWaves) - 1):
                A = self.timModParam[f"WAVE{jj}"]["A"]
                B = self.timModParam[f"WAVE{jj}"]["B"]
                arg = jj * waveFreq * (self.timeMJD - waveEpoch)
                phases_waves_all += (A * np.sin(arg)) + (B * np.cos(arg))

        # Normalizing by frequency to create residuals in seconds (per tempo2)
        return phases_waves_all * self.timModParam['F0']  # 1-D array (or scalar broadcast)


def calcphase(timeMJD, timMod):
    """
    Function that adds all above (taylor expansion, glitches, waves) phase calculations
    :param timeMJD: time array in modified julian day
    :type timeMJD: float
    :param timMod: timing model, i.e., .par file or a dictionary
    :type timMod: str
    returns
            - totalphases (float): phases (normalised by 2pi)
            - cycleFoldedPhases (float): cycle folded phases [0,1)
    """
    phases = Phases(timeMJD, timMod)
    te = phases.taylorexpansion()
    gl = phases.glitches()
    wav = phases.waves()

    total = te + gl + wav  # vectorized 1-D
    folded = total - np.floor(total)

    # reshape back to original input shape; if scalar, return scalars
    orig_shape = phases._orig_shape
    if orig_shape == ():  # scalar input like np.float64
        return total.item(), folded.item()
    else:
        return total.reshape(orig_shape), folded.reshape(orig_shape)
