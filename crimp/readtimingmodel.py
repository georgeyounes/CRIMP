####################################################################################
# readtimingmodel.py is a script that reads in a .par file. It is made to be as
# compatible with tempo2 as possible. It reads in all glitches, frequency derivatives,
# and any wave functions, which are typically used to align pulses in noisy systems,
# i.e., magnetars and isolated neutron stars. Returns a dictionary 'timModParam' of
# parameters. As a reminder this tool does not yet accomodate IFUNC, proper motion,
# binary systems, or parallax. These may get included in later versions, likely in
# that order.
#####################################################################################

import sys
import numpy as np

sys.dont_write_bytecode = True


class ReadTimingModel:
    """
        A class to read in a .par file into a dictionary

        Attributes
        ----------
        timMod : str
            timing model, i.e., .par file

        Methods
        -------
        readtaylorexpansion():
            Read Taylor expansion related parameters (F0, F1, F2, etc.)
        readglitches():
            Read glitch related parameters
        readwaves():
            Read wave related parameters
        """

    def __init__(self, timMod: str):
        """
        Constructs all the necessary attributes for the ReadTimingModel object

        Parameters
        ----------
            timMod : str
                timing model, i.e., .par file
        """
        self.timMod = timMod

    def readtaylorexpansion(self):
        """
        Reading taylor expansion parameters from .par file
        :return: timModParamTE, dictionary of taylor expansion parameters
        :rtype: dict
        """
        data_file = open(self.timMod, 'r+')

        # Reading frequency and its derivatives
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
        blockF12 = ""

        # Reading lines in the .par file
        for line in data_file:
            # Stripping lines vertically to create a list of characters
            li = line.lstrip()

            if li.startswith("F0"):
                blockF0 = line
                F0 = np.float64((" ".join(blockF0.split('F0')[1].split())).split(' ')[0])

            elif li.startswith("PEPOCH"):
                blockt0 = line
                PEPOCH = np.float64((" ".join(blockt0.split('PEPOCH')[1].split())).split(' ')[0])

            elif li.startswith(("F1 ", "F1\t")):  # Necessary to be more specific to distinguish it from F10+
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

        timModParamTE = {'PEPOCH': PEPOCH, 'F0': F0, 'F1': F1, 'F2': F2, 'F3': F3, 'F4': F4,
                         'F5': F5, 'F6': F6, 'F7': F7, 'F8': F8, 'F9': F9,
                         'F10': F10, 'F11': F11, 'F12': F12}

        return timModParamTE

    def readglitches(self):
        """
        Reading glitch parameters from .par file
        :return: timModParamGlitches, dictionary of glitch parameters
        :rtype: dict
        """
        data_file = open(self.timMod, 'r+')

        nmbrOfGlitches = np.array([])
        for line in data_file:
            # Stripping lines vertically to create a list of characters
            li = line.lstrip()
            # Glitch parameters
            if li.startswith("GLEP"):
                blockglep = line
                nmbrOfGlitches = np.append(nmbrOfGlitches, blockglep.split('_')[1].split(' ')[0])

        # Check if there are glitches
        timModParamGlitches = {}  # initialize timModParamGlitches
        if not nmbrOfGlitches.size:
            return timModParamGlitches

        # We now loop over all glitches and add them to the dictionary
        for jj in nmbrOfGlitches:
            data_file.seek(0)
            # Glitch parameters
            # no need to define glep (blockglep), glph (blockglph), and glf0 (blockglf0)
            # These are mandatory for any glitch
            blockglf1 = ""
            blockglf2 = ""
            blockglf0d = ""
            blockgltd = ""

            for line in data_file:
                li = line.lstrip()
                # Glitch parameters
                if li.startswith(("GLEP_" + jj + " ", "GLEP_" + jj + "\t")):
                    blockglep = line
                    glep = np.float64((" ".join(blockglep.split("GLEP_" + jj)[1].split())).split(' ')[0])

                elif li.startswith(("GLPH_" + jj + " ", "GLPH_" + jj + "\t")):
                    blockglph = line
                    glph = np.float64((" ".join(blockglph.split("GLPH_" + jj)[1].split())).split(' ')[0])

                elif li.startswith(("GLF0_" + jj + " ", "GLF0_" + jj + "\t")):
                    blockglf0 = line
                    glf0 = np.float64((" ".join(blockglf0.split("GLF0_" + jj)[1].split())).split(' ')[0])

                elif li.startswith(("GLF1_" + jj + " ", "GLF1_" + jj + "\t")):
                    blockglf1 = line
                    glf1 = np.float64((" ".join(blockglf1.split("GLF1_" + jj)[1].split())).split(' ')[0])

                elif li.startswith(("GLF2_" + jj + " ", "GLF2_" + jj + "\t")):
                    blockglf2 = line
                    glf2 = np.float64((" ".join(blockglf2.split("GLF2_" + jj)[1].split())).split(' ')[0])

                elif li.startswith(("GLF0D_" + jj + " ", "GLF0D_" + jj + "\t")):
                    blockglf0d = line
                    glf0d = np.float64((" ".join(blockglf0d.split("GLF0D_" + jj)[1].split())).split(' ')[0])

                elif li.startswith(("GLTD_" + jj + " ", "GLTD_" + jj + "\t")):
                    blockgltd = line
                    gltd = np.float64((" ".join(blockgltd.split("GLTD_" + jj)[1].split())).split(' ')[0])

                # Setting to 0 glitch parameters that are not mandatory, and are not in the glitch model in the .par file
                if not blockglf1:
                    glf1 = 0

                if not blockglf2:
                    glf2 = 0

                if not blockglf0d:
                    glf0d = 0

                if not blockgltd:
                    gltd = 1  # to avoid a devide by 0

            timModParamGlitches["GLEP_" + jj] = glep
            timModParamGlitches["GLPH_" + jj] = glph
            timModParamGlitches["GLF0_" + jj] = glf0
            timModParamGlitches["GLF1_" + jj] = glf1
            timModParamGlitches["GLF2_" + jj] = glf2
            timModParamGlitches["GLF0D_" + jj] = glf0d
            timModParamGlitches["GLTD_" + jj] = gltd

        return timModParamGlitches

    def readwaves(self):
        """
        Reading wave parameters from .par file
        :return: timModParamWaves, dictionary of glitch parameters
        :rtype: dict
        """
        data_file = open(self.timMod, 'r+')

        waveHarms = np.array([])
        for line in data_file:
            # Stripping lines vertically to create a list of characters
            li = line.lstrip()
            # Wave parameters
            if li.startswith("WAVE"):
                waveHarms = np.append(waveHarms, line.split('WAVE')[1].split(' ')[0])

        # Check if there are waves
        timModParamWaves = {}  # initialize timModParamWaves
        if waveHarms.size:
            data_file.seek(0)
        else:
            return timModParamWaves

        for line in data_file:
            li = line.lstrip()
            # Wave parameters
            if li.startswith("WAVEEPOCH "):
                waveEpoch = np.float64((" ".join(line.split("WAVEEPOCH ")[1].split())).split(' ')[0])

            elif li.startswith("WAVE_OM "):
                waveFreq = np.float64((" ".join(line.split("WAVE_OM ")[1].split())).split(' ')[0])

        timModParamWaves["WAVEEPOCH"] = waveEpoch
        timModParamWaves["WAVE_OM"] = waveFreq

        # Now we loop through the rest of the wave parameters, i.e., the harmonics
        for jj in range(1, len(waveHarms) - 1):
            data_file.seek(0)

            for line in data_file:
                li = line.lstrip()
                # Wave parameters
                if li.startswith(("WAVE" + str(jj) + " ", "WAVE" + str(jj) + "\t")):
                    waveHarmA = np.float64((" ".join(line.split("WAVE" + str(jj))[1].split())).split(' ')[0])
                    waveHarmB = np.float64((" ".join(line.split("WAVE" + str(jj))[1].split())).split(' ')[1])
                    waveHarmAmp = {"A": waveHarmA, "B": waveHarmB}
                    timModParamWaves["WAVE" + str(jj)] = waveHarmAmp

        return timModParamWaves

    def readfulltimingmodel(self):
        """
        Read full .par timing model into a dictionary
        :return: timModParams, dictionary of timing parameters
        :rtype: dict
        """
        timModParamTE = ReadTimingModel.readtaylorexpansion(self)
        timModParamGlitches = ReadTimingModel.readglitches(self)
        timModParamwaves = ReadTimingModel.readwaves(self)
        timModParams = {**timModParamTE, **timModParamGlitches, **timModParamwaves}
        #print(timModParamwaves)
        return timModParams
