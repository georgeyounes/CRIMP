####################################################################
# A simple script to convert phase shifts to a .tim file compatible
# with Tempo2 and PINT. The phase shifts are read-in from a "ToAs.txt"
# file which could be created with measureToAs.py. The .par file that
# was used to build the "ToAs.txt" file must also be supplied.
# 
# Inputs:
# 1- ToAs: text file containing phase-shifts (created with measureToAs.py)
# 2- timMod: timing model (e.g. .par file)
# 3- timFile: output .tim file (default = residuals(.tim))
# 4- tempModPP: name of tempolate pulse profile used to measure the phase shifts (default = ppTemplateMod)
# 5- flag: a flag inserted into the .timFile to give it a sort of identity (default = Xray)
# 
# Return:
# 1- .tim file as pandas table
#
####################################################################
import argparse

import numpy as np
import pandas as pd

# Custom scripts
from crimp.ephemIntegerRotation import ephemIntegerRotation


class TimFile:

    def __init__(self, timfile: str):
        """
        Constructs the necessary attribute for the Phases object.

        :param timfile: name of the .tim file
        :type timfile: str
        """
        self.timfile = timfile

    def readtimfile(self, comment: int = '#', skiprows: int = 1):
        """
        Reads .tim file
        :return: ToAs_Tim - pandas dataframe of ToAs from .tim file
        :rtype: pandas.DataFrame
        """
        # Opening the fits file
        ToAs_Tim = pd.read_csv(self.timfile, sep='\s+', comment=comment, skiprows=skiprows, header=None)

        return ToAs_Tim

    def writetimfile(self, ToAs_Tim: pd):
        """
        Writes .tim file
        :param ToAs_Tim: pandas dataframe of ToAs, could be built with phshiftTotimfile function
        :type ToAs_Tim: pandas.DataFrame
        :return: ToAs_Tim - pandas dataframe of ToAs from .tim file
        :rtype: pandas.DataFrame
        """
        # Reading ToAs
        ToAs_Tim.to_csv(self.timfile + '.tim', sep=' ', index=False, header=False, mode='x')

        # Append .tim file with the format at the start of file - no straightforward way to do this
        with open(self.timfile + '.tim', 'r+') as appendtimfile:
            lines = appendtimfile.readlines()  # lines is list of line, each element '...\n'
            lines.insert(0, "FORMAT 1\n")  # you can use any index if you know the line index
            appendtimfile.seek(0)  # file pointer locates at the beginning to write the whole file again
            appendtimfile.writelines(lines)  # write whole lists again to the same file

        return ToAs_Tim


def phshiftTotimfile(ToAs, timMod, timfile='residuals', tempModPP='ppTemplateMod', inst='Xray', addpn=False):
    """
    Convert phase shifts to a .tim file compatible with Tempo2 and PINT
    :param ToAs: text file containing phase-shifts (created with measureToAs.py)
    :type ToAs: str
    :param timMod: timing model (e.g. .par file)
    :type timMod: str
    :param timfile: name of the output .tim file (default = residuals)
    :type timfile: str
    :param tempModPP: name of tempolate pulse profile used to measure the phase shifts (default = ppTemplateMod)
    :type tempModPP: str
    :param inst: flag inserted in .tim file to give it a sort of identity (default = Xray)
    :type inst: str
    :param addpn: flag to add pulse numbers (default = False)
    :type addpn: bool
    :return: .tim file as pandas DataFrame
    :rtype: pandas.DataFrame
    """
    df_phShs = pd.read_csv(ToAs, sep='\s+', comment='#')
    tToA_MJD = df_phShs['ToA_mid'].to_numpy()
    dph = df_phShs['phShift'].to_numpy() / (2 * np.pi)
    dph_err = np.hypot(df_phShs['phShift_LL'].to_numpy() / (2 * np.pi),
                       df_phShs['phShift_UL'].to_numpy() / (2 * np.pi)) / np.sqrt(2)  # converting to cycles

    # number of ToAs
    nbrToAs = len(tToA_MJD)

    # array of constant information in tim file
    freqInst = np.full(nbrToAs, 700)  # X-ray frequency
    tempModPPflag = np.full(nbrToAs, tempModPP)  # name of template used for ToAs
    timeunit = np.full(nbrToAs, '@')  # Barycenter
    instflag = np.full(nbrToAs, '-i')  # flag to identify set of ToAs
    pulsenumberflag = np.full(nbrToAs, '-pn')  # pulsenumbering

    # ToA arrays
    ToATim = np.zeros(nbrToAs)
    ToATim_err_mus = np.zeros(nbrToAs)
    pulsenumber = np.zeros(nbrToAs)

    for ii in range(nbrToAs):
        ephemerides_intRot = ephemIntegerRotation([tToA_MJD[ii]], timMod)

        # Time corresponding to phase shift 
        deltaT = dph[ii] * (1 / ephemerides_intRot["freq_intRotation"])
        deltaT_err = dph_err[ii] * (1 / ephemerides_intRot["freq_intRotation"])

        ToATim[ii] = ephemerides_intRot["Tmjd_intRotation"] + deltaT / 86400
        ToATim_err_mus[ii] = deltaT_err * 1.0e6

        # Pulse numbering, though not normalized to first ToA
        pulsenumber[ii] = ephemerides_intRot["ph_intRotation"]

    ToAs_Tim_dict = {'template': tempModPPflag, 'Frequency': freqInst, 'TOA': np.round(ToATim, 12),
                     'TOA_err': np.round(ToATim_err_mus, 5), 'timeunit': timeunit, 'flag_instrument': instflag,
                     'instrument': inst}

    if addpn:
        pulsenumber -= np.min(pulsenumber)
        ToAs_Tim_dict['pulsenumberflag'] = pulsenumberflag
        ToAs_Tim_dict['pulsenumber'] = pulsenumber

    # Convert dictionary to pandas dataframe
    ToAs_Tim = pd.DataFrame.from_dict(ToAs_Tim_dict)

    TimFile(timfile).writetimfile(ToAs_Tim)

    return ToAs_Tim


def main():
    parser = argparse.ArgumentParser(description="Convert a phase-shift text file into a .tim file")
    parser.add_argument("ToAs", help=".txt file of phase shifts created with measureToAs.py, e.g., ToAs.txt", type=str)
    parser.add_argument("timMod", help=".par timing model", type=str)
    parser.add_argument("-tf", "--timfile", help="output .tim file, default = residuals(.tim)", type=str,
                        default="residuals")
    parser.add_argument("-tp", "--tempModPP",
                        help="Name of best-fit template model used to measure ToAs, default = ppTemplateMod", type=str,
                        default='ppTemplateMod')
    parser.add_argument("-in", "--inst", help="Instrument flag keyword in the .tim file, default = Xray", type=str, default='Xray')
    parser.add_argument("-ap", "--addpn", help="Flag to add pulse numbering, default = False", type=bool, default=False,
                        action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    phshiftTotimfile(args.ToAs, args.timMod, args.timfile, args.tempModPP, args.inst, args.addpn)


if __name__ == '__main__':
    main()
