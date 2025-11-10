"""
A simple class to operate on .tim files
Most importantly, the function phshiftTotimfile converts phase shifts
to a .tim file compatible with Tempo2 and PINT. At the least, it requires
"ToAs.txt" file which is created with measureToAs.py and the exact
.par file that was used to create it.

Can be run from command line as "phshifttotimfile"
"""

import argparse

import numpy as np
import pandas as pd

# Custom scripts
from crimp.ephemIntegerRotation import ephemIntegerRotation
# Log config
############
from crimp.logging_utils import get_logger

logger = get_logger(__name__)


def readtimfile(timfile: str, comment: int = '#', skiprows: int = 1):
    """
    Reads .tim file
    :return: ToAs_Tim - pandas dataframe of ToAs from .tim file
    :rtype: pandas.DataFrame
    """
    # Read everything into a DataFrame (start with the first 5 required columns, and go from there)
    ToAs_Tim = pd.read_csv(timfile, sep=r'\s+', comment=comment, skiprows=skiprows, header=None, engine='python')
    ToAs_Tim = ToAs_Tim.rename(
        columns={0: 'template', 1: 'frequency', 2: 'pulse_ToA', 3: 'pulse_ToA_err', 4: 'time_ref'})

    for c in ["frequency", "pulse_ToA", "pulse_ToA_err"]:
        ToAs_Tim[c] = pd.to_numeric(ToAs_Tim[c], errors="coerce")

    # If there are no trailing columns, we're done
    if ToAs_Tim.shape[1] <= 5:
        return ToAs_Tim

    # Parse trailing flag-value pairs
    extra = ToAs_Tim.iloc[:, 5:].fillna("")
    for idx, row in extra.iterrows():
        toks = row.tolist()
        for j in range(0, len(toks), 2):
            flag, val = toks[j], toks[j + 1] if j + 1 < len(toks) else None  # just in case a flag doesn't have a value
            if not flag or not str(flag).startswith(
                    "-"):  # in case of no flag, or the expected flag does not start with a - sign
                continue
            key = str(flag).lstrip("-")
            try:
                ToAs_Tim.loc[idx, key] = pd.to_numeric(val)
            except Exception:
                ToAs_Tim.loc[idx, key] = val

    return ToAs_Tim


class PulseToAs(object):
    """
    class to work with pulse TOA DataFrame

    Attributes
    ----------
    self.df: pd.DataFrame
        pandas dataframe that hosts the .tim file content (copy of input parameter pulsetoas)
    self._original: pd.DataFrame
        same as above, except it won't be modified

    Methods
    -------
    reset():
        reset dataframe to original
    time_filter():
        Filter dataframe according to time (in MJD)
    writetimfile():
        write dataframe to .tim tempo2 format TOA file
    """

    def __init__(self, pulsetoas: pd.DataFrame):
        """
        Constructs the necessary attribute for the PulseToAs object

        :param pulsetoas: name of the pandas dataframe that hosts a .tim file content
        :type pulsetoas: pandas.DataFrame
        """
        # Keep original for resets; work on a copy for filtering
        self._original = pulsetoas.copy()
        self.df = pulsetoas.copy()

    def reset(self) -> "PulseToAs":
        """Reset working DataFrame to the original"""
        self.df = self._original.copy()
        return self

    def time_filter(self, t_start: float | None = None, t_end: float | None = None, inplace: bool = True
                    ) -> "PulseToAs":
        """
        filter ToAs according to start and end time in MJD
        :param t_start: Start time
        :type t_start: float
        :param t_end: End time
        :type t_end: float
        :param inplace: in place filtering or creating new DataFrame
        :type inplace: bool
        :return: time filtered pandas dataframe of ToAs
        :rtype: pandas.DataFrame
        """
        left = -np.inf if t_start is None else t_start
        right = np.inf if t_end is None else t_end
        mask = self.df["pulse_ToA"].between(left, right)

        if inplace:
            self.df = self.df.loc[mask].copy()
            return self
        else:
            return self.df.loc[mask].copy()

    def writetimfile(self, timfilename: str, clobber: bool = False) -> None:
        """
        Writes .tim file
        :param timfilename: name of output .tim filename
        :type timfilename: str
        :param clobber: override existing .tim file
        :type clobber: bool
        """

        # Create CSV of local ephemerides
        assert isinstance(clobber, bool), "Clobber must be of type boolean"
        if not clobber:
            self.df.to_csv(timfilename + '.tim', sep=' ', index=False, header=False, mode='x')
        else:
            self.df.to_csv(timfilename + '.tim', sep=' ', index=False, header=False)

        # Append .tim file with the format at the start of file - no straightforward way to do this
        with open(timfilename + '.tim', 'r+') as appendtimfile:
            lines = appendtimfile.readlines()  # lines is list of line, each element '...\n'
            lines.insert(0, "FORMAT 1\n")  # you can use any index if you know the line index
            appendtimfile.seek(0)  # file pointer locates at the beginning to write the whole file again
            appendtimfile.writelines(lines)  # write whole lists again to the same file

        return None


def phshiftTotimfile(ToAs, timMod, timfile='residuals', tempModPP='ppTemplateMod', inst='Xray', addpn=False,
                     clobber=False):
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
    :param clobber: override .tim file? (default = False)
    :type clobber: bool
    :return: .tim file as pandas DataFrame
    :rtype: pandas.DataFrame
    """
    df_phShs = pd.read_csv(ToAs, sep=r'\s+', comment='#')
    tToA_MJDs = df_phShs['ToA_mid'].to_numpy()
    dph = df_phShs['phShift'].to_numpy() / (2 * np.pi)
    dph_err = np.hypot(df_phShs['phShift_LL'].to_numpy() / (2 * np.pi),
                       df_phShs['phShift_UL'].to_numpy() / (2 * np.pi)) / np.sqrt(2)  # converting to cycles

    # number of ToAs
    nbrToAs = len(tToA_MJDs)

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

    for ii, tToA_MJD in enumerate(tToA_MJDs):
        ephemerides_intRot = ephemIntegerRotation(tToA_MJD, timMod)

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

    PulseToAs(ToAs_Tim).writetimfile(timfile, clobber=clobber)

    return ToAs_Tim


def main():
    parser = argparse.ArgumentParser(description="Convert a phase-shift text file into a .tim file")
    parser.add_argument("ToAs", help=".txt file of phase shifts created with measureToAs, e.g., ToAs.txt",
                        type=str)
    parser.add_argument("timMod", help=".par timing model", type=str)
    parser.add_argument("-tf", "--timfile", help="output .tim file, default = residuals(.tim)", type=str,
                        default="residuals")
    parser.add_argument("-tp", "--tempModPP",
                        help="Name of best-fit template model used to measure ToAs, default = ppTemplateMod", type=str,
                        default='ppTemplateMod')
    parser.add_argument("-in", "--inst", help="Instrument flag keyword in the .tim file, default = Xray",
                        type=str, default='Xray')
    parser.add_argument("-ap", "--addpn", help="Flag to add pulse numbering, default = False", type=bool,
                        default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-cl", "--clobber", help="Override .tim file (default=False)",
                        default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    phshiftTotimfile(args.ToAs, args.timMod, args.timfile, args.tempModPP, args.inst, args.addpn, args.clobber)


if __name__ == '__main__':
    main()
