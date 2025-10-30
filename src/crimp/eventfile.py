"""
eventfile.py is a module to perform simple operations on X-ray event files,
reading useful header words, filter for energy, add a phase column, etc. X-ray
instruments that are accepted are Swift/XRT, NICER, XMM/EPIC, IXPE, Fermi/GBM,
NuSTAR

The method addPhaseColEF() can be called via command line as "addphasecolumn"
"""

import sys
import argparse
import numpy as np
import pandas as pd

from astropy.table import Table, Column
from astropy.io import fits

# Custom modules
from crimp.calcphase import calcphase

sys.dont_write_bytecode = True

# Log config
############
from crimp.logging_utils import get_logger
logger = get_logger(__name__)


########################################################################################################
# Class that performs simple operations on event files, reading, energy filtering, adding phase Column #
########################################################################################################

class EvtFileOps:
    """
    A class to operate on a fits event file

    Attributes
    ----------
    evtFile : str
        name of the fits event file
    time_energy_df : pd.DataFrame
        TIME and PI dataframe

    Methods
    -------
    open_fits(): opens the fits and returns an HDUList
    readEF(): reads essential keywords from an event file
    read_fpmsel(): reads FPMSEL table (only valid for NICER)
    readGTI(): reads GTI table from an event
    build_time_energy_df(): builds ['TIME', 'PI'] dataframe from EVENT table (creates time_energy_df attribute)
    filtenergy(eneLow: float, eneHigh: float): filter build_time_energy_df for energy range
    filttime(self, t_start: float | None = None, t_end: float | None = None): filter build_time_energy_df for time range
    addPhaseColEF(): adds phase column to event file according to timing model (.par file)
    """

    def __init__(self, evtFile: str):
        """
        Constructs the necessary attribute

        :param evtFile: name of the fits event file
        :type evtFile: str
        """
        self.evtFile = evtFile
        self.time_energy_df = None

    #################################################################
    def open_fits(self):
        hdulist = fits.open(self.evtFile)
        return hdulist

    #################################################################
    def readEF(self):  # Reading fits event file from X-ray satellites
        """
        Reads essential keywords from an event file
        :return: evtFileKeyWords - dictionary of essential keywords
        :rtype: dict
        """
        # Opening the fits file
        hdulist = self.open_fits()

        # Reading some essential keywords
        TELESCOPE = hdulist['EVENTS'].header['TELESCOP']
        INSTRUME = hdulist['EVENTS'].header['INSTRUME']
        TSTART = hdulist['EVENTS'].header['TSTART']
        TSTOP = hdulist['EVENTS'].header['TSTOP']
        TIMESYS = hdulist['EVENTS'].header['TIMESYS']
        DATEOBS = hdulist['EVENTS'].header['DATE-OBS']

        # Some keywords are mission specific - check for them and move on
        evt_hd = hdulist['EVENTS'].header

        if 'TIMEZERO' not in evt_hd:
            TIMEZERO = None
        else:
            TIMEZERO = hdulist['EVENTS'].header['TIMEZERO']

        if 'OBS_ID' not in evt_hd:
            OBS_ID = None
        else:
            OBS_ID = hdulist['EVENTS'].header['OBS_ID']

        if 'LIVETIME' not in evt_hd:
            LIVETIME = None
        else:
            LIVETIME = hdulist['EVENTS'].header['LIVETIME']

        if 'ONTIME' not in evt_hd:
            ONTIME = None
        else:
            ONTIME = hdulist['EVENTS'].header['ONTIME']

        if 'DETNAME' not in evt_hd:
            DETNAME = None
        else:
            DETNAME = hdulist['EVENTS'].header['DETNAME']

        if 'DATATYPE' not in evt_hd:
            DATATYPE = None
        else:
            DATATYPE = hdulist['EVENTS'].header['DATATYPE']

        if 'CCDSRC' not in evt_hd:
            CCDSRC = None
        else:
            CCDSRC = hdulist['EVENTS'].header['CCDSRC']

        if 'MJDREFI' in evt_hd:
            MJDREF = hdulist['EVENTS'].header['MJDREFI'] + hdulist['EVENTS'].header['MJDREFF']
        elif 'MJDREF' in evt_hd:
            MJDREF = hdulist['EVENTS'].header['MJDREF']
        else:
            logger.error('No reference time in event file, need either MJDREFI or MJDREF keywords')

        evtFileKeyWords = {'TELESCOPE': TELESCOPE, 'INSTRUME': INSTRUME, 'OBS_ID': OBS_ID, 'TSTART': TSTART,
                           'TSTOP': TSTOP, 'LIVETIME': LIVETIME, 'ONTIME': ONTIME, 'TIMESYS': TIMESYS,
                           'MJDREF': MJDREF, 'TIMEZERO': TIMEZERO, 'DATEOBS': DATEOBS, 'DETNAME': DETNAME,
                           'DATATYPE': DATATYPE, 'CCDSRC': CCDSRC}

        # Checking if event file is barycentered
        if TIMESYS != "TDB":
            logger.warning("\n Event file is not barycentered. Proceed with care!")

        # Close the HDUList
        hdulist.close()

        return evtFileKeyWords

    #################################################################
    def read_fpmsel(self):
        """
        Reads FPM_SEL extension from a NICER event file
        :return: FPMSEL_table_condensed - same as FPM_SEL but with total number of detectors per time stamp
        :rtype: pandas.DataFrame
        """
        evtFileKeyWords = self.readEF()
        TELESCOPE = evtFileKeyWords["TELESCOPE"]
        MJDREF = evtFileKeyWords["MJDREF"]

        if TELESCOPE == 'NICER':
            hdulist = self.open_fits()

            # Reading the table
            FPMSEL_table = hdulist["FPM_SEL"].data
            TIME = FPMSEL_table['TIME'] / 86400 + MJDREF
            FPMsel = FPMSEL_table['FPM_SEL']
            FPMon = FPMSEL_table['FPM_ON']
            # Adding up all selected FPMs
            totfpmsel = np.zeros(np.size(TIME))
            for k in range(len(TIME)):
                totfpmsel[k] = (np.sum(FPMsel[k]))
            # Adding up all FPMs that were on
            totfpmon = np.zeros(np.size(TIME))
            for k in range(len(TIME)):
                totfpmon[k] = (np.sum(FPMon[k]))

            FPMSEL_table_condensed = pd.DataFrame(np.vstack((TIME, totfpmsel, totfpmon)).T,
                                                  columns=['TIME', 'TOTFPMSEL', 'TOTFPMON'])

        else:
            logger.error('No FPM selection is possible for non-NICER observations')

        # Close the HDUList
        hdulist.close()

        return FPMSEL_table, FPMSEL_table_condensed

    #################################################################
    def readGTI(self):  # Reading fits event file GTI lists
        """
        Reads GTI table from an event file
        :return: gtiList as a numpy array with first column as START and second column as STOP (in MJD)
        :rtype: numpy.ndarray
        """
        # Reading EF for some necessary keywords
        evtFileKeyWords = self.readEF()
        TELESCOPE = evtFileKeyWords["TELESCOPE"]
        MJDREF = evtFileKeyWords["MJDREF"]
        # Open event file
        hdulist = self.open_fits()

        if TELESCOPE == 'XMM':
            CCDSRC = int(evtFileKeyWords["CCDSRC"])
            if CCDSRC < 10:
                GTIdata = hdulist["STDGTI0" + str(CCDSRC)].data
                ST_GTI = GTIdata.field("START")
                ET_GTI = GTIdata.field("STOP")
                gtiList = (np.vstack((ST_GTI, ET_GTI))).T
            else:
                GTIdata = hdulist["STDGTI" + str(CCDSRC)].data
                ST_GTI = GTIdata.field("START")
                ET_GTI = GTIdata.field("STOP")
                gtiList = (np.vstack((ST_GTI, ET_GTI))).T

        elif TELESCOPE in ['NICER', 'SWIFT', 'NuSTAR', 'IXPE']:
            GTIdata = hdulist["GTI"].data
            ST_GTI = GTIdata.field("START")
            ET_GTI = GTIdata.field("STOP")
            gtiList = (np.vstack((ST_GTI, ET_GTI))).T

        elif TELESCOPE == 'GLAST':
            DATATYPE = hdulist['Primary'].header['DATATYPE']
            GTIdata = hdulist["GTI"].data
            ST_GTI = GTIdata.field("START")
            ET_GTI = GTIdata.field("STOP")
            gtiList = (np.vstack((ST_GTI, ET_GTI))).T
            if DATATYPE == "TTE":
                logger.warning(
                    "Default GTI of GBM TTE file is simply start and end time of day. Run appendGTIsToTTE.py for a quick fix.")
        else:
            logger.error('Check TELESCOP keyword in event file. Likely telescope not supported yet')

        gtiList = gtiList / 86400 + MJDREF
        # Close the HDUList
        hdulist.close()

        return evtFileKeyWords, gtiList

    def build_time_energy_df(self):
        """
        Build dataframe of TIME (MJD) and ENERGY (in keV) - column name remains PI, but it is energy in keV
        """
        # Open fits file and read EVENTS table
        hdulist = self.open_fits()
        tbdata = hdulist['EVENTS'].data
        # Telescope and MJDREF keywords
        header_keywords = self.readEF()
        TELESCOPE = header_keywords['TELESCOPE']
        MJDREF = header_keywords['MJDREF']

        # different column names and coversion factors for different telescopes
        if TELESCOPE in ['NICER', 'SWIFT']:
            dataTP = pd.DataFrame(np.vstack((tbdata.field('TIME'), tbdata.field('PI'))).T, columns=['TIME', 'PI'])
            dataTP['PI'] *= 0.01

        elif TELESCOPE == 'NuSTAR':
            dataTP = pd.DataFrame(np.vstack((tbdata.field('TIME'), tbdata.field('PI'))).T, columns=['TIME', 'PI'])
            dataTP['PI'] = (dataTP['PI'] * 0.04) + 1.6

        elif TELESCOPE == 'XMM':
            dataTP = pd.DataFrame(np.vstack((tbdata.field('TIME'), tbdata.field('PI'))).T, columns=['TIME', 'PI'])
            dataTP['PI'] *= 0.001

        elif TELESCOPE == 'IXPE':
            dataTP = pd.DataFrame(np.vstack((tbdata.field('TIME'), tbdata.field('PI'))).T, columns=['TIME', 'PI'])
            dataTP['PI'] *= 0.04

        elif TELESCOPE == 'GLAST':
            logger.warning(
                "GBM only provides PHAs, and their conversion to energy is non-linear, "
                "hence, eneLow and eneHigh inputs are really PHA values. Use with care!")
            dataTP = pd.DataFrame(np.vstack((tbdata.field('TIME'), tbdata.field('PHA'))).T, columns=['TIME', 'PHA'])

        # Convert TIME to MJD
        dataTP['TIME'] = dataTP['TIME'] / 86400 + MJDREF

        # Close the HDUList
        hdulist.close()

        self.time_energy_df = dataTP
        return self

    def filtenergy(self, eneLow: float, eneHigh: float):
        """
        Filters the [TIME, ENERGY] dataframe according to energy (in keV)
        :param eneLow: low energy cutoff
        :type eneLow: float
        :param eneHigh: high energy cutoff
        :type eneHigh: float
        :return: self, updated time_energy_df attribute filtered for energy
        :rtype: object
        """
        if self.time_energy_df is None:
            raise Exception("TIME ENERGY dataframe is empty - please run build_time_energy_df method first ")
        if 'PI' not in self.time_energy_df.columns:
            raise Exception("NO PI column name to filter against ")
        mask = self.time_energy_df["PI"].between(eneLow, eneHigh)
        self.time_energy_df = self.time_energy_df.loc[mask].copy()
        return self

    def filttime(self, t_start: float | None = None, t_end: float | None = None):
        """
        Filters the [TIME, ENERGY] dataframe according to energy (in keV)
        :param t_start: Filter all times earlier than t_start
        :type t_start: float
        :param t_end: Filter all times later than t_end
        :type t_end: float
        :return: self, updated time_energy_df attribute filtered for time
        :rtype: object
        """
        if self.time_energy_df is None:
            raise Exception("TIME ENERGY dataframe is empty - please run build_time_energy_df first ")
        left = -np.inf if t_start is None else t_start
        right = np.inf if t_end is None else t_end
        mask = self.time_energy_df["TIME"].between(left, right)
        self.time_energy_df = self.time_energy_df.loc[mask].copy()
        return self

    ################################################################################
    def addphasecolEF(self, timMod: str, nonBaryEvtFile: str = None):  # Filtering event file according to energy
        """
        Adds phase column to event file according to timing model (.par file)
        the phase column will be added to the self.evtFile, but also could be added to a non-barycentered event file
        (though see warning in source code)
        :param timMod: name of timing model (.par file)
        :type timMod: str
        :param nonBaryEvtFile: name of non-barycentered event file (default = None)
        :type nonBaryEvtFile: str
        """
        # Reading some necessary keywords
        evtFileKeyWords = self.readEF()

        # Calculating phases and appending the barycentered event file
        # Reading the MJDREF keyword
        MJDREF = evtFileKeyWords["MJDREF"]
        # opening the event file in update mode
        hdulEFPH = fits.open(self.evtFile, mode='update')
        tbdataPH = hdulEFPH['EVENTS'].data
        hdrEventsPH = hdulEFPH['EVENTS'].header

        # Creating phases
        timeMET = tbdataPH.field('TIME')
        timeMJD = timeMET / 86400 + MJDREF
        _, cycleFoldedPhases = calcphase(timeMJD, timMod)

        #####################################
        # Creating an astropy Table that corresponds to the EVENTS table
        baryTable = Table.read(self.evtFile, format='fits', hdu='EVENTS')
        phCol = Column(name='PHASE', data=cycleFoldedPhases)
        baryTable.add_column(phCol)

        # Updating event file
        newhdulEFPH = fits.BinTableHDU(data=baryTable, header=hdrEventsPH, name='EVENTS')
        fits.update(self.evtFile, newhdulEFPH.data, newhdulEFPH.header, 'EVENTS')

        #####################################
        # adding phase column to NON-barycentered event file fits table. Why do this?
        # In some instances, you wish to perform phase-resolved spectroscpoy along with some native filtering,
        # e.g., with NICER specific heasoft tools. The latter are typically performed on mkf files
        # which are not barycenter corrected. Mixing barycenter-corrected time filtering with non-barycenter
        # corrected filtering can get messy. This way, filtering for certain phases and native
        # filtering are done on non-barycentered times and the merged GTIs should be valid.
        # Use with EXTREME care
        if nonBaryEvtFile is not None:
            nonBaryhdulEFPH = fits.open(nonBaryEvtFile, mode='update')
            nonBaryhdrEventsPH = nonBaryhdulEFPH['EVENTS'].header

            # adding same phase column as above (i.e., calculated with barycentered event file)
            nonBaryTable = Table.read(nonBaryEvtFile, format='fits', hdu='EVENTS')
            nonBaryTable.add_column(phCol)

            # Updating NON-barycentered event file
            nonBaryNewhdulEFPH = fits.BinTableHDU(data=nonBaryTable, header=nonBaryhdrEventsPH, name='EVENTS')
            fits.update(nonBaryEvtFile, nonBaryNewhdulEFPH.data, nonBaryNewhdulEFPH.header, 'EVENTS')

        return evtFileKeyWords


def main():
    """
    This runs the method addPhaseColEF from class EvtFileOps
    Included as a script called "addphasecolumn"
    """
    parser = argparse.ArgumentParser(description="Create and append event file with Phase column")
    parser.add_argument("evtFile", help="Name of (X-ray) fits event file", type=str)
    parser.add_argument("timMod", help="Timing model for phase folding, e.g., a .par file", type=str)
    parser.add_argument("-ne", "--nonBaryEvtFile", help="Name of non-barycentered event file",
                        type=str, default=None)
    args = parser.parse_args()

    EvtFileOps(args.evtFile).addphasecolEF(args.timMod, args.nonBaryEvtFile)


if __name__ == '__main__':
    main()
