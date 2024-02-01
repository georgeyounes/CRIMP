####################################################################################
# eventfile.py is a module to perform simple operations on X-ray event files,
# reading useful header words, filter for energy, add a phase column, etc. X-ray
# instruments that are accepted are Swift/XRT, NICER, XMM/EPIC, IXPE, Fermi/GBM,
# NuSTAR
####################################################################################

import sys
import argparse
import numpy as np
import pandas as pd
import warnings

from astropy.table import Table, Column
from astropy.io import fits

# Custom modules
from crimp.calcphase import calcphase

sys.dont_write_bytecode = True


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

        Methods
        -------
        readEF(): reads essential keywords from an event file
        readGTI(): reads GTI table from an event
        filtEneEF(): filters the event list accoring to energy (in keV)
        addPhaseColEF(): adds phase column to event file according to timing model (.par file)
        """

    def __init__(self, evtFile: str):
        """
        Constructs the necessary attribute for the Phases object.

        :param evtFile: name of the fits event file
        :type evtFile: str
        """
        self.evtFile = evtFile

    #################################################################
    def readEF(self):  # Reading fits event file from X-ray satellites
        """
        Reads essential keywords from an event file
        :return: evtFileKeyWords - dictionary of essential keywords
        :rtype: dict
        """
        # Opening the fits file
        hdulist = fits.open(self.evtFile)

        # Reading some essential keywords
        TELESCOPE = hdulist['EVENTS'].header['TELESCOP']
        INSTRUME = hdulist['EVENTS'].header['INSTRUME']
        TSTART = hdulist['EVENTS'].header['TSTART']
        TSTOP = hdulist['EVENTS'].header['TSTOP']
        TIMESYS = hdulist['EVENTS'].header['TIMESYS']
        DATEOBS = hdulist['EVENTS'].header['DATE-OBS']
        # All of the below are mission specific
        OBS_ID = None  # for instance, no OBS_ID for GBM data
        TIMEZERO = None
        LIVETIME = None  # no LIVETIME for GBM data either
        DETNAME = None
        DATATYPE = None
        CCDSRC = None  # This is XMM specific

        # initializing some instrument-specific keywords
        if TELESCOPE == 'GLAST':
            DETNAME = hdulist['EVENTS'].header['DETNAM']
            DATATYPE = hdulist['Primary'].header['DATATYPE']
            MJDREF = hdulist['EVENTS'].header['MJDREFI'] + hdulist['EVENTS'].header['MJDREFF']

        elif TELESCOPE == 'XMM':
            LIVETIME = hdulist['EVENTS'].header['LIVETIME']
            OBS_ID = hdulist['EVENTS'].header['OBS_ID']
            MJDREF = hdulist['EVENTS'].header['MJDREF']
            TIMEZERO = hdulist['EVENTS'].header['TIMEZERO']
            CCDSRC = hdulist['EVENTS'].header['CCDSRC']

        elif TELESCOPE == 'NICER':
            LIVETIME = hdulist['EVENTS'].header['ONTIME']
            OBS_ID = hdulist['EVENTS'].header['OBS_ID']
            MJDREF = hdulist['EVENTS'].header['MJDREFI'] + hdulist['EVENTS'].header['MJDREFF']
            TIMEZERO = hdulist['EVENTS'].header['TIMEZERO']

        elif (TELESCOPE == 'SWIFT') or (TELESCOPE == 'IXPE'):
            LIVETIME = hdulist['EVENTS'].header['LIVETIME']
            OBS_ID = hdulist['EVENTS'].header['OBS_ID']
            MJDREF = hdulist['EVENTS'].header['MJDREFI'] + hdulist['EVENTS'].header['MJDREFF']
            TIMEZERO = hdulist['EVENTS'].header['TIMEZERO']

        else:
            raise Exception('Check TELESCOP keyword in event file. Likely telescope not supported yet')

        evtFileKeyWords = {'TELESCOPE': TELESCOPE, 'INSTRUME': INSTRUME, 'OBS_ID': OBS_ID, 'TSTART': TSTART,
                           'TSTOP': TSTOP,
                           'LIVETIME': LIVETIME, 'TIMESYS': TIMESYS, 'MJDREF': MJDREF, 'TIMEZERO': TIMEZERO,
                           'DATEOBS': DATEOBS,
                           'DETNAME': DETNAME, 'DATATYPE': DATATYPE, 'CCDSRC': CCDSRC}
        return evtFileKeyWords

    #################################################################
    def readGTI(self):  # Reading fits event file GTI lists
        """
        Reads GTI table from an event
        :return: gtiList
        :rtype: numpy.ndarray
        """
        # Reading EF for some necessary keywords
        evtFileKeyWords = self.readEF()
        TELESCOPE = evtFileKeyWords["TELESCOPE"]

        if TELESCOPE == 'XMM':
            hdulist = fits.open(self.evtFile)
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

        elif TELESCOPE in ['NICER', 'SWIFT', 'NUSTAR', 'IXPE']:
            hdulist = fits.open(self.evtFile)
            GTIdata = hdulist["GTI"].data
            ST_GTI = GTIdata.field("START")
            ET_GTI = GTIdata.field("STOP")
            gtiList = (np.vstack((ST_GTI, ET_GTI))).T

        elif TELESCOPE == 'GLAST':
            hdulist = fits.open(self.evtFile)
            DATATYPE = hdulist['Primary'].header['DATATYPE']
            GTIdata = hdulist["GTI"].data
            ST_GTI = GTIdata.field("START")
            ET_GTI = GTIdata.field("STOP")
            gtiList = (np.vstack((ST_GTI, ET_GTI))).T
            if DATATYPE == "TTE":
                warnings.warn(
                    "Default GTI of GBM TTE file is simply start and end time of day. Run appendGTIsToTTE.py for a quick fix.",
                    stacklevel=2)

        else:
            raise Exception('Check TELESCOP keyword in event file. Likely telescope not supported yet')

        return gtiList

    ################################################################################
    def filtenergyEF(self, eneLow: float, eneHigh: float):  # Filtering event file according to energy
        """
        Filters the event list accoring to energy (in keV)
        :param eneLow: low energy cutoff
        :type eneLow: float
        :param eneHigh: high energy cutoff
        :type eneHigh: float
        :return: dataTP_eneFlt, dictionary of TIME and PI, filtered for energy
        :rtype: dict
        """
        # Reading columns TIME and PI (pulse-invariant - proxy for photon energy) from binary table
        hdulist = fits.open(self.evtFile)
        tbdata = hdulist['EVENTS'].data
        # Telescope keyword for energy PI conversion
        TELESCOPE = hdulist['EVENTS'].header['TELESCOP']

        if TELESCOPE == 'NICER':
            piLow = eneLow / 0.01
            piHigh = eneHigh / 0.01
            dataTP = pd.DataFrame(np.vstack((tbdata.field('TIME'), tbdata.field('PI'))).T, columns=['TIME', 'PI'])
            dataTP_eneFlt = dataTP.loc[((dataTP['PI'] >= piLow) & (dataTP['PI'] <= piHigh))]

        elif TELESCOPE == 'SWIFT':
            piLow = eneLow * 100
            piHigh = eneHigh * 100
            dataTP = pd.DataFrame(np.vstack((tbdata.field('TIME'), tbdata.field('PI'))).T, columns=['TIME', 'PI'])
            dataTP_eneFlt = dataTP.loc[((dataTP['PI'] >= piLow) & (dataTP['PI'] <= piHigh))]

        elif TELESCOPE == 'NuSTAR':
            piLow = (eneLow - 1.6) / 0.04
            piHigh = (eneHigh - 1.6) / 0.04
            dataTP = pd.DataFrame(np.vstack((tbdata.field('TIME'), tbdata.field('PI'))).T, columns=['TIME', 'PI'])
            dataTP_eneFlt = dataTP.loc[((dataTP['PI'] >= piLow) & (dataTP['PI'] <= piHigh))]

        elif TELESCOPE == 'XMM':
            piLow = eneLow * 1000
            piHigh = eneHigh * 1000
            dataTP = pd.DataFrame(np.vstack((tbdata.field('TIME'), tbdata.field('PI'))).T, columns=['TIME', 'PI'])
            dataTP_eneFlt = dataTP.loc[((dataTP['PI'] >= piLow) & (dataTP['PI'] <= piHigh))]

        elif TELESCOPE == 'IXPE':
            piLow = eneLow * 25
            piHigh = eneHigh * 25
            dataTP = pd.DataFrame(np.vstack((tbdata.field('TIME'), tbdata.field('PI'))).T, columns=['TIME', 'PI'])
            dataTP_eneFlt = dataTP.loc[((dataTP['PI'] >= piLow) & (dataTP['PI'] <= piHigh))]

        elif TELESCOPE == 'GLAST':
            warnings.warn(
                "GBM only provides PHAs, and their conversion to energy is non-linear, hence, eneLow and eneHigh inputs are really PHA values. Use with care!",
                stacklevel=2)
            dataTP = pd.DataFrame(np.vstack((tbdata.field('TIME'), tbdata.field('PHA'))).T, columns=['TIME', 'PHA'])
            dataTP_eneFlt = dataTP.loc[((dataTP['PHA'] >= eneLow) & (dataTP['PHA'] <= eneHigh))]

        else:
            raise Exception(
                'Check TELESCOP keyword in event file. Likely telescope not supported yet for energy filtering.')

        return dataTP_eneFlt

    ################################################################################
    def addphasecolEF(self, timMod: str, nonBaryEvtFile: str = ""):  # Filtering event file according to energy
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

        # Checking if event file is barycentered 
        if evtFileKeyWords["TIMESYS"] != "TDB":
            raise Exception('Event file is not barycentered. Cannot create phase column')

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
        # adding phase column to barycentered event file fits table, called table
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
        if nonBaryEvtFile:
            nonBaryhdulEFPH = fits.open(nonBaryEvtFile, mode='update')
            nonBaryhdrEventsPH = nonBaryhdulEFPH['EVENTS'].header

            # adding same phase column as above (i.e., calculated with barycentered event file)
            nonBaryTable = Table.read(nonBaryEvtFile, format='fits', hdu='EVENTS')
            nonBaryTable.add_column(phCol)

            # Updating NON-barycentered event file
            nonBaryNewhdulEFPH = fits.BinTableHDU(data=nonBaryTable, header=nonBaryhdrEventsPH, name='EVENTS')
            fits.update(self.evtFile, nonBaryNewhdulEFPH.data, nonBaryNewhdulEFPH.header, 'EVENTS')


def main():
    """
    Main function for eventfile.py
    This runs the method addPhaseColEF from class EvtFileOps
    Included as a script called "addPhaseColumn"
    """
    parser = argparse.ArgumentParser(description="Create and append event file with Phase column")
    parser.add_argument("evtFile", help="Name of (X-ray) fits event file", type=str)
    parser.add_argument("timMod", help="Timing model for phase folding, e.g., a .par file", type=str)
    parser.add_argument("-ne", "--nonBaryEvtFile", help="Name of non-barycentered event file", type=str, default="")
    args = parser.parse_args()

    addPhaseCol = EvtFileOps(args.evtFile)
    addPhaseCol.addphasecolEF(args.timMod, args.nonBaryEvtFile)


if __name__ == '__main__':
    main()
