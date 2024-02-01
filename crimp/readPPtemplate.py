####################################################################################
# readPPtemplate.py is a module that reads in a template model of a pulse profile. The
# readPPtemplate is sort of a "wrapper" that directs to the appropriate function
# according to whichever template is being read-in. For now, the allowed templates
# are fourier, vonmises, wrapped cauchy. It returns a dictionary of model parameters
# The template model could be built using the module pulseprofile.py
####################################################################################

import sys

import numpy as np

sys.dont_write_bytecode = True


def readPPtemplate(tempModPP):
    """
    Reading template model parameters from .txt best fit template model to pulse profile
    :param tempModPP:
    :type tempModPP: str
    :return: tempModPPparam, dictionary of template model parameters
    :rtype: dict
    """
    data_file = open(tempModPP, 'r+')

    blockModel = ""
    for line in data_file:
        li = line.lstrip()
        if li.startswith("model"):
            blockModel = line
            model = (" ".join(blockModel.split('=')[1].split()))
    if not blockModel:
        raise Exception('The "model" parameter must exist in template file')

    # Direct to appropriate function based on model keyword
    if model.casefold() == str.casefold('fourier'):
        tempModPPparam = readPPtempFour(tempModPP)
    elif model.casefold() in [str.casefold('vonmises'), str.casefold('cauchy')]:
        tempModPPparam = readPPtempVonMisesCauchy(tempModPP)
    else:
        raise Exception('Model {} is not supported yet; fourier, vonmises, cauchy are supported'.format(model))

    return tempModPPparam


def readstandard(tempModPP):
    """
    Reading standard keywords, "model" and "norm", from template and calculating number of components/harmonics
    :param tempModPP:
    :type tempModPP: str
    return
            - data_file (io.TextIOBase): file object containing the template file data
            - tempModPPparam (dict): dictionary of "model" and "norm" keywords
            - nbrOfComp (int): number of components/harmonics
    """
    data_file = open(tempModPP, 'r+')

    ###############################
    # Reading the model and normalization
    blockNorm = ""
    for line in data_file:
        li = line.lstrip()
        if li.startswith("norm"):
            blockNorm = line
            norm = np.float64(" ".join(blockNorm.split('=')[1].split()))
        elif li.startswith("model"):  # These parameters could be made case-insensitive
            blockModel = line
            model = (" ".join(blockModel.split('=')[1].split()))
    if not blockNorm:
        raise Exception('The "norm" parameter must exist in template file')

    tempModPPparam = {'model': model, 'norm': norm}
    data_file.seek(0)

    ###############################
    # Reading fourier parameters
    # First let's determine how many harmonics we have
    nbrOfComp = np.array([])
    for line in data_file:
        # Stripping lines vertically to create a list of characters
        li = line.lstrip()
        if li.startswith("amp"):
            nbrOfComp = np.append(nbrOfComp, (" ".join(line.split('=')[0].split())).split('_')[1])

    return data_file, tempModPPparam, nbrOfComp


def readPPtempFour(tempModPP):
    """
    Reading template model parameters from .txt best fit Fourier template model to pulse profile
    :param tempModPP:
    :type tempModPP: str
    :return: tempModPPparam, dictionary of template model parameters
    :rtype: dict
    """
    data_file, tempModPPparam, nbrOfComp = readstandard(tempModPP)

    # We now loop over all harmonics and add them to the dictionary
    for jj in nbrOfComp:
        # Resetting the read from top of text file - not the most efficient but all I can think of
        data_file.seek(0)
        for line in data_file:
            # Stripping lines vertically to create a list of characters
            li = line.lstrip()
            # Harmonic parameters
            if li.startswith("amp_" + jj):
                ampTmp = np.float64(" ".join(line.split('=')[1].split()))
                tempModPPparam["amp_" + jj] = ampTmp

            elif li.startswith("ph_" + jj):
                phTmp = np.float64(" ".join(line.split('=')[1].split()))
                tempModPPparam["ph_" + jj] = phTmp

    if (tempModPPparam.get("amp_1") is None) or (tempModPPparam.get("ph_1") is None):
        raise Exception('Parameter of first harmonic, "amp_1" and "ph_1", must exist in template file')

    return tempModPPparam


def readPPtempVonMisesCauchy(tempModPP):
    """
    Reading template model parameters from .txt best fit von Mises or Cauchy template model to pulse profile
    :param tempModPP:
    :type tempModPP: str
    :return: tempModPPparam, dictionary of template model parameters
    :rtype: dict
    """
    data_file, tempModPPparam, nbrOfComp = readstandard(tempModPP)

    # We now loop over all harmonics and add them to the dictionary
    for jj in nbrOfComp:
        # Resetting the read from top of text file - not the most efficient but all I can think of
        data_file.seek(0)
        for line in data_file:
            # Stripping lines vertically to create a list of characters
            li = line.lstrip()
            # Component parameters
            if li.startswith("amp_" + jj):
                ampTmp = np.float64(" ".join(line.split('=')[1].split()))
                tempModPPparam["amp_" + jj] = ampTmp
            elif li.startswith("cen_" + jj):
                phTmp = np.float64(" ".join(line.split('=')[1].split()))
                tempModPPparam["cen_" + jj] = phTmp
            elif li.startswith("wid_" + jj):
                phTmp = np.float64(" ".join(line.split('=')[1].split()))
                tempModPPparam["wid_" + jj] = phTmp

    if ((tempModPPparam.get("amp_1") is None) or (tempModPPparam.get("cen_1") is None) or
            (tempModPPparam.get("wid_1") is None)):
        raise Exception('Parameter of first component, "amp_1", "cen_1", wid_1, must exist in template file')

    return tempModPPparam
