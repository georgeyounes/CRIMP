####################################################################################
# readPPTemp.py is a module that reads in a template model of a pulse profile. The
# readPPTempAny is sort of a "wrapper" that directs to the appropriate function
# according to whichever template is being read-in. For now, the allowed templates
# are fourier, vonmises, wrapped cauchy. It returns a dictionary of model parameters
# The template model could be built using the module crtPPTemplateMod.py
####################################################################################

import os
import sys
import argparse

import numpy as np

sys.dont_write_bytecode = True

#########################################################################
## Module to read-in template pulse profile for use in ToA calculation ##
#########################################################################

#########################################
# Reading a template by its header keyword 'model'
def readPPTempAny(tempModPP):

    data_file = open(tempModPP,'r+')
    
    blockModel = ""
    for line in data_file:
        li=line.lstrip()
        if li.startswith("model"):
            blockModel = line
            model = (" ".join(blockModel.split('=')[1].split()))
    if not blockModel:
        raise Exception('The "model" parameter must exist in template file')

    # Direct to appropriate function based on model keyword
    if model.casefold()==str.casefold('fourier'):
        tempModPPparam = readPPTempFour(tempModPP)
    elif model.casefold()==str.casefold('vonmises'):
        tempModPPparam = readPPTempVonMises(tempModPP)
    elif model.casefold()==str.casefold('cauchy'):
        tempModPPparam = readPPTempCauchy(tempModPP)
    else:
        raise Exception('Model {} is not supported yet; fourier, vonmises, cauchy are supported'.format(model))
    
    return tempModPPparam


#########################################
# Specifically reading a fourier template
def readPPTempFour(tempModPP):

    data_file = open(tempModPP,'r+')

    ###############################
    # Reading the model and normalization
    blockModel = ""
    blockNorm = ""
    for line in data_file:
        li=line.lstrip()
        if li.startswith("norm"):
            blockNorm = line
            norm = np.float64(" ".join(blockNorm.split('=')[1].split()))
        elif li.startswith("model"): # These parameters could be made case insensitive
            blockModel = line
            model = (" ".join(blockModel.split('=')[1].split()))
    if not blockNorm:
        raise Exception('The "norm" parameter must exist in template file')
    
    tempModPPparam = {'model':model, 'norm':norm}
    data_file.seek(0)
    
    ###############################
    # Reading fourier parameters
    # First let's determine how many harmonics we have
    nbrOfHarm = np.array([])
    for line in data_file:
        # Stripping lines vertically to create a list of characters
        li=line.lstrip()
        if li.startswith("amp"):
            nbrOfHarm = np.append(nbrOfHarm, (" ".join(line.split('=')[0].split())).split('_')[1])
    data_file.seek(0)

    ###############################
    # We now loop over all harmonics and add them to the dictionary
    for jj in nbrOfHarm:
        # Resetting the read from top of text file - not the most efficient but all I can think of
        data_file.seek(0)
        for line in data_file:
            # Stripping lines vertically to create a list of characters
            li=line.lstrip()
            # Harmonic parameters
            if li.startswith("amp_"+jj):
                ampTmp = np.float64(" ".join(line.split('=')[1].split()))
                tempModPPparam["amp_"+jj] = ampTmp

            elif li.startswith("ph_"+jj):
                phTmp = np.float64(" ".join(line.split('=')[1].split()))
                tempModPPparam["ph_"+jj] = phTmp
            
    if (tempModPPparam.get("amp_1")==None) or (tempModPPparam.get("ph_1")==None):
        raise Exception('Parameter of first harmonic, "amp_1" and "ph_1", must exist in template file')
    
    return tempModPPparam


#########################################
# Specifically reading a von Mises template
def readPPTempVonMises(tempModPP):

    data_file = open(tempModPP,'r+')

    ###############################
    # Reading the model and normalization
    blockModel = ""
    blockNorm = ""
    for line in data_file:
        li=line.lstrip()
        if li.startswith("norm"):
            blockNorm = line
            norm = np.float64(" ".join(blockNorm.split('=')[1].split()))
        elif li.startswith("model"): # These parameters could be made case insensitive
            blockModel = line
            model = (" ".join(blockModel.split('=')[1].split()))
    if not blockNorm:
        raise Exception('The "norm" parameter must exist in template file')
    
    tempModPPparam = {'model':model, 'norm':norm}
    data_file.seek(0)
    
    ###############################
    # Reading vonmises parameters
    # First let's determine how many components we have
    nbrOfComp = np.array([])
    for line in data_file:
        # Stripping lines vertically to create a list of characters
        li=line.lstrip()
        if li.startswith("amp"):
            nbrOfComp = np.append(nbrOfComp, (" ".join(line.split('=')[0].split())).split('_')[1])
    data_file.seek(0)

    ###############################
    # We now loop over all harmonics and add them to the dictionary
    for jj in nbrOfComp:
        # Resetting the read from top of text file - not the most efficient but all I can think of
        data_file.seek(0)
        for line in data_file:
            # Stripping lines vertically to create a list of characters
            li=line.lstrip()
            # Component parameters
            if li.startswith("amp_"+jj):
                ampTmp = np.float64(" ".join(line.split('=')[1].split()))
                tempModPPparam["amp_"+jj] = ampTmp

            elif li.startswith("cen_"+jj):
                phTmp = np.float64(" ".join(line.split('=')[1].split()))
                tempModPPparam["cen_"+jj] = phTmp
            elif li.startswith("wid_"+jj):
                phTmp = np.float64(" ".join(line.split('=')[1].split()))
                tempModPPparam["wid_"+jj] = phTmp
            
    if (tempModPPparam.get("amp_1")==None) or (tempModPPparam.get("cen_1")==None) or (tempModPPparam.get("wid_1")==None):
        raise Exception('Parameter of first component, "amp_1", "cen_1", wid_1, must exist in template file')
    
    return tempModPPparam


#########################################
# Specifically reading a Cauchy template
def readPPTempCauchy(tempModPP):

    data_file = open(tempModPP,'r+')

    ###############################
    # Reading the model and normalization
    blockModel = ""
    blockNorm = ""
    for line in data_file:
        li=line.lstrip()
        if li.startswith("norm"):
            blockNorm = line
            norm = np.float64(" ".join(blockNorm.split('=')[1].split()))
        elif li.startswith("model"): # These parameters could be made case insensitive
            blockModel = line
            model = (" ".join(blockModel.split('=')[1].split()))
    if not blockNorm:
        raise Exception('The "norm" parameter must exist in template file')
    
    tempModPPparam = {'model':model, 'norm':norm}
    data_file.seek(0)
    
    ###############################
    # Reading cauchy parameters
    # First let's determine how many components we have
    nbrOfComp = np.array([])
    for line in data_file:
        # Stripping lines vertically to create a list of characters
        li=line.lstrip()
        if li.startswith("amp"):
            nbrOfComp = np.append(nbrOfComp, (" ".join(line.split('=')[0].split())).split('_')[1])
    data_file.seek(0)

    ###############################
    # We now loop over all harmonics and add them to the dictionary
    for jj in nbrOfComp:
        # Resetting the read from top of text file - not the most efficient but all I can think of
        data_file.seek(0)
        for line in data_file:
            # Stripping lines vertically to create a list of characters
            li=line.lstrip()
            # Component parameters
            if li.startswith("amp_"+jj):
                ampTmp = np.float64(" ".join(line.split('=')[1].split()))
                tempModPPparam["amp_"+jj] = ampTmp
            elif li.startswith("cen_"+jj):
                phTmp = np.float64(" ".join(line.split('=')[1].split()))
                tempModPPparam["cen_"+jj] = phTmp
            elif li.startswith("wid_"+jj):
                phTmp = np.float64(" ".join(line.split('=')[1].split()))
                tempModPPparam["wid_"+jj] = phTmp
            
    if (tempModPPparam.get("amp_1")==None) or (tempModPPparam.get("cen_1")==None) or (tempModPPparam.get("wid_1")==None):
        raise Exception('Parameter of first component, "amp_1", "cen_1", wid_1, must exist in template file')
    
    return tempModPPparam

            
##############
# End Script #
##############

if __name__ == '__main__':

    #############################
    ## Parsing input arguments ##
    #############################
    
    parser = argparse.ArgumentParser(description="Simple script to read in a .txt template pulse profile model")
    parser.add_argument("tempModPP", help="Template model in text format. Could be created with crtPPTemplateMod.py",type=str)
    args = parser.parse_args()

    readPPTempAny(args.tempModPP)
