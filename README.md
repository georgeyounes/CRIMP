CRIMP - Code for Rotational-analysis of Isolated Magnetars and Pulsars
======================================================================

Description
-----------

These are a collection of scripts and modules to perform timing analysis on isolated neutron stars observed with high energy telescopes such as NICER, Swift, NuSTAR, XMM-Newton, Fermi, IXPE. Its main *objective and strength* is to derive pulse time-of-arrival (ToAs). It can be used to perform simple Z^2 and Htest periodicity searches, and 2D Z^2 search for P and P_dot, and to derive root-mean-square pulsed flux and fraction, among other things.


This code is born from a collection of scripts I have been using since some time ago. Most of the timing work in Younes et al. 2015ApJ...809..165Y, 2020ApJ...896L..42Y, 2022ApJ...924L..27Y, 2023NatAs...7..339Y, a reproduction of the Hu et al. 2024Natur.626..500H timing results, amont others, have come from a version of these sample codes.

Because of my science interest, it is naturally geared towards the analysis of magnetars, but should suffice for the analysis of any slow isolated neutron star. Currently, binary motion is not incorporated, nor are any astrometric correction.


## Acknowledgements

I am grateful to countless discussions I have had with the late Mark Finger, Allyn Tenant (NASA/MSFC), and especially Paul Ray (NRL). Thank you for always being available for a discussion on pulsar timing and to answer the many confusing questions that I have had (sure will) about the topic over the years.


## Installation

At the moment, CRIMP can be installed locally after cloning the directory or simply downloading and untarring it. Then from the CRIMP root directory:

```bash
  python -m pip install .
```

You may add the '-e' option to pip to install as an editable package.

The code has been tested on Python 3.9.16, and it requires numpy, matplotlib, pandas, scipy, astropy. It will also install a couple of requirements which are lmfit and plotly.
## Quick example usage

Upon installation, few command line scripts will be available, some of which are really all you need to derive your pulse time-of-arrivals (TOAs). These are **templatepulseprofile**, **timeintervalsfortoas**, and **measuretoas**. You can get their help messages and the list of required and optional arguments with the usual '-h' option.

We shall reproduce the results of [Younes et al. 2020, ApJ...896L..42Y](https://ui.adsabs.harvard.edu/abs/2020ApJ...896L..42Y/abstract) on the magnetar 1E 2259+586. A pre-requisite for any timing analysis is the availability of a timing model (usually called .par file), which, for 1E 2259+586, exists in the folder ./data/1e2259.

#### 1- Creating a template
First, we produce a template pulse profile that shall be used to derive ToAs. For this purpose, we will use the event file from the nicer observation ID 1020600110, which has a long exposure resulting in a high S/N pulse profile. This event file is called "1e2259_ni1020600110.fits" and can be found in the same folder. First, we run

```bash
>> templatepulseprofile 1e2259_ni1020600110.fits 1e2259.par -el 1 -eh 5 -nb 70 -nc 6 -fg 1e2259_template -tf 1e2259_template

Template fourier best fit statistics
 chi2 = 71.21299787883065 for dof = 57
 Reduced chi2 = 1.249350839979485
```

The '-el' and '-eh' flags are event energy cuts in keV, '-nb' flag is for the number of bins in the pulse profile, '-nc' is for the number of Fourier harmonics in the model (by default, the code will fit a Fourier model with 'nc' number of harmonics to the data), '-fg' is to produce a .pdf figure of the pulse profile and best fit model, and '-tf' flag is to produce a .txt file of the best fit results. The command will also print simple statistical properties of the fit. Sometimes, the initial fit will not look good, though maximum likelihood does not easily converge to a global minimum. Rerunning the command with the flag '-it' (initial template) and providing it with the template that was just created:

```bash
>> templatepulseprofile 1e2259_ni1020600110.fits 1e2259.par -el 1 -eh 5 -nb 70 -fg 1e2259_template -tf 1e2259_template -it 1e2259_template.txt 

Template fourier best fit statistics
 chi2 = 57.25146669421473 for dof = 57
 Reduced chi2 = 1.004411696389732
```

The fit is better. Rerunning the above does not change the statistics considerably. Notice here we use 6 Fourier harmonics. One way to choose the appropriate number is through a simple Ftest comparing the fit results for the differing number of harmonics used.

The "1e2259_template.txt" will be overwritten with the new resulting best-fit model. Notice that the '-nc' flag was removed; you do not need to, but bear in mind that it is ignored when an initial template ('-it') is provided; i.e., model to be used and number of harmonics in the model shall be read from the initial-template model ".txt" file. The script will also create a simple .log file with a summary of input and output parameters, results, and any warning that you should pay attention to.
This is what the best-fit model looks like 

[1e2259_template.pdf](data%2F1e2259_template.pdf)

There are few other flags that can be set in this command, one of which is the model to be used. It can support a Fourier series model (default), a wrapped Gaussian (von-Mises), or a wrapped Cauchy distribution. I typically use the former, unless the source pulse has a small duty cycle. consequently, the former is more tested than the two latter.

#### 2- Creating time intervals that define TOAs

Typically, you would like to perform pulsar timing over an extended baseline. This could be anywhere from days to decades. Hence, it is more practical to work with a merged event file of your source of interest spanning some amount of time. This can be produced with the [HEASOFT](https://heasarc.gsfc.nasa.gov/docs/software/heasoft/) tools [ftmerge](https://heasarc.gsfc.nasa.gov/lheasoft/help/ftmerge.html), [ftmergesort](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/ftmergesort.html), [niobsmerge](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/niobsmerge.html), etc. Make sure you merge your event files along the TIME column in the EVENTS table **and** START or STOP column in the GTI table.
Such a file for the magnetar 1E 2259+586 exists in the folder data under "1e2259_nicer.fits" (spanning about 1 year)

We run

```bash
>> timeintervalsfortoas 1e2259_nicer.fits -tc 10000 -wt 1 -el 1 -eh 5 -of timIntToAs_1e2259

[2024-09-16 22:37:39]  WARNING 
 If NICER event files were generaed with HEASOFT version 6.32+,
 it is advisable to correct for the number of selected FPMs with the flag -ce for accurate measurement of count rates
 (buildtimeintervalsToAs.py:269)
Total number of time intervals that define each ToA: 94
```

This command only requires an event file to run. The '-tc' flag defines the total number of counts that should be accumulated for each ToA (in this case 10000 counts). The '-wt' flag enforces the maximum wait time between GTIs before starting a new ToA (in days). The '-el' and '-eh' flags are the same as above. The '-of' defines the name of the output file, in this case "timIntToAs_1e2259.txt". Two more output files are created, a simple '.log' file, and an intermediary '_bunches.txt' file which can be ignored.

The NICER warning above can be safely ignored. It will be covered when detailed documentation is made available.

#### 3- Deriving TOAs

Last step in our analysis is to use the template we created, the TOA time intervals, the .par file we have, and the merged event file to derive our pulse TOAs. We run 

```bash
>> measuretoas 1e2259_nicer.fits 1e2259.par 1e2259_template.txt timIntToAs_1e2259.txt -el 1 -eh 5 -tf ToAs_2259 -mf ToAs_2259 -bm

ToA 0
ToA 1
...
...
ToA 92
ToA 93
```

The '-el' and '-eh' flags are the same as above. The '-tf' flag provides the name of the ".txt" output pulse phase residual file, the '-mf' flag defines the ".tim" file. Finally, the '-bm' (brute minimization) is a boolean flag that forces the use of the brute minimization method (grid-search).

In the case of 1E 2259+586, the pulse is double-peaked, separated by approximately 0.5 in phase, and the two peaks have a similar height. Hence, the maximum likelihood can get stuck in a local minimum around the wrong peak. The '-bm' flag ensures that you grid the full parameter space for the phase-shift to avoid these issues. In the case where the pulse shape of your pulsar is simpler, e.g., single-peaked, this flag can be removed to speed up the TOA calculation.

This script will also produce a simple log file ('.log'), and a plot ('-tf'+"_phaseResiduals.pdf") of the phase residuals in cycles which, for our example of 1E 2259+586, is this

[ToAs_2259_phaseResiduals.pdf](data%2FToAs_2259_phaseResiduals.pdf)

Looks like a glitch. Evidently, the phase wrap at the end is normal, and it implies you need to add one cycle to those TOAs. We just reproduced the upper-right panel of Figure 2 in [Younes et al. 2020, ApJ...896L..42Y](https://ui.adsabs.harvard.edu/abs/2020ApJ...896L..42Y/abstract).

From this point on, you can use Tempo2, PINT, or your own tools to fit the TOAs to whatever timing model you would like.
## Few words on what is happening under the hood

CRIMP utilizes maximum likelihood estimate as its fitting engine. To fit a template model to a high S/N pulse profile (```templatepulseprofile```), CRIMP utilizes a Gaussian likelihood, while for TOA calculation (```measuretoas```) it utilizes a Poisson likelihood. For the latter, data is unbinned, yet we also fit for normalization (not only phase-shift, i.e., shape); in practice, this is known as the extended maximum likelihood.

CRIMP also allows for the variation of the pulsed fraction in the template when deriving TOAs through the flag '-va' in ```measuretoas```. Magnetars go into outbursts and quite often the pulsed fraction of the signal varies, sometimes by a large factor. This is important to ensure a good fit to each TOA and in turn a proper uncertainty measurement.
## Wish list

CRIMP is still in its early development and my wish list is long. Here are a few things I would like to do, in no particular order:

- Add a simple TOA fitting algorithm.
- Develop more the use of the other two templates (an easy task at this point, but needs to be tested).
- Add RXTE data to the list of accepted missions.
- Write module to derive local ephemerides
- Work out the inclusion of other timing model parameters, e.g., IFUNC, binary models, astrometry (big task, and maybe use [PINT](https://github.com/nanograv/PINT) at this point)
- Upload to pypi and allow direct pip install
- Full documentation (covering the already available functionality, which is a little more than the above example)
- More example usage

## License

[MIT](https://choosealicense.com/licenses/mit/)

