# Overview of CRIMP

CRIMP is born from a collection of scripts I have been using since some time ago to perform magnetar timing. its
main objective and strength is to derive pulse **time-of-arrival (ToAs)** from high energy telescopes such as NICER, 
Swift-XRT, XMM-Newton, IXPE, NuSTAR, Chandra, etc.

This page describes the standard, high-level workflow for generating pulse ToAs using **CRIMP**.
ToAs are the primary end product of any pulsar timing analysis and serve as the input for building and refining timing 
solutions.

CRIMP provides tools to generate ToAs from event files, given a timing model (.par file), a pulse profile template, and 
user-defined time intervals.

---

## Contents

### Getting started
- [What you need before starting](#what-you-need-before-starting)
  - [Event file](#1--event-file-outside-the-scope-of-crimp)
  - [Timing model (.par file)](#2--timing-model-par-file)
  - [Pulse profile template](#3--high-signal-to-noise-pulse-profile-template)
  - [Time intervals for ToAs](#4--time-intervals-for-toas-toa-gti-file)

### Workflows and examples
- [Typical end-to-end workflow](#typical-end-to-end-workflow)
- [Outputs and next steps](#outputs-and-next-steps)
- [Worked example in detail: 1E 2259+586](example_1e2259_toas.md)

### Command-line tools (CLTs)
- [templatepulseprofile](templatepulseprofile.md)
- [timeintervalsfortoas](timeintervalsfortoas.md)
- [measuretoas](measuretoas.md)
- [diagnosetoas](diagnosetoas.md)
- [phshifttotimfile](phshifttotimfile.md)
- [fittoas](fittoas.md)

### Technical notes and limitations

- [Caveats](#caveats)
- [Inner workings of CRIMP](#few-words-on-what-is-happening-under-the-hood-)
- [Wish list](#wish-list)
---

## What you need before starting

### 1- Event file (outside the scope of CRIMP)

CRIMP operates on **event files**, but it does **not** create or merge them.
Event files should be generated using mission-specific software prior to running CRIMP.

If multiple observations are combined, which is usually the case, the event files must be merged as follows:

- The **EVENTS** table must be merged **along `TIME`**
- The **GTI** table must be merged **along `START` / `STOP`**

`TIME` and `START` / `STOP` should also be in ascending order.
Such a merged and sorted event file can be produced with the [HEASOFT](https://heasarc.gsfc.nasa.gov/docs/software/heasoft/) tools [ftmerge](https://heasarc.gsfc.nasa.gov/lheasoft/help/ftmerge.html) or 
[ftmeld](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/ftmeld.html), [ftmergesort](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/ftmergesort.html), [niobsmerge](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/niobsmerge.html), 
etc. Incorrect merging may lead to invalid phase assignments and unreliable ToAs.

### 2- Timing model (`.par` file)

CRIMP requires a pulsar timing model provided as a Tempo/Tempo2/PINT-style `.par` file.

CRIMP does not yet provide utilities to automatically build timing models.
However, it includes periodicity search tools (H-test and Z-test) that can be used to estimate an initial 
spin frequency (`F0`) at a given reference epoch (`PEPOCH`).

At a minimum, the timing model must include:

- Pulsar name
- Sky position (`RAJ`, `DECJ`)
- Spin frequency (`F0`)
- Reference epochs (`PEPOCH`, `POSEPOCH`)

A minimal example `.par` file might look like:

```text
PSR              J2259+586
RAJ           23:01:08.295
DECJ           58:52:44.45
POSEPOCH           51790.0
F0     0.14328254547263483
PEPOCH   58359.55765869704
```

More complex timing models (including frequency derivatives, glitches, and wave terms) are supported, 
but are not required to get started. On the other hand, binary and astrometric models are not included (See 
[Caveats](#Caveats) section below). 

### 3- High signal-to-noise pulse profile template

ToAs are measured by matching observed pulse profiles to a **template**.
For this reason, a high signal-to-noise pulse profile template is required.

CRIMP includes a command-line tool (CLT) to generate pulse profile templates directly from event data:

- `templatepulseprofile`

**More information on using this CLT can be found [here](templatepulseprofile.md).**

### 4- Time intervals for ToAs (ToA GTI file)

ToAs are measured over discrete time intervals. These intervals are provided to CRIMP via a simple text file 
specifying start and stop times.

CRIMP includes a CLT utility to generate such files:

- `timeintervalsfortoas`

**More information on using this CLT can be found [here](timeintervalsfortoas.md).**

---

## Typical end-to-end workflow

A typical end-to-end sequence of commands might look like:

```bash
templatepulseprofile high_snr_eventfile.fits model.par
timeintervalsfortoas events_merged.fits
measuretoas events_merged.fits model.par template.txt toas_gti.txt
```

The exact options used at each stage will depend on the data set, signal-to-noise ratio, and desired ToA cadence.

**A complete end-to-end example timing 1E 2259+586 can be found [here](example_1e2259_toas.md).**

---

## Outputs and next steps

The primary output of this workflow is a ToA (.tim) file suitable for use in pulsar timing analyses.
These ToAs can then be used to:
- Refine or extend the timing model
- Fit for frequency derivatives or glitches
- Study timing noise
- Compare timing solutions across instruments or epochs

**CRIMP** is typically used iteratively as part of a broader timing ecosystem that includes Tempo/Tempo2/PINT/Tempo3 
(oh wait, there is no Tempo3, or is there?!).

---

## Caveats

Because of its original scientific motivation, **CRIMP** is naturally geared toward the analysis of magnetars, but it 
should also suffice for the analysis of any slow, isolated neutron star.

Currently, **binary motion is not incorporated**, nor are **astrometric corrections**. These features may be included 
at a later stage, likely by interfacing more directly with [PINT](https://github.com/nanograv/PINT).

---

## Few words on what is happening under the hood  
  
CRIMP utilizes maximum likelihood estimate as its fitting engine. To fit a template model to a high S/N pulse profile 
(`templatepulseprofile`), CRIMP utilizes a Gaussian likelihood, while for TOA calculation (`measuretoas`) it utilizes 
a Poisson likelihood. For the latter, data is unbinned, yet we also fit for normalization (not only phase-shift, i.e., 
shape); in practice, this is known as the extended maximum likelihood.  
  
CRIMP also allows for the variation of the pulsed fraction in the template when deriving TOAs through the flag '-va' 
in `measuretoas`. Magnetars go into outbursts and quite often the pulsed fraction of the signal varies, sometimes by 
a large factor. This is important to ensure a good fit to each ToA and in turn a proper uncertainty measurement.  
This flag is also helpful when utilizing NICER data to generate ToAs; NICER has variable background which can sometime
affect the observed pulse fraction of a given source.

---

## Wish list
  
CRIMP is still being actively developed and my wish list is long. Here are a few things I would like to do, in no particular order:  

- Add RXTE data to the list of accepted missions.  
- Work out the inclusion of other timing model parameters, e.g., IFUNC, binary models, astrometry (big task, and will likely use [PINT](https://github.com/nanograv/PINT) at that point)  
- Upload to pypi and allow direct pip install  
- Full documentation covering all available functionality  
- More example usage
