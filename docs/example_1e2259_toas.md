# Worked Example: End-to-End ToA Measurement for 1E 2259+586

This page provides a complete, end-to-end example of deriving pulse **Times of Arrival (ToAs)** with **CRIMP**, using the magnetar **1E 2259+586**.

The goal of this example is to demonstrate a realistic timing workflow using data and files that are already distributed with CRIMP, and to reproduce the timing behavior reported in
[Younes et al. 2020, ApJ, 896, L42](https://ui.adsabs.harvard.edu/abs/2020ApJ...896L..42Y/abstract).

---

## Overview of the workflow

In this example, we will:

1. Generate a high signal-to-noise pulse profile template
2. Define time intervals over which ToAs will be measured
3. Measure pulse ToAs using the template, timing model, and event data

The three CRIMP command-line tools used here are:

- `templatepulseprofile`
- `timeintervalsfortoas`
- `measuretoas`

You can inspect their available options at any time using the `-h` flag.

---

## Input data and timing model

A prerequisite for any timing analysis is a valid timing model (a `.par` file).
For **1E 2259+586**, a suitable timing model is already provided in [data/1e2259.par](../data/1e2259.par).


This example also uses NICER event files that are distributed with CRIMP:

- A **single high-exposure observation** for template generation 
([data/1e2259_ni1020600110.fits](../data/1e2259_ni1020600110.fits)).
- A **merged event file spanning ~1 year** for ToA measurements
([data/1e2259_nicer.fits](../data/1e2259_nicer.fits)).

---

## Step 1 — Creating a pulse profile template

We first generate a pulse profile template that will be used to measure ToAs.

For this purpose, we use the NICER observation **1020600110**, which has a long exposure and produces a high 
signal-to-noise pulse profile. The corresponding event file is [1e2259_ni1020600110.fits](../data/1e2259_ni1020600110.fits).


```bash
templatepulseprofile 1e2259_ni1020600110.fits 1e2259.par \
    -el 1 -eh 5 \
    -nb 70 \
    -nc 6 \
    -fg 1e2259_template \
    -tf 1e2259_template
```

This command produces a Fourier-series pulse profile template and reports basic goodness-of-fit statistics:

```bash
Template fourier best fit statistics chi2 = 71.21 for dof = 57
Reduced chi2 = 1.25
```

In some cases, the maximum-likelihood fit may converge to a local minimum. When this happens, it is useful to re-run 
the fit using the previously generated template as an initial guess:

```bash
templatepulseprofile 1e2259_ni1020600110.fits 1e2259.par \
    -el 1 -eh 5 \
    -nb 70 \
    -fg 1e2259_template \
    -tf 1e2259_template \
    -it 1e2259_template.txt
```

which sometimes improves convergence:

```bash
Template fourier best fit statistics chi2 = 57.25 for dof = 57
Reduced chi2 = 1.00
```

- The final template is stored in [data/1e2259_template.txt](../data/1e2259_template.txt)
- A PDF visualization of the template is also produced: [data/1e2259_template.pdf](../data/1e2259_template.pdf)

> **Notes**
> - When an initial template is provided via `-it`, the number of harmonics (`-nc`)
> is read from the template file and the command-line value is ignored. Same for the flag `-pm` which defines the model
> (Fourier series (default), wrapped Gaussian (von-Mises), or wrapped Cauchy) to be used to build the template.
> - It is prudent that the event file falls within the validity range of the timing solution. While the solution 
> does not need to be perfect, something reasonable is required to get an accurate representation of the pulse profile 
> shape.

**Meaning of the options used:**
- `-el`, `-eh`: energy cuts in keV
- `-nb`: number of phase bins in the pulse profile
- `-nc`: number of Fourier harmonics in the model
- `-fg`: output PDF figure of the profile and best-fit model
- `-tf`: output text file containing the best-fit template

**More information on using this CLT can be found [here](template_pulse_profile.md).**


---

## Step 2 — Creating time intervals for ToAs

To perform timing over an extended baseline, it is convenient to work with a merged event file spanning many 
observations. For **1E 2259+586**, such a file is already provided [here](../data/1e2259_nicer.fits). This file spans 
approximately one year of NICER observations.

> **Important**\
> When merging event files manually, make sure that:
> - The **EVENTS** table is merged along `TIME`
> - The **GTI** table is merged along `START` / `STOP`
> - Both are sorted in ascending order

We now generate the time intervals that will define individual ToAs:
```bash
timeintervalsfortoas 1e2259_nicer.fits \
    -tc 10000 \
    -wt 1 \
    -el 1 -eh 5 \
    -of timIntToAs_1e2259
```

Example output:

```bash
Total number of time intervals that define the TOAs: 84
```

**Meaning of the options used:**
- `-tc`: target number of counts per ToA
- `-wt`: maximum waiting time between GTIs before force starting a new ToA (in days)
- `-el`, `-eh`: energy cuts in keV
- `-of`: base name for output files

**This command produces:**
- timIntToAs_1e2259.txt — the ToA interval file
- a .log file summarizing the run
- an intermediate _bunches.txt file (which can be ignored)

**Additional optional parameters include:**
- `--max_counts (-mc)`: minimum counts required to keep a ToA interval
- `--max_wait (-mw)`: maximum time separation for merging intervals

These options help prevent very low-S/N ToAs by merging neighboring intervals when needed.

The NICER-related warning printed by the tool can be safely ignored for this example.

---

## Step 3 — Measuring pulse ToAs

With the template, timing model, ToA intervals, and merged event file in place, we can now measure pulse ToAs. We run:

```bash
measuretoas 1e2259_nicer.fits 1e2259.par \
    1e2259_template.txt \
    timIntToAs_1e2259.txt \
    -el 1 -eh 5 \
    -tf ToAs_2259 \
    -mf ToAs_2259 \
    -bm
```
This produces a sequence of ToAs:
```
ToA 0
ToA 1
...
ToA 82
ToA 83
```

**Meaning of the options used:**
- `-tf`: output .txt text file with phase residuals
- `-mf`: output .tim file
- `-bm`: enables brute-force minimization (grid search)

For 1E 2259+586, the pulse profile is double-peaked, with peaks separated by ~0.5 in phase and of comparable height. 
In such cases, maximum-likelihood minimization can converge to the wrong peak. The `-bm` flag ensures that the full 
phase-shift parameter space is explored, avoiding this issue. For simpler, single-peaked pulse profiles, 
this option can usually be omitted to speed up the calculation.

**This step also produces:**
- a .log file summarizing the run 
- a phase-residual plot, e.g., [ToAs_2259_phaseResiduals.pdf](../data/ToAs_2259_phaseResiduals.pdf)

The phase wrap at the end of the plot is expected and indicates that one cycle should be added to the affected ToAs.
At this stage, we have reproduced the upper-right panel of Figure 2 in 
[Younes et al. 2020, ApJ, 896, L42](https://ui.adsabs.harvard.edu/abs/2020ApJ...896L..42Y/abstract).

---

## Next steps

The resulting ToAs can now be used with standard pulsar timing tools, including:

- [Tempo2](https://bitbucket.org/psrsoft/tempo2/src/master/)
- [PINT](https://github.com/nanograv/PINT)
- CRIMP’s `fittoas` command-line tool or `fit_toas.py` module
- custom fitting pipelines

The CRIMP `fittoas` tool currently supports fitting rotational Taylor expansions, glitches, and WAVE **only**.






















