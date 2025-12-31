# templatepulseprofile

This page documents the **`templatepulseprofile`** command-line tool (CLT) in **CRIMP**.
This tool is used to generate a high signal-to-noise **pulse profile template**, which is a required input for measuring pulse Times of Arrival (ToAs).

The template represents the intrinsic pulse shape of the source and is obtained by folding event data using an existing timing model.

---

## Purpose

`templatepulseprofile` builds a parametric model of the pulse profile from event data.
This model is later used by `measuretoas` to determine phase shifts and uncertainties when deriving ToAs.

In CRIMP, the default pulse profile model is a **Fourier series**, though alternative wrapped distributions 
(von Mises and wrapped Cauchy) are also supported.

---

## When should I use this tool?

You should run `templatepulseprofile` when:

- You are starting a new timing analysis and need a template
- You have new, higher-quality data that can improve an existing template
- The pulse profile has evolved significantly (e.g., during a magnetar outburst)
- You want to experiment with different pulse profile models

Typically, templates are generated from **high signal-to-noise** observations.

---

## Required inputs

The command requires:

1. **Event file**  
   A mission-specific FITS event file (e.g., NICER, XMM-Newton, NuSTAR), already barycenter-corrected if appropriate.

2. **Timing model (`.par` file)**  
   A Tempo/Tempo2/PINT-style timing model providing at least a reasonable rotational solution over the time span of the event file.

---

## Quick example

A minimal example looks like:

```bash
templatepulseprofile events.fits model.par
```

In practice, most users will apply energy cuts and control the pulse profile resolution:

```bash
templatepulseprofile events.fits model.par -el 1 -eh 5 -nb 64
```

---

## Core options (most users)

These are the options most commonly used in practice.

### `-el`, `-eh`

Lower and upper energy cuts (in keV) applied to the event data.

- Use these to restrict the analysis to the energy range where pulsations are strongest.
- The optimal range is source- and instrument-dependent.

### `-nb`

Number of phase bins used to build the pulse profile.

- Typical values are 32–128.
- Larger values provide higher resolution but require higher S/N.

### `-nc`

Number of components in the model (either Fourier harmonics, or vonmises/cauchy components).

- Typical values range from 1 to 8.
- More harmonics allow for more complex pulse shapes.
- ***Excessively*** large values may overfit noise (not recommended).

We recommend utilizing an F-test to assess the number of components statistically required by the profile.
At the moment, this is not implemented in CRIMP, but all the statistical properties required to run the F-test 
are output by the tool.

> **Note**  
> This option is ignored if an initial template is provided via `-it`.

### `-fg`

Base name for the output **PDF figure** showing the pulse profile and best-fit model.

A file named `<fg>.pdf` will be produced.

### `-tf`

Base name for the output **text file** containing the best-fit template parameters.

A file named `<tf>.txt` will be produced and can be reused as an initial template.

---

## Model configuration options

### `-pm`

Pulse model utilized to fit the profile. Supported models are:

- `fourier` (default)
- `vonmises` (wrapped Gaussian)
- `cauchy` (wrapped Cauchy)

For more peaked profiles (small duty cycle), vonmises or cauchy are recommended. For broad profiles, 
Fourier series is more appropriate.

> **Note**  
> This option is ignored if an initial template is provided via `-it`.

### `-it`

Initial template file.

- Use this option to provide an existing template as a starting point for the fit.
- This is useful if the optimizer converges to a local minimum.
- When provided, the model type and number of harmonics are read from the template file.

---

## Special case options

### `-fp`
                        
Flag to fix the component phases in the input initial template model (-it). Default = False.

When deriving ToAs from different mission detectors (XTI, XRT, PN, etc.) for the same source, one could adjust the 
template to match the different detector characteristics, e.g., effective area, background level, etc., yet while 
forcing the pulse peaks to maitain the same phases.

> **Note**  
> This option is valid if the ToAs are measured in a similar energy band across detectors. If the ToAs are measured at
> different energy bands, e.g., 2-10 keV and 50-100 keV, then it is best to produce separate templates since the pulse 
> shape may be energy dependent. 

---

## Verbosity

### `-v`, `--verbose`

Controls the verbosity level of screen output (minimal at the moment).

- No flag: warnings only
- `-v`: informational messages
- `-vv`: debug-level output

---

## Output files

Depending on the options used, this command may produce:

- A **template text file** (`.txt`) containing best-fit parameters
- A **PDF figure** showing the pulse profile and model
- A **log file** summarizing inputs, options, and fit statistics

If output files already exist, they will be overwritten.

---

## Common pitfalls and recommendations

- Always use **high S/N data** to generate templates.
- Ensure the timing model is valid over the event file time range.
- If the fit appears poor, try:
  - increasing `-nc`
  - changing `-pm`
  - re-running with `-it` using the previously generated template
- Avoid overfitting by using unnecessarily large numbers of harmonics.

---

## Full option reference

The complete list of available options can be obtained at any time via:

```bash
templatepulseprofile -h
```

This page documents the meaning and recommended usage of all options exposed by the command-line interface.

---

## See also

- [Worked example: 1E 2259+586](example_1e2259_toas.md)
- [`timeintervalsfortoas`](time_intervals_for_toas.md)
- [`measuretoas`](measuretoas.md)

---

This page serves as the authoritative reference for the `templatepulseprofile` command-line tool.
