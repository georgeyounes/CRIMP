# fittoas

This page documents the **`fittoas`** command-line tool (CLT) in **CRIMP**.
This tool is used to **fit pulse Times of Arrival (ToAs)** to a timing model,
optionally sampling posterior distributions using MCMC.

`fittoas` is designed as a lightweight, flexible timing fitter focused on
magnetars and young isolated neutron stars, **supporting Taylor-expanded
spin evolution, glitches, and WAVE**.

Learning and utilizing [Tempo2](https://bitbucket.org/psrsoft/tempo2/src/master/) and/or [PINT](https://github.com/nanograv/PINT) 
are highly recommended given their far more complete library of models.

---

## Purpose

`fittoas` fits a set of measured ToAs to a timing model provided as a `.par` file.
The fit can be performed using deterministic optimization alone, or augmented
with **MCMC sampling** to explore parameter posteriors.

The tool supports:

- restricting fits to specific time ranges
- applying explicit phase-cycle corrections
- incorporating user-defined priors and bounds
- generating diagnostic plots and MCMC products

---

## When should I use this tool?

You should run `fittoas` when:

- You have produced a `.tim` file using `measuretoas` or `phshifttotimfile`
- You want to refine or extend an existing timing solution
- You need to fit for glitches and/or noise (via WAVE)
- You want posterior distributions for timing parameters

This tool is typically run **after** ToA generation and diagnostics.

---

## Required inputs

The command requires the following positional arguments, in order:

1. **`timfile_path`**  
   Path to a `.tim` file containing measured ToAs.

2. **`parfile`**  
   Initial timing model `.par` file.
   Parameters to be fit must have a trailing `1` flag (Tempo2-style).

3. **`newparfile`**  
   Output `.par` file containing the post-fit timing solution.

---

## Quick example

A minimal deterministic fit:

```bash
fittoas ToAs_2259.tim 1e2259.par 1e2259_postfit.par
```

A more advanced example with MCMC:

```bash
fittoas ToAs_2259.tim 1e2259.par 1e2259_postfit.par \
    --mcmc \
    --corner_plot corner.pdf
```

---

## Time selection and phase handling

### `-ts`, `--t_start`

Start time for the fit (MJD).

- Only ToAs with MJD ≥ `t_start` are included.

### `-te`, `--t_end`

End time for the fit (MJD).

- Only ToAs with MJD ≤ `t_end` are included.

### `-tm`, `--t_mjd`

Apply an explicit **phase-cycle shift** to all ToAs with MJD ≥ `t_mjd`.

- Multiple values may be supplied.
- Shifts are cumulative.

This is commonly used to correct known phase wraps.

### `-md`, `--mode`

Specify whether to **add** or **subtract** one cycle at each `t_mjd`.

- Choices: `add`, `subtract`
- Default behavior is `add`.

---

## Initial conditions and priors

### `-iy`, `--init_yaml`

YAML file defining:

- initial starting points
- priors

This file maps parameter names to values and constraints and is particularly
useful for MCMC runs.

---

## MCMC options

### `-mc`, `--mcmc`, `--no-mcmc`

Enable MCMC sampling using **emcee**.

- Default is `False`.

### `-st`, `--mcmc-steps`

Number of MCMC steps per walker.

- Default: 10000

### `-bu`, `--mcmc-burn`

Number of burn-in steps to discard when flattening chains.

- Default: 500

### `-wa`, `--mcmc-walkers`

Number of MCMC walkers.

- Default: 32

---

## Output and diagnostic products

### `-cp`, `--corner_plot`

Path to save a **corner plot** of posterior distributions.

- Output is a PDF file.
- Only produced when MCMC is enabled.

### `-ch`, `--chain-npy`

Path to save the full MCMC chain as a NumPy `.npy` file.

### `-fl`, `--flat-npy`

Path to save flattened, post burn-in samples as a NumPy `.npy` file.

### `-rp`, `--residual_plot`

Generate a plot of pre- and post-fit residuals

- Useful for visualizing best-fit model behavior.

---

## Output files

Depending on the options used, `fittoas` may produce:

- a post-fit `.par` file
- MCMC chain and flattened samples (`.npy`)
- corner plot PDFs
- Residual plot PDFs

---

## Common use cases

- Refining spin frequency and derivatives
- Modeling glitches
- Modeling noise via WAVE
- Estimating uncertainties via posterior sampling
- Exploring parameter degeneracies

---

## Limitations and scope

`fittoas` currently supports:

- isolated neutron stars
- Taylor-expanded rotational evolution
- glitches
- WAVE for simple noise modeling

It does **not** currently support:

- binary systems
- astrometric fitting
- relativistic timing effects

For more complex use cases, consider utilizing Tempo2 or PINT.

---

## Full option reference

The complete list of options is available via:

```bash
fittoas -h
```

---

## See also

- [`measuretoas`](measuretoas.md)
- [`diagnosetoas`](diagnosetoas.md)
- [`phshifttotimfile`](phshifttotimfile.md)
- [`timeintervalsfortoas`](timeintervalsfortoas.md)

---

This page serves as the authoritative reference for the `fittoas`
command-line tool.
