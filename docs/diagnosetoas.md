# diagnosetoas

This page documents the **`diagnosetoas`** command-line tool (CLT) in **CRIMP**.
This tool generates an **interactive diagnostic summary** of measured pulse **Times of Arrival (ToAs)** in the form of an HTML file.

The output is a Plotly-based visualization designed to help users quickly assess the quality, consistency, and reliability of ToAs produced by `measuretoas`.

---

## Purpose

`diagnosetoas` provides a compact, visual overview of the properties of each ToA.
It is intended as a **quality-control and sanity-check tool**, allowing users to identify:

- problematic ToAs
- intervals with low exposure or counts
- changes in count rate or pulsed signal strength
- poor template fits
- phase ambiguities or wraps

This tool does **not** modify ToAs; it is purely diagnostic.

---

## When should I use this tool?

You should run `diagnosetoas` when:

- You have just generated ToAs with `measuretoas`
- You want to visually inspect ToA quality before fitting
- You want a quick overview of a large ToA set

It is strongly recommended to run `diagnosetoas` **before** fitting ToAs with Tempo2, PINT, or `fittoas`.

---

## Required inputs

The command requires a single positional argument:

1. **`ToAs`**  
   A text file containing phase shifts as created with `measuretoas`
   (i.e., the `.txt` ToA file, not the `.tim` file).

---

## Quick example

A minimal example looks like:

```bash
diagnosetoas ToAs_2259.txt
```

To explicitly control the output file name:

```bash
diagnosetoas ToAs_2259.txt -of ToAs_2259_diagnostics
```

---

## Output

### Interactive HTML diagnostic plot

The primary output of this tool is an **interactive HTML file** generated using Plotly:

```
ToADiagnosticsPlot.html
```

(or `<outputFile>.html` if `-of` is provided).

This file can be opened in any modern web browser and supports:
- zooming and panning
- hover information for individual ToAs
- interactive exploration of correlations

---

## Layout of the diagnostic plot

The diagnostic figure is arranged as a **two-column grid**:

- **Left column:** quantities plotted as a function of **ToA index**
- **Right column:** the same quantities plotted as a function of **ToA interval mid-time (MJD)**

Each **row** corresponds to a specific ToA property.

---

## Quantities shown (rows)

From top to bottom, the rows display:

### 1. ToA interval length (days)

Duration of the time interval over which each ToA was measured.

- Helps identify unusually long or short ToAs
- Sensitive to GTI structure and merging behavior

### 2. ToA exposure (seconds)

Total good exposure time contributing to each ToA.

- Low exposure ToAs may have poorly constrained phases
- Useful for spotting low signal-to-noise ToAs

### 3. Number of counts

Total number of photon events used to construct each ToA.

- Should roughly reflect the target count threshold used in `timeintervalsfortoas`
- Deviations may indicate merging or filtering effects

### 4. Count rate (counts/s)

Average count rate during each ToA interval.

- Highlights changes in source brightness or instrumental effects
- Useful for identifying outbursts or anomalous intervals
- For **NICER**, if the `-ce` option was used in `measuretoas`, the rate will 
be corrected for the number of selected detectors per ToA (it may deviate from `numberofcounts / ToAexposure`)  

### 5. H-test power

H-test statistic computed for each ToA interval.

- Provides a measure of pulsation significance on a per-ToA basis
- Low values may indicate weak or undetectable pulsations

### 6. Reduced χ² of the template fit

Reduced chi-squared value of the template fit to the ToA pulse profile.

- Values near unity indicate good fits
- Large values may signal template mismatch or complex pulse evolution

### 7. Phase residuals (cycles)

Measured phase shifts (residuals) in units of pulse cycles.

- This is the primary timing observable
- Phase wraps, trends, or discontinuities are easily visible here
- Error bars reflect phase uncertainty per ToA

---

## How to use this plot effectively

Some common use cases include:

- **Outlier detection:**\
Identify ToAs with unusually low counts, low H-test values, or large χ².\
Identify ToAs with low exposure, derived during a long time interval.
- **Template validation:**\
Systematically high χ² values may indicate that the template is no longer appropriate.
- **Timing discontinuities:**\
Sudden jumps in phase residuals may signal glitches or missed phase wraps.
- **Coverage assessment:**\
The MJD column makes data gaps and cadence changes immediately apparent.

Because the output is interactive, users can zoom, pan, and hover over individual points to inspect values.

---

## Bad ToA exclusion

There are several options to remove bad (flagged) ToAs from further analysis:

- Delete the corresponding rows in the text file containing phase shifts as created with `measuretoas`
(i.e., the `.txt` ToA file, not the `.tim` file).
- Comment out the corresponding rows by placing the symbol `#` at the start of the line
- Also, the user can do either of the above on the text file created with the tool `timeintervalsfortoas`; 
i.e., `<timIntToAs>.txt` which defines the ToAs start and stop times. This way any new runs of `measuretoas` 
will ignore the flagged ToAs

Once bad ToAs have been excluded, a new `.tim` file can be created with the CLT [`phshifttotimfile`](phshifttotimfile.md). 

---

## Options

### `-of`, `--outputFile`

Base name of the output diagnostics file.

- Default is `ToADiagnosticsPlot`
- Produces `<outputFile>.html`

---

## Full option reference

The complete list of options is available via:

```bash
diagnosetoas -h
```

---

## See also

- [`measuretoas`](measuretoas.md)
- [`timeintervalsfortoas`](timeintervalsfortoas.md)
- [`templatepulseprofile`](templatepulseprofile.md)
- [`phshifttotimfile`](phshifttotimfile.md)
- [Worked example: 1E 2259+586](example_1e2259_toas.md)

---

This page serves as the authoritative reference for the `diagnosetoas` command-line tool.
