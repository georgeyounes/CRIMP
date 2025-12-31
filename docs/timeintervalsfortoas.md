# timeintervalsfortoas

This page documents the **`timeintervalsfortoas`** command-line tool (CLT) in **CRIMP**.
This tool is used to generate **time intervals** over which individual pulse **Times of Arrival (ToAs)** will be measured.

The output of this command is a plain text file defining start and stop times for each ToA interval along with some 
useful information about each ToA, e.g., total counts, rate, etc. This file is a required input for `measuretoas`.

---

## Purpose

`timeintervalsfortoas` defines how the event data are segmented in time for ToA measurements.

Rather than measuring a ToA over a fixed duration (e.g., per GTI), this tool allows intervals to be defined based on:
- a target number of counts per ToA
- gaps in Good Time Intervals (GTIs)
- optional merging rules to avoid low signal-to-noise ToAs

The reason why each ToA interval is defined by a total number of counts (rather than duration) is the variable nature 
of the X-ray emission from magnetars. Hence, this provides a flexible and robust way to generate ToAs over long, 
irregularly sampled observations which may cover different source states.

---

## When should I use this tool?

You should run `timeintervalsfortoas` when:

- You are preparing to measure ToAs over a long time baseline
- Your observations are split across many GTIs or pointings
- You want roughly uniform signal-to-noise across ToAs
- You want to avoid ToAs defined over very short or sparsely populated intervals

This tool is typically run **after** a template has been generated, but **before** measuring ToAs.

---

## Required inputs

The command requires a single input:

1. **Event file**  
   A merged FITS event file spanning the time range of interest.

If multiple observations are combined, the event file must be prepared upstream (outside CRIMP):

- The **EVENTS** table must be merged along `TIME`
- The **GTI** table must be merged along `START` / `STOP`
- Both must be sorted in ascending order

Incorrect merging may lead to invalid ToA definitions.

---

## Quick example

A minimal example looks like:

```bash
timeintervalsfortoas events_merged.fits
```

In practice, most users will specify a target number of counts per ToA:

```bash
timeintervalsfortoas events_merged.fits -tc 10000
```

---

## Core options (most users)

These options control how ToA intervals are constructed.

### `-tc`, `--totCtsEachToA`

Desired number of counts per ToA interval.

- This is the primary parameter controlling ToA segmentation.
- Typical values range from a thousand to several tens of thousands, depending on source brightness and pulsed fraction.
- Higher values produce fewer ToAs with higher signal-to-noise.

### `-wt`, `--waitTimeCutoff`

Maximum allowed gap between GTIs (in days) before forcing a new ToA.

- Prevents ToAs from spanning large data gaps.
- Particularly important for sparsely sampled observations.
- Typical values range from hours to days, depending on cadence.

### `-el`, `-eh`

Low and high energy cuts (in keV) applied to the event data.

- Should generally match the energy range used to generate the template.
- Defaults are instrument-agnostic but may not be optimal.

### `-of`, `--outputFile`

Base name of the output files.

This option controls the names of:
- the ToA interval definition file (`.txt`)
- the associated log file (`.log`)

Default base name is `timIntToAs`.

---

## Interval merging options

These options control how short or low-count intervals are handled.

### `-mc`, `--min_counts`

Minimum number of counts required to keep a ToA interval.

- If an interval contains fewer counts than this threshold, it will be merged with a neighboring interval.
- Default value is half of `totCtsEachToA`.

This option helps prevent very low signal-to-noise ToAs.

### `-mw`, `--max_wait`

Maximum time separation (in days) allowed when merging neighboring intervals.

- If the separation exceeds this value, merging will not occur.
- Default value is equal to `waitTimeCutoff`.

---

## Exposure correction option

### `-ce`, `--correxposure`, `--no-correxposure`

Apply a correction to exposure, and in turn to count rates based on the number of selected detector units (FPMs).
Exclusively used for NICER data.

- Recommended for NICER data processed with recent HEASoft versions.
- Default is `False`.
- For merged event files, FPM_SEL must be merged along TIME for accurate exposure and rate corrections.

If unsure, this option can typically be left unset.

---

## Verbosity

### `-v`, `--verbose`

Controls the verbosity level of screen output (minimal at the moment).

- No flag: warnings only
- `-v`: informational messages
- `-vv`: debug-level output

---

## Output files

This command produces:

- A **ToA interval file** (`.txt`)  
  Each row defines the start and stop time of one ToA along with some 
useful information about each ToA, e.g., total counts, rate, etc.

- A **log file** (`.log`)  
  Summarizes input parameters, merging decisions, and warnings.

An intermediate file (`*_bunches.txt`) will also be created and can usually be ignored.

---

## Common pitfalls and recommendations

- Choose `-tc` large enough to ensure reliable ToA uncertainties.
- Ensure energy cuts match those used for template generation (which were presumably picked to maximize S/N).
- Avoid spanning long data gaps by tuning `-wt`.
- Inspect the `.log` file if the number of ToAs is unexpected.

---

## Full option reference

The complete list of options is available via:

```bash
timeintervalsfortoas -h
```

This page provides guidance on how and when to use each option.

---

## See also

- [Worked example: 1E 2259+586](example_1e2259_toas.md)
- [`templatepulseprofile`](templatepulseprofile.md)
- [`measuretoas`](measuretoas.md)

---

This page serves as the authoritative reference for the `timeintervalsfortoas` command-line tool.
