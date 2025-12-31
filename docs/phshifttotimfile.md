# phshifttotimfile

This page documents the **`phshifttotimfile`** command-line tool (CLT) in **CRIMP**.
This tool converts a **phase-shift text file** produced by `measuretoas` into a
**Tempo2 / PINT–compatible `.tim` file**.

While `measuretoas` can directly produce a `.tim` file, `phshifttotimfile` is
useful when phase shifts have been modified, filtered, or otherwise post-processed
before being written to a final timing file. It also allows the inclusion of 
instrument and pulse numbering flags.

---

## Purpose

`phshifttotimfile` takes a text file of phase residuals (output of `measuretoas`) 
and the corresponding `.par` file, and generates a standard `.tim` file.

This allows users to:

- regenerate `.tim` files after editing or correcting phase shifts
- add or update instrument flags
- optionally include explicit pulse numbering
- decouple ToA measurement from `.tim` file generation

The tool performs **no fitting**; it is a formatting and bookkeeping utility.

---

## When should I use this tool?

You should use `phshifttotimfile` when:

- You have a `.txt` ToA file created by `measuretoas` for which you want the corresponding `.tim` file
- You want to regenerate the `.tim` file after ToA post-processing and filtering
- You want tighter control over `.tim` file metadata such as instrument and pulse numbering flags

This tool is typically run **after** `measuretoas` and **before** fitting with
Tempo2, PINT, or CRIMP’s `fittoas`.

---

## Required inputs

The command requires the following positional arguments, in order:

1. **`ToAs`**  
   A `.txt` file of phase shifts created with `measuretoas`
   (e.g., `ToAs.txt`).

2. **`timMod`**  
   The corresponding timing model file (`.par`) with which the above `.txt` file was created.

---

## Quick example

A minimal invocation looks like:

```bash
phshifttotimfile ToAs_2259.txt 1e2259.par
```

This will create a default `.tim` file named `residuals.tim`.

A more explicit invocation might be:

```bash
phshifttotimfile ToAs_2259.txt 1e2259.par \
    -tf ToAs_2259.tim \
    -in NICER \
    -ap
```

---

## Optional arguments

### `-tf`, `--timfile`

Name of the output `.tim` file.

- Default is `residuals.tim`
- The file is written in a format compatible with Tempo2 and PINT

### `-tp`, `--tempModPP`

Name of the best-fit template model used to measure the ToAs.

- Default is `ppTemplateMod`
- This value is written as metadata in the `.tim` file
- Useful for provenance and reproducibility

### `-in`, `--inst`

Instrument keyword written to the `.tim` file.

- Default is `Xray`
- Examples: `NICER`, `XMM`, `NuSTAR`
- Useful when combining ToAs from multiple instruments

### `-ap`, `--addpn`, `--no-addpn`

Add explicit **pulse numbering** to the `.tim` file.

- Default is `False`
- Useful when:
  - phase wraps are present
  - long data gaps exist
  - explicit pulse tracking is required

This option does **not** alter phase shifts; it only affects `.tim` formatting.

### `-cl`, `--clobber`, `--no-clobber`

Allow overwriting an existing `.tim` file.

- Default is `False`
- If `False`, the tool will refuse to overwrite an existing file

---

## Output

The primary output of this tool is a **Tempo2 / PINT–compatible `.tim` file**

---

## Common use cases

- Rebuilding `.tim` files after ToA post-processing/filtering
- Adding pulse numbering
- Standardizing instrument labels across data sets

---

## Full option reference

The complete list of options is available via:

```bash
phshifttotimfile -h
```

---

## See also

- [`measuretoas`](measuretoas.md)
- [`diagnosetoas`](diagnosetoas.md)
- [`timeintervalsfortoas`](timeintervalsfortoas.md)
- CRIMP `fittoas` command-line tool

---

This page serves as the authoritative reference for the `phshifttotimfile`
command-line tool.
