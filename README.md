# CRIMP  
**Code for Rotational Analysis of Isolated Magnetars and Pulsars**

CRIMP is a collection of Python modules and command-line tools for timing analysis of isolated neutron stars observed with high-energy telescopes such as **NICER, Swift, NuSTAR, XMM-Newton, Fermi, IXPE, and Chandra**.

Its primary objective and main strength is the derivation of pulse **Times of Arrival (ToAs)**, which form the basis of pulsar and magnetar timing analyses. CRIMP also includes utilities for periodicity searches (Z² and H-test), local ephemeris construction, pulsed flux estimation, and basic ToA fitting.

The code originated from scripts used in several published X-ray timing studies, including  
Younes et al. (2015, 2020, 2022, 2023), Lower et al. (2023), De Grandis et al. (2022), and related works.

CRIMP is naturally geared toward **magnetar timing**, but should suffice for any **slow, isolated neutron star**. Binary motion and astrometric corrections are not currently implemented.

---

## Installation

CRIMP can be installed locally by cloning the repository and running, from the CRIMP root directory:

```bash
python -m pip install .
```

Use the `-e` flag for an editable install if desired.

The code has been tested on **3.12.10**. All required dependencies are listed in the project’s `.toml` file and will be installed automatically.

---

## Quick start (ToAs in three commands)

After installation, the following command-line tools are typically sufficient to derive pulse ToAs:

- `templatepulseprofile`
- `timeintervalsfortoas`
- `measuretoas`

A minimal end-to-end workflow looks like:

```bash
templatepulseprofile high_snr_eventfile.fits model.par
timeintervalsfortoas events_merged.fits
measuretoas events_merged.fits model.par template.txt toas_gti.txt
```

The exact options will depend on the data set and science goals.

---

## Documentation

📘 **Full documentation is now available in the `docs/` directory.**

In particular:

- **High-level overview and workflow:**  
  [`docs/overview.md`](docs/overview.md)

- **Complete worked example (1E 2259+586):**  
  [`docs/example_1e2259_toas.md`](docs/example_1e2259_toas.md)

These pages describe all prerequisites, assumptions, and provide a fully reproducible end-to-end timing example.

More detailed, tool-specific documentation is being added incrementally.

---

## Acknowledgements

I am grateful for the many discussions over the years with the late **Mark Finger**, **Allyn Tennant** (NASA/MSFC), and especially **Paul Ray** (NRL), whose insights into pulsar timing have been invaluable.

---

## License

This project is distributed under the **MIT License**.  
See [`LICENSE`](LICENSE) for details.

---

## Disclaimer
This code is provided without any expressed or implied warranty.  
Use it only if you understand what it is doing.