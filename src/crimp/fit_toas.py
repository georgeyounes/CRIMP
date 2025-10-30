"""
Simple module that performs a fit to the pulse ToAs
Models that are allowed currently are taylor expansion (F0, F1, F2, ..., F12) and glitches
Similar to Tempo2 and PINT, a flag following each of those parameters in a .par file will indicate whether they are
fixed (0 or missing flag) or free to vary (1)

Can be called via command line as "fittoas"

Note on the fitting: we optimize for the error in the timing model parameters and not the model parameters themselves,
hence, we fit the phase-residuals
"""

import matplotlib.pyplot as plt

import yaml
import pandas as pd
import numpy as np
from dataclasses import dataclass
import re

from scipy.optimize import minimize
from scipy import stats

import emcee
import corner

import argparse

from crimp.calcphase import calcphase
from crimp.readtimingmodel import ReadTimingModel, patch_par_values, patch_statistics, patch_miscellaneous
from crimp.timfile import readtimfile, PulseToAs


# -------- Starting point --------
def load_toas_for_fit(tim_file: str, parfile: str, t_start: float | None = None, t_stop: float | None = None,
                      t_mjd_phasewrap: float | None = None, mode: str = "add") -> pd.DataFrame:
    """
    Prepare the TOAs from a .tim file
    :param tim_file: .tim TOA file in tempo2 format
    :type tim_file: str
    :param parfile: timing solution as a .par file
    :type parfile: str
    :param t_start: load TOAs from t_start (in MJD)
    :type t_start: float | None
    :param t_stop: stop TOAs at t_stop (in MJD)
    :type t_stop: float | None
    :param t_mjd_phasewrap: phase wrap at t_mjd (in MJD)
    :type t_mjd_phasewrap: float | None
    :param mode: "add" or "remove" 1 phase wrap starting at each t_mjd_phasewrap
    :type mode: str
    :return: Essential TOA info as pandas dataframe ['ToA', 'phase', 'phase_err_cycle']
    :rtype: pd.DataFrame
    """
    # Reading F0 to convert time uncertainties to phase uncertainties
    F0 = readparfile(parfile)[0]['F0']

    # Reading .tim file and store as an object of PulseToAs
    tim_file_df = readtimfile(tim_file)
    pt = PulseToAs(tim_file_df)

    # Filter for time if necessary
    pt.time_filter(t_start, t_stop)

    # Keep things ordered and reindex to 0..N-1
    pt.df = pt.df.sort_values('pulse_ToA').reset_index(drop=True)

    # Build arrays (avoid index alignment issues)
    toas = pd.to_numeric(pt.df['pulse_ToA'], errors='coerce').to_numpy(dtype=float)
    perr = pd.to_numeric(pt.df['pulse_ToA_err'], errors='coerce').to_numpy(dtype=float)

    # Calculate phases from TOAs and parfile
    phases, _ = calcphase(toas, parfile)
    phases = ((phases + 0.5) % 1.0) - 0.5  # keep things between [-0.5, 0.5)
    # Average so that residuals align to 0,
    # for visualization purposes and avoid adding a constant phase to the model - see also model_phases() below
    phases -= np.mean(phases)

    # Uncertainties on phases
    phase_err_cycle = (perr / 1e6) * F0

    toas_to_fit = pd.DataFrame({
        'ToA': toas,
        'phase': phases,
        'phase_err_cycle': phase_err_cycle,
    })

    # Add cycles if prompted
    if t_mjd_phasewrap is not None:
        toas_to_fit = add_phasewrap(toas_to_fit, t_mjd_phasewrap, mode=mode)
        toas_to_fit['phase'] -= np.mean(toas_to_fit['phase'])

    return toas_to_fit


def add_phasewrap(toas_to_fit, t_mjd: float, mode: str = "add") -> pd.DataFrame:
    """
    Shift phases by ± one for rows with column >= each t_mjd.
    :param toas_to_fit: dataframe with columns ['ToA', 'phase', 'phase_err_cycle'] - from load_toas_for_fit()
    :type toas_to_fit: pandas.DataFrame
    :param t_mjd: One or more MJD cut points. For each cut point t, all rows with df[column] >= t are shifted
                  by ± one cycle. With multiple cut points, shifts are cumulative (one cycle per crossing).
    :type t_mjd: float | Sequence[float]
    :param mode: {'add', 'subtract'} Direction of the shift.
    :type mode: str
    :return: PulseTOAs dataframe of ToAs
    :rtype: pandas.DataFrame
    """
    # Normalize t_mjd to a sorted numpy array of unique thresholds
    cuts = np.atleast_1d(np.array(t_mjd, dtype=float))
    # cuts = np.unique(np.sort(cuts))

    if cuts.size == 0:
        return toas_to_fit

    sign = 1.0 if mode.lower() == "add" else -1.0 if mode.lower() == "subtract" else None
    if sign is None:
        raise ValueError("mode must be 'add' or 'subtract'.")

    # Vectorized cumulative shift:
    # For each value x, count how many cuts are <= x (equivalent to x >= t for each t),
    # then shift by count * (±cycle).
    vals = toas_to_fit["ToA"].to_numpy(dtype=float, copy=False)
    counts = np.searchsorted(cuts, vals, side="right")  # number of thresholds <= each value
    toas_to_fit["phase"] += sign * counts

    return toas_to_fit


def readparfile(parfile: str):
    """
    Simple wrapper around ReadTimingModel which reads in a .par file as a list of dictionaries
    :param parfile: timing solution as a .par file
    :type parfile: str
    :return: parameters as a dictionary without flags, and one that includes the flags
    :rtype: tuple
    """
    parfile_params, _, parfile_params_flags = ReadTimingModel(parfile).readfulltimingmodel()
    return parfile_params, parfile_params_flags


# -------- Utilities --------
def list_fit_keys(parfile: dict):
    """
    Collect keys with flag==1 in a deterministic order (e.g., F0,F1,... then glitch terms).
    Simply to understand what is what
    :param parfile: timing solution as a dictionary
    :type parfile: dict
    :return: list of keys items in parfile which are to be fit
    :rtype: list
    """
    keys = [k for k, v in parfile.items()
            if isinstance(v, dict) and "value" in v and "flag" in v and v["flag"] == 1]

    # deterministic: sort human-friendly: F0,F1,... GLEP_1,GLF0_1, ... PEPOCH last, etc.
    def sort_key(k):
        # pull numeric suffix if present (e.g., 'GLEP_2' -> ('GLEP_', 2))
        if "_" in k and k.rsplit("_", 1)[-1].isdigit():
            base, idx = k.rsplit("_", 1)
            return base, int(idx)
        return k, -1

    return sorted(keys, key=sort_key)


def extract_free_params(parfile, yaml_initialguesses: str | None = None):
    """
    Extract free parameter vector
    :param parfile: timing solution as a dictionary
    :type parfile: dict
    :param yaml_initialguesses: yaml file containing initial guesses
    :type yaml_initialguesses: str | None
    :return: Array of the values of free parameters return[0], and their corresponding key names return[1]
    :rtype: tuple
    """
    keys = list_fit_keys(parfile)

    # if yaml file provided initialize parameters from it
    if yaml_initialguesses is not None:
        p0 = initialize_params(keys, yaml_initialguesses)
    else:
        p0 = np.zeros(len(keys), dtype=float)

    return p0, keys  # np.array(p0, dtype=float), keys


def inject_free_params(parfile, pvec: np.ndarray, keys: list):
    """
    Build two dicts from a parsed .par timing model
      - parfile_dict_fit: used by fitter (deltas from base .par parameter values)
      - parfile_dict_full: absolute values after applying the 'base - delta' rule, i.e., full timing solution

    Initialization rules:
      * Zero out all scalar params in parfile_dict_fit EXCEPT:
          - PEPOCH: keep base value
          - GLEP_1: keep base value
      * Nested structures pass through unchanged in BOTH outputs.
      * parfile_dict_full starts as the base 'value' (or structure) everywhere.

    Override rules (applied in order of zip(keys, pvec), i.e., for free-to-vary parameters with flag=1):
      * Skip PEPOCH entirely (never touch).
      * GLEP_1 = pvec[i] (base is ignored) in both parfile_dict_fit and parfile_dict_full
      * All other keys (F0, F1, F2, ..., GLF0_n, GLF1_n, ...) are set to:
          parfile_dict_fit[k]  = pvec[i]
          parfile_dict_full[k] = base - pvec[i]

    :param parfile: dict like {"F0": {"value": 11.19, "flag": 1}, "PEPOCH": {"value": ...}, "WAVE1": {...}, ...}
    :param pvec: iterable of numbers (deltas)
    :param keys: list of parameter names corresponding to pvec
    :returns: (parfile_dict_fit, parfile_dict_full)
    """
    parfile_dict_fit = {}
    parfile_dict_full = {}

    def is_scalar_param(vv):
        return isinstance(vv, dict) and "value" in vv

    def is_glep(name):
        # Match GLEP_1, GLEP_2, ...
        return bool(re.match(r"^GLEP_\d+$", name))

    def is_gltd(name):
        # Match GLTD_1, GLTD_2, ...
        return bool(re.match(r"^GLTD_\d+$", name))

    # Seed outputs with defaults
    for k, v in parfile.items():
        if is_scalar_param(v):
            base = float(v["value"])
            # fit: zero by default except PEPOCH and GLEP_1 retain base
            if k == "PEPOCH" or is_glep(k):
                parfile_dict_fit[k] = base
            else:
                parfile_dict_fit[k] = 0.0
            # full: always start at base
            parfile_dict_full[k] = base
        else:
            # pass through nested structures unchanged
            parfile_dict_fit[k] = v
            parfile_dict_full[k] = v

    # Apply overrides from keys/pvec (except for PEPOCH)
    for k, val in zip(keys, pvec):
        if k == "PEPOCH":
            # Never modify PEPOCH, just in case by mistake a user has a 1 flag next to it
            # (not that it ever happened to me)
            continue
        if k not in parfile:
            raise KeyError(f"Parameter '{k}' not found in parfile.")  # shoud not happen but just in case

        v = parfile[k]
        base = float(v["value"])
        delta = float(val)

        # For overrides
        parfile_dict_fit[k] = delta
        if is_glep(k):
            parfile_dict_full[k] = delta
        elif is_gltd(k):
            parfile_dict_full[k] = base + delta  # GLTD is in days (hence must be added to base)
        else:
            parfile_dict_full[k] = base - delta  # absolute value (base - delta)

    return parfile_dict_fit, parfile_dict_full


# -------- Validation --------
def validate_parfile(parfile):
    """
    Validate a flags-only initial timing model:
      - every scalar param is {"value": float-like, "flag": 0 or 1}
      - nested structures (e.g., WAVE1 dicts) are passed
      - ensure at least one flag==1
    Raises ValueError with a helpful message if invalid.
    :param parfile: timing solution as a .par file
    :type parfile: dict
    """
    if not isinstance(parfile, dict):
        raise ValueError("Initial timing model must be a dict")

    n_fit = 0
    for k, v in parfile.items():

        # Skip nested dicts that are NOT scalar params (e.g., WAVE1 {"A":..., "B":...})
        if isinstance(v, dict) and not (("value" in v) and ("flag" in v)):
            # Treat as non-scalar container → skip validation
            continue

        if not (isinstance(v, dict) and "value" in v and "flag" in v):
            raise ValueError(f"Parameter '{k}' must be a dict with 'value' and 'flag'")

        # numeric value?
        val = v["value"]
        if not isinstance(val, (int, float, np.floating)):
            raise ValueError(f"Parameter '{k}': value must be numeric, got {type(val)}")

        # fit flag 0/1?
        flag = v["flag"]
        if flag not in (0, 1):
            raise ValueError(f"Parameter '{k}': fit flag must be 0 or 1, got {flag}")

        if flag == 1:
            n_fit += 1

    if n_fit == 0:
        raise ValueError("Template has no free parameters (flag==1). Nothing to optimize.")


# -------- Maximum likelihood estimate --------
# -------- Model + Likelihood --------
def model_phases(x_mjd, timmodel):
    """
    Calculate the phases according to a timing model. This is done for fitting purposes:
    We are fitting the phase shifts, and hence our timing model is some error on the timing parameters
    which, when corrected for, the new timing solution will result in whitened residuals
    """
    phases, _ = calcphase(x_mjd, timmodel)  # full phases
    # Average so that after whitening, residuals align to 0,
    # for visualization purposes and avoid adding a constant phase to the model
    phases -= np.mean(phases)
    return phases


def gaussian_nll(y, mu, sigma):
    """
    Returns the Gaussian negative log-likelihood (NLL) of the data y given the model predicted distribution (mu, sigma)
    """
    return -np.sum(stats.norm.logpdf(y, loc=mu, scale=sigma))


# -------- Objective factory (closure) --------
def make_nll(x, y, y_err, parfile, yaml_init=None):
    """
    Creates the NLL
    Returns:
      nll(pvec) -> scalar NLL
      p0 (initial vector)
      keys (fit parameter names)
      template (the same validated template you passed in)
    """
    validate_parfile(parfile)

    p0, keys = extract_free_params(parfile, yaml_init)

    def nll(pvec):
        newparfile_dict, _ = inject_free_params(parfile, pvec, keys)
        mu = model_phases(x, newparfile_dict)
        return gaussian_nll(y, mu, y_err)

    return nll, p0, keys, parfile


def initialize_params(keys: list[str], yaml_init: str) -> tuple:
    """
    Given a list of parameter keys and a YAML file defining initial guesses,
    return a 1D NumPy array of initial parameter values in the same order as keys.
    :param keys: List of parameter names (order will be preserved)
    :param yaml_init: Path to YAML file defining initial guesses
    Return tuple
        - p0 : np.ndarray 1D array of parameter values ordered as in `keys`
        - keys : list[str] The same ordered list of parameter names
    """
    # Read initial guesses (unlike the name says, no need to have priors in the initial guesses -
    # see prior_from_yaml for more info)
    prior = initguess_prior_from_yaml(yaml_init)

    if not prior.initial_guess:
        raise ValueError("No initial guesses found in YAML file.")

    # Check that all requested keys exist
    missing = [k for k in keys if k not in prior.initial_guess]
    if missing:
        raise KeyError(f"Missing initial guesses for: {', '.join(missing)}")

    # Preserve order of keys
    p0 = np.array([prior.initial_guess[k] for k in keys], dtype=float)
    return p0


# -------- MCMC through emcee --------
# -------- Reading initial parameters and /or priors --------
@dataclass
class Prior:
    # If no bounds were supplied at all, this dict can be empty.
    bounds: dict[str, tuple[float, float]]
    # If no guesses were supplied at all, this dict can be empty.
    initial_guess: dict[str, float]

    def log_prior(self, theta: np.ndarray, keys: list[str]) -> float:
        """Uniform (box) priors. Missing key in bounds => improper (no penalty)."""
        for val, name in zip(theta, keys):
            if name in self.bounds:
                lo, hi = self.bounds[name]
                if not (lo < val < hi):
                    return -np.inf
        return 0.0


def initguess_prior_from_yaml(path: str) -> Prior:
    """
    YAML may define, for each parameter:
      - [low, high]                      -> bounds only
      - number                           -> guess only
      - {low: ..., high: ..., guess: ...}-> both (guess optional but see global rule)

    ENFORCED CONSISTENCY:
      - If any param has bounds, all params must have bounds.
      - If any param has a guess,  all params must have a guess.
    """
    data = yaml.safe_load(open(path, "r"))
    if not isinstance(data, dict):
        raise ValueError("YAML must map parameter -> prior/guess")

    params = list(data.keys())
    bounds: dict[str, tuple[float, float]] = {}
    guesses: dict[str, float] = {}

    any_bounds = False
    any_guess = False

    for k, v in data.items():
        if isinstance(v, (list, tuple)):
            # [low, high]
            if len(v) != 2:
                raise ValueError(f"{k}: expected [low, high]")
            low, high = map(float, v)
            if not (low < high):
                raise ValueError(f"{k}: low < high required")
            bounds[k] = (low, high)
            any_bounds = True

        elif isinstance(v, dict):
            # {low, high, (optional) guess}
            has_low = "low" in v
            has_high = "high" in v
            if has_low != has_high:
                raise ValueError(f"{k}: need both 'low' and 'high' if providing bounds")

            if has_low and has_high:
                low, high = float(v["low"]), float(v["high"])
                if not (low < high):
                    raise ValueError(f"{k}: low < high required")
                bounds[k] = (low, high)
                any_bounds = True

            if "guess" in v:
                guesses[k] = float(v["guess"])
                any_guess = True

        elif isinstance(v, (int, float)):
            # scalar -> guess only
            guesses[k] = float(v)
            any_guess = True

        else:
            raise ValueError(f"{k}: unsupported value {v!r}")

    # --- Global consistency checks ---
    if any_bounds:
        missing_bounds = [p for p in params if p not in bounds]
        if missing_bounds:
            raise ValueError(
                "Bounds provided for some parameters but missing for others: "
                + ", ".join(missing_bounds)
            )

    if any_guess:
        missing_guesses = [p for p in params if p not in guesses]
        if missing_guesses:
            raise ValueError(
                "Initial guesses provided for some parameters but missing for others: "
                + ", ".join(missing_guesses)
            )

    return Prior(bounds=bounds, initial_guess=guesses)


# -------- Running emcee --------
def run_mcmc(x, y, yerr, init_parfile: dict, keys: list[str], prior: Prior,
             steps: int = 10000, burn: int = 500, walkers: int = 32,
             corner_pdf: str | None = None, chain_npy: str | None = None, flat_npy: str | None = None,
             progress: bool = True):
    # Initial starting points
    walkers, ndim = walkers, len(keys)
    rng = np.random.default_rng()
    p0 = np.empty((walkers, ndim), dtype=float)
    for i, name in enumerate(keys):
        lo, hi = prior.bounds[name]
        p0[:, i] = rng.uniform(lo, hi, size=walkers)

    # Define the log probability
    def log_probability(theta):
        lp = prior.log_prior(theta, keys)
        if not np.isfinite(lp):
            return -np.inf
        newparfile_dict, _ = inject_free_params(init_parfile, theta, keys)
        mu = model_phases(x, newparfile_dict)
        # gaussian_nll is negative hence the - sign in front (as required by emcee.EnsembleSampler)
        return lp - gaussian_nll(np.asarray(y), np.asarray(mu), np.asarray(yerr))

    # Define sampler and run mcmc
    sampler = emcee.EnsembleSampler(walkers, ndim, log_probability)
    sampler.run_mcmc(p0, steps, progress=progress)

    # Save chain if prompted - don't by default
    chain = sampler.get_chain()  # (steps, walkers, ndim)
    if chain_npy:
        np.save(chain_npy, chain)

    # Flatten after burn + thinning
    discard = max(0, burn)
    thin = 1
    flat = sampler.get_chain(discard=discard, thin=thin, flat=True)
    # Save flattened chain if prompted - don't by default
    if flat_npy:
        np.save(flat_npy, flat)

    # Create corner plot if prompted - don't by default
    if corner_pdf is not None:
        labels = [name for name in keys]
        # default arguments
        default_args = {"show_titles": True, "title_fmt": ".3g", "labels": labels, "quantiles": [0.16, 0.50, 0.84],
                        "fontsize": 14}
        fig = corner.corner(flat, bins=32, smooth=1.5, plot_datapoints=False, fill_contours=True, plot_density=True,
                            **default_args)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.13)
        fig.savefig(corner_pdf+'.pdf', format='pdf', dpi=300)
        plt.close(fig)

    # Summary: median and 16/84 percentiles
    summaries = {}
    for i, name in enumerate(keys):
        q16, q50, q84 = np.percentile(flat[:, i], [16, 50, 84])
        summaries[name] = {
            "median": float(q50),
            "minus": float(q50 - q16),
            "plus": float(q84 - q50),
        }
    return sampler, flat, summaries


# -------- Best-fit dict reconstruction and some statistics--------
def rms_residual(phaseresid, model_phaseresid):
    F1_rms = phaseresid - model_phaseresid
    rms_mod_cycle = np.sqrt((1 / np.size(phaseresid)) * np.sum(F1_rms ** 2))
    return rms_mod_cycle


def chi2_fit(phaseresid, model_phaseresid, phase_err, freeparameters):
    chi2 = np.sum(np.divide(((phaseresid - model_phaseresid) ** 2), phase_err ** 2))
    redchi2 = np.divide(chi2, np.size(phaseresid) - freeparameters)
    dof = np.size(phaseresid) - freeparameters
    model_stats = {'chi2': chi2, 'redchi2': redchi2, 'dof': dof}
    return model_stats


# -------- Pre- and post-fit residual plots --------
def plot_residulas(toas_pre_fit, phase_residulas_post_fit, plotname=None):
    # Creating pre- and post-fit residual plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k', sharex=True,
                            sharey=False, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 0.7]})
    axs = axs.ravel()

    ##################
    # Fomatting axs[0]
    axs[0].tick_params(axis='both', labelsize=14)
    axs[0].xaxis.offsetText.set_fontsize(14)
    axs[0].ticklabel_format(style='plain', axis='y', scilimits=(0, 0))
    axs[0].xaxis.offsetText.set_fontsize(14)
    axs[0].yaxis.offsetText.set_fontsize(14)
    axs[0].set_ylabel(r'$\,\mathrm{Residulas\,(cycle)}$', fontsize=14)

    # Plot data
    axs[0].errorbar(toas_pre_fit['ToA'], toas_pre_fit['phase'], yerr=toas_pre_fit['phase_err_cycle'], color='k',
                    fmt='o', zorder=0, markersize=8, ls='', label='Pre-fit residuals', alpha=0.5)
    # Plot model
    axs[0].plot(toas_pre_fit['ToA'], phase_residulas_post_fit, color='k', linestyle='-', markersize=8, alpha=0.5,
                label='Best-fit model', zorder=10)

    axs[0].legend(fontsize=14)

    ##################
    # Fomatting axs[1]
    axs[1].tick_params(axis='both', labelsize=14)
    axs[1].xaxis.offsetText.set_fontsize(14)
    axs[1].ticklabel_format(style='plain', axis='y', scilimits=(0, 0))
    axs[1].xaxis.offsetText.set_fontsize(14)
    axs[1].yaxis.offsetText.set_fontsize(14)
    axs[1].set_xlabel(r'$\,\mathrm{Time\,(MJD)}$', fontsize=14)
    axs[1].set_ylabel(r'$\,\mathrm{Residulas\,(cycle)}$', fontsize=14)

    axs[1].errorbar(toas_pre_fit['ToA'], toas_pre_fit['phase'] - phase_residulas_post_fit,
                    yerr=toas_pre_fit['phase_err_cycle'], color='k', fmt='o', zorder=10, markersize=8, ls='',
                    label='Post-fit (data-model) residuals', alpha=0.5)

    # Plotting line at 0 residual
    axs[1].axhline(0, color='k', linestyle='-', linewidth=2.0, zorder=10, markersize=2.0, alpha=0.5)

    axs[1].legend(fontsize=14)

    # ax1.set_xlim(-0.5e7, -0.4e7);  # ax1.set_ylim(-1.0, 0.5)

    ###################
    # Finishing touches
    for axis in ['top', 'bottom', 'left', 'right']:
        axs[0].spines[axis].set_linewidth(1.5)
        axs[0].tick_params(width=1.5)
        axs[1].spines[axis].set_linewidth(1.5)
        axs[1].tick_params(width=1.5)

    fig.tight_layout()

    if plotname is None:
        plt.show()
    else:
        fig.savefig(str(plotname) + '.pdf', format='pdf', dpi=300, bbox_inches="tight")
    return


def main():
    parser = argparse.ArgumentParser(description="Script to fit ToAs to a timing model")
    parser.add_argument("timfile_path", help="path to .tim file", type=str)
    parser.add_argument("parfile", help="Initial timing .par file, with fitting flags "
                                        "(1 following at least one parameter)", type=str)
    parser.add_argument("newparfile", help="New post-fit .par file", type=str)

    # Time filtering options
    parser.add_argument("-ts", "--t_start", type=float, default=None,
                        help="Start time for fit (MJD)")
    parser.add_argument("-te", "--t_end", type=float, default=None,
                        help="End time for fit (MJD)")

    # Adding cycle wraps at MJDs
    parser.add_argument("-tm", "--t_mjd", type=float, nargs='+', default=None,
                        help="Shift 1 cycle all ToAs >= t_mjd (MJD) - list of t_mjds is allowed, shift is cumulative")
    parser.add_argument("-md", "--mode", choices=["add", "subtract"], default='add',
                        help="Add or subtract 1 cycle as desired at times > each t_mjd")

    # Fitting and MCMC options
    parser.add_argument("-iy", "--init_yaml", type=str,
                        help="YAML file mapping parameter names to initial starting points and/or bounds")
    parser.add_argument("-mc", "--mcmc", action="store_true", help="Run emcee to sample posteriors")
    parser.add_argument("-st", "--mcmc-steps", type=int, default=10000,
                        help="Number of MCMC steps (default = 10000)")
    parser.add_argument("-bu", "--mcmc-burn", type=int, default=500,
                        help="Burn-in steps to discard when saving flat chain (default=500)")
    parser.add_argument("-wa", "--mcmc-walkers", type=int, default=32,
                        help="Number of walkers (default=32)")
    parser.add_argument("-cp", "--corner-pdf", type=str, default=None,
                        help="Path to save Corner plot PDF (default=None)")
    parser.add_argument("-ch", "--chain-npy", type=str, default=None,
                        help="Path to save full chain as .npy (default=None)")
    parser.add_argument("-fl", "--flat-npy", type=str, default=None,
                        help="Path to save flattened post burn-in samples as .npy (default=None)")
    # Plotting residuals
    parser.add_argument("-ep", "--ephem_plot", help="Plot of local ephemerides", type=str, default=None)

    # Parse all provided arguments
    args = parser.parse_args()

    ###########################
    # reading TOAs and par file
    toas_pre_fit = load_toas_for_fit(args.timfile_path, args.parfile, args.t_start, args.t_end, args.t_mjd, args.mode)
    _, init_parfile = readparfile(args.parfile)
    F0 = init_parfile['F0']['value']

    ############################
    # Validate .par timing model
    validate_parfile(init_parfile)

    START = toas_pre_fit["ToA"].min()
    FINISH = toas_pre_fit["ToA"].max()
    misc_keys = {"START": START, "FINISH": FINISH}

    ########################################################################################
    # MCMC posteriors if desired (currently only way to get uncertainties on fit parameters)
    if args.mcmc:
        keys = list_fit_keys(init_parfile)
        # check for the prior yaml file
        if args.mcmc and args.init_yaml is None:
            parser.error("-iy (or --init_yaml) is required when -mc (or --mcmc) is used")
        # Initilize priors
        prior = initguess_prior_from_yaml(args.init_yaml)
        # Run MCMC with emcee
        print("Running MCMC with emcee...")
        sampler, flat, summaries = run_mcmc(x=toas_pre_fit["ToA"], y=toas_pre_fit["phase"],
                                            yerr=toas_pre_fit["phase_err_cycle"], init_parfile=init_parfile,
                                            keys=keys, prior=prior, steps=args.mcmc_steps, burn=args.mcmc_burn,
                                            walkers=args.mcmc_walkers, corner_pdf=args.corner_pdf,
                                            chain_npy=args.chain_npy, flat_npy=args.flat_npy, progress=True)

        print("Posterior summaries(median - / + 1σ approx using 16th / 84th percentiles):")
        uncertainties_max = {}  # define before loop
        for name, s in summaries.items():
            print(f"  {name}: {s['median']:.8e} -{s['minus']:.2e} +{s['plus']:.2e}")
            # convert to 1-sided uncertainty - choose max (conservative approach)
            uncertainties_max[name] = max(s["minus"], s["plus"])

        # prepare medians in keys order
        med_vec = np.array([summaries[name]['median'] for name in keys], dtype=float)
        post_mcmc_timdict_fit, post_mcmc_timdict = inject_free_params(init_parfile, med_vec, keys)

        source_label = "MCMC (posterior medians)"

        # plot residuals after MCMC run
        phase_residulas_post_fit = model_phases(toas_pre_fit["ToA"], post_mcmc_timdict_fit)
        plot_residulas(toas_pre_fit, phase_residulas_post_fit, args.ephem_plot)

        # Write updated par file with new best-fit values
        patch_par_values(in_path=args.parfile, out_path=args.newparfile, new_values=post_mcmc_timdict,
                         uncertainties=uncertainties_max)
        print("---------------------------")
        print(f"Wrote new timing model to {args.newparfile} using {source_label} values")

        # Measure some statistical terms
        rms_residual_cycle = rms_residual(toas_pre_fit["phase"].to_numpy(), phase_residulas_post_fit)
        nbr_free_param = len(keys)
        simple_stats = chi2_fit(toas_pre_fit["phase"].to_numpy(), phase_residulas_post_fit,
                                toas_pre_fit["phase_err_cycle"].to_numpy(), nbr_free_param)
        print("Statistics of new best-fit:")
        print(f"RMS residual in cycle = {rms_residual_cycle}")
        print(f"RMS residual in seconds = {rms_residual_cycle * (1 / F0)} (assuming F0 = {F0} from input .par file)")
        print(f"Chi2 = {simple_stats['chi2']} for {simple_stats['dof']} dof")
        print(f"reduced Chi2 = {simple_stats['redchi2']}\n")

        # Append statistics to par file
        stats_forparfile = {"CHI2R": simple_stats['redchi2'], "NTOA": len(toas_pre_fit["ToA"]),
                            "TRES": rms_residual_cycle * (1 / F0) * 1.0e6, "CHI2R_DOF": simple_stats['dof']}
        patch_statistics(in_path=args.newparfile, out_path=args.newparfile, new_stats=stats_forparfile)
        patch_statistics(in_path=args.newparfile, out_path=args.newparfile, new_stats=stats_forparfile)
        patch_miscellaneous(in_path=args.newparfile, out_path=args.newparfile, new_misc=misc_keys)
        print(f"Appended best-fit statistical properties to {args.newparfile} par file\n")

    # Otherwise, a simple MLE fit to the residuals
    else:
        ##############################
        # Build the objective function
        nll, p0, keys, tmpl_parfile = make_nll(toas_pre_fit["ToA"], toas_pre_fit["phase"],
                                               toas_pre_fit["phase_err_cycle"], init_parfile, args.init_yaml)

        ########################
        # MLE of above objective
        res = minimize(nll, p0, method="Nelder-Mead", options={"maxiter": int(1e6)})

        ###############################################
        # Recover best-fit timing dict according to MLE
        timing_dict_fit, timing_dict_full = inject_free_params(tmpl_parfile, res.x, keys)

        # Plot MLE fit residuals - first model residuals according to best-fit model
        phase_residulas_post_fit = model_phases(toas_pre_fit["ToA"], timing_dict_fit)
        plot_residulas(toas_pre_fit, phase_residulas_post_fit, args.ephem_plot)

        source_label = "Maximum Likelihood Estimation"
        # write the MEL results into a new (or override existing one) .par file
        patch_par_values(in_path=args.parfile, out_path=args.newparfile, new_values=timing_dict_full)
        print("---------------------------")
        print(f"Wrote new timing model to {args.newparfile} using {source_label} values\n")

        # Measure some statistical terms
        rms_residual_cycle = rms_residual(toas_pre_fit["phase"].to_numpy(), phase_residulas_post_fit)
        nbr_free_param = len(keys)
        simple_stats = chi2_fit(toas_pre_fit["phase"].to_numpy(), phase_residulas_post_fit,
                                toas_pre_fit["phase_err_cycle"].to_numpy(), nbr_free_param)

        print("Statistics of new best-fit:")
        print(f"RMS residual in cycle = {rms_residual_cycle}")
        print(f"RMS residual in seconds = {rms_residual_cycle * (1 / F0)} (assuming F0 = {F0} from input .par file)")
        print(f"Chi2 = {simple_stats['chi2']} for {simple_stats['dof']} dof")
        print(f"reduced Chi2 = {simple_stats['redchi2']}")

        # Append statistics to par file
        stats_forparfile = {"CHI2R": simple_stats['redchi2'], "NTOA": len(toas_pre_fit["ToA"]),
                            "TRES": rms_residual_cycle * (1 / F0) * 1.0e6, "CHI2R_DOF": simple_stats['dof']}
        patch_statistics(in_path=args.newparfile, out_path=args.newparfile, new_stats=stats_forparfile)
        patch_miscellaneous(in_path=args.newparfile, out_path=args.newparfile, new_misc=misc_keys)
        print(f"Appended best-fit statistical properties to {args.newparfile} par file\n")


if __name__ == '__main__':
    main()
