"""
Simple module that performs a fit to the pulse ToAs
Models that are allowed currently are taylor expansion (F0, F1, F2, ..., F12) and glitches
Similar to Tempo2 and PINT, a flag following each of those parameters in a .par file will indicate whether they are
fixed (0 or missing flag) or free to vary (1)

Can be called via command line as "fittoas"
"""

import matplotlib.pyplot as plt

import yaml
import pandas as pd
import numpy as np
from dataclasses import dataclass

from scipy.optimize import minimize

import emcee
import corner

import argparse

from crimp.calcphase import calcphase
from crimp.readtimingmodel import ReadTimingModel, patch_par_values, patch_statistics
from crimp.timfile import readtimfile, PulseToAs


# -------- Starting point --------
def load_toas_for_fit(tim_file: str, parfile: str, t_start: float | None = None, t_stop: float | None = None,
                      t_mjd_phasewrap: float | None = None, mode: str = "add") -> pd.DataFrame:
    """

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
    phase = np.asarray(model_phases(toas, parfile), dtype=float)

    toas_to_fit = pd.DataFrame({
        'ToA': toas,
        'phase': phase,
        'phase_err_cycle': (perr / 1e6) * F0,  # assuming pulse_ToA_err in microseconds
    })

    # Add cycles if prompted
    if t_mjd_phasewrap is not None:
        toas_to_fit = add_phasewrap(toas_to_fit, t_mjd_phasewrap, mode=mode)
        # re-calibrates things to 0 residuals which is what we compare to - see model_phases
        toas_to_fit['phase'] -= toas_to_fit['phase'].mean()

    return toas_to_fit


def readparfile(parfile: str):
    """
    Simple wrapper around ReadTimingModel which reads in a .par file as a list of dictionaries
    :param parfile: timing solution as a .par file
    :type parfile: str
    :return: parameters as a dictionary without flags, and one that includes the flags
    :rtype: list
    """
    parfile_params, _, parfile_params_flags = ReadTimingModel(parfile).readfulltimingmodel()
    return parfile_params, parfile_params_flags


# -------- Utilities --------
def list_fit_keys(parfile: str | dict):
    """
    Collect keys with flag==1 in a deterministic order (e.g., F0,F1,... then glitch terms).
    Simply to understand what is what
    :param parfile: timing solution as a .par file
    :type parfile: str
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


def extract_free_params(parfile):
    """
    Extract free parameter vector
    :param parfile: timing solution as a .par file
    :type parfile: str | dict
    :return: Array of the values of free parameters list[0], and their corresponding key names list[1]
    :rtype: list
    """
    keys = list_fit_keys(parfile)
    p0 = [parfile[k]["value"] for k in keys]
    return np.array(p0, dtype=float), keys


def inject_free_params(parfile, pvec, keys):
    """
    Return a *new* full timdict with updated 'value' for fit keys.
    Also flatten back to the simple format expected by calcphase:
      {"F0": 11.19, "F1": -1.2e-11, "WAVE1": {"A":..., "B":...}, ...}
    :param parfile: timing solution as a .par file
    :type parfile: str | dict
    :param pvec: Array of the values of free parameters
    :type pvec: numpy.ndarray
    :param keys: Corresponding keys of pvec
    :type keys: list
    :return: *new* timdict with updated values for fit parameters
    :rtype: dict
    """
    new_parfile_dict = {}
    for k, v in parfile.items():
        if isinstance(v, dict) and "value" in v and "flag" in v:
            new_parfile_dict[k] = float(v["value"])  # default
        else:
            new_parfile_dict[k] = v  # nested structures (WAVE*, etc.)
    for k, val in zip(keys, pvec):
        new_parfile_dict[k] = float(val)
    return new_parfile_dict


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
    cuts = np.unique(np.sort(cuts))

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


# -------- Validation --------
def validate_parfile(parfile):
    """
    Validate a flags-only initial timing model:
      - every scalar param is {"value": float-like, "flag": 0 or 1}
      - nested structures (e.g., WAVE1 dicts) are passed
      - ensure at least one flag==1
    Raises ValueError with a helpful message if invalid.
    :param parfile: timing solution as a .par file
    :type parfile: str
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


# -------- MCMC through emcee --------
# -------- Model + Likelihood --------
def model_phases(x_mjd, parfile):
    phases, _ = calcphase(x_mjd, parfile)
    phases -= np.round(phases)
    return phases - np.mean(phases)  # this calibrates things to 0 residuals which is what we compare to


def gaussian_nll(y, mu, sigma):
    sigma = np.asarray(sigma, dtype=float)
    return 0.5 * np.sum(((y - mu) / sigma) ** 2 + np.log(2.0 * np.pi * sigma ** 2))


# -------- Objective factory (closure) --------
def make_nll(x, y_err, parfile):
    """
    Returns:
      nll(pvec) -> scalar NLL
      p0 (initial vector)
      keys (fit parameter names)
      template (the same validated template you passed in)
    """
    validate_parfile(parfile)

    p0, keys = extract_free_params(parfile)

    def nll(pvec):
        newparfile_dict = inject_free_params(parfile, pvec, keys)
        mu = model_phases(x, newparfile_dict)
        return gaussian_nll(np.zeros(len(mu)), np.asarray(mu), np.asarray(y_err))

    return nll, p0, keys, parfile


# -------- Reading priors --------
@dataclass
class Prior:
    bounds: dict[str, tuple[float, float]]

    def log_prior(self, theta: np.ndarray, keys: list[str]) -> float:
        # Uniform box priors; missing keys -> 0 (improper prior)
        for val, name in zip(theta, keys):
            if name in self.bounds:
                lo, hi = self.bounds[name]
                if not (lo < val < hi):
                    return -np.inf
        return 0.0


def prior_from_yaml(path: str) -> Prior:
    """
    Read priors from a .yaml file as [low, high] - only flat priors are allowed. E.g.,
    F0: [0.1118, 0.1120]
    F1: [-2e-11, -1e-11]
    GLF0_1: [1e-5, 2e-5]
    """
    data = yaml.safe_load(open(path, "r"))
    if not isinstance(data, dict):
        raise ValueError("YAML must map parameter -> [low, high]")

    b: dict[str, tuple[float, float]] = {}
    for k, v in data.items():
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            raise ValueError(f"{k}: expected [low, high]")
        low, high = map(float, v)
        if not (low < high):
            raise ValueError(f"{k}: low < high required")
        b[k] = (low, high)
    return Prior(bounds=b)


# -------- Running emcee --------
def run_mcmc(x, yerr, init_parfile: dict, keys: list[str], prior: Prior,
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
        newparfile_dict = inject_free_params(init_parfile, theta, keys)
        mu = model_phases(x, newparfile_dict)
        # gaussian_nll is negative hence the - sign in front (as required by emcee.EnsembleSampler)
        return lp - gaussian_nll(np.zeros(len(mu)), np.asarray(mu), np.asarray(yerr))

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
        fig.savefig(corner_pdf, format='pdf', dpi=300)
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
def bestfit_timing_dict(result, keys, parfile):
    return inject_free_params(parfile, result.x, keys)


def rms_residual(phase_residulas):
    F1_rms = phase_residulas
    rms_mod_cycle = np.sqrt((1 / np.size(phase_residulas)) * np.sum(F1_rms ** 2))
    return rms_mod_cycle


def chi2_fit(phase, phase_err, freeparameters):
    chi2 = np.sum(np.divide((phase ** 2), phase_err ** 2))
    redchi2 = np.divide(chi2, np.size(phase) - freeparameters)
    dof = np.size(phase) - freeparameters
    model_stats = {'chi2': chi2, 'redchi2': redchi2, 'dof': dof}
    return model_stats


# -------- Pre- and post-fit residual plots --------
def plot_residulas(toas_pre_fit, phase_residulas_post_fit, plotname=None):
    # Creating pre- and post-fit residual plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k', sharex=True,
                            sharey=False, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1]})
    axs = axs.ravel()

    ##################
    # Fomatting axs[0]
    axs[0].tick_params(axis='both', labelsize=14)
    axs[0].xaxis.offsetText.set_fontsize(14)
    axs[0].ticklabel_format(style='plain', axis='y', scilimits=(0, 0))
    axs[0].xaxis.offsetText.set_fontsize(14)
    axs[0].yaxis.offsetText.set_fontsize(14)
    axs[0].set_ylabel(r'$\,\mathrm{Residulas\,(cycle)}$', fontsize=14)

    axs[0].errorbar(toas_pre_fit['ToA'], toas_pre_fit['phase'], yerr=toas_pre_fit['phase_err_cycle'], color='k',
                    fmt='o', zorder=0, markersize=8, ls='', label='Pre-fit residuals', alpha=0.5)

    # Plotting line at 0 residual
    axs[0].axhline(0, color='k', linestyle='-', linewidth=2.0, zorder=10, markersize=2.0)

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

    axs[1].errorbar(toas_pre_fit['ToA'], phase_residulas_post_fit, yerr=toas_pre_fit['phase_err_cycle'], color='k',
                    fmt='o', zorder=10, markersize=8, ls='', label='Post-fit residuals', alpha=0.5)

    # Plotting line at 0 residual
    axs[1].axhline(0, color='k', linestyle='-', linewidth=2.0, zorder=10, markersize=2.0)

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

    # MCMC options
    parser.add_argument("-mc", "--mcmc", action="store_true", help="Run emcee to sample posteriors")
    parser.add_argument("-pr", "--prior-yaml", type=str,
                        help="YAML file mapping parameter names to [low, high]")
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

    args = parser.parse_args()

    # reading TOAs and par file
    toas_pre_fit = load_toas_for_fit(args.timfile_path, args.parfile, args.t_start, args.t_end, args.t_mjd, args.mode)
    _, init_parfile = readparfile(args.parfile)
    F0 = init_parfile['F0']['value']

    # Validate .par timing model
    validate_parfile(init_parfile)

    # Build the objective
    nll, p0, keys, tmpl = make_nll(toas_pre_fit["ToA"], toas_pre_fit["phase_err_cycle"], init_parfile)

    # MLE of above objective
    res = minimize(nll, p0, method="Nelder-Mead", options={"maxiter": int(1e6)})

    # Recover best-fit timing dict
    timdict_best = bestfit_timing_dict(res, keys, tmpl)

    # MCMC posteriors if desired (currently only way to get uncertainties on fit parameters)
    if args.mcmc:
        # check for the prior yaml file
        if args.mcmc and args.prior_yaml is None:
            parser.error("-pr (or --prior-yaml) is required when -mc (or --mcmc) is used")
        # Initilize priors
        prior = prior_from_yaml(args.prior_yaml)
        # Run MCMC with emcee
        print("Running MCMC with emcee...")
        sampler, flat, summaries = run_mcmc(x=toas_pre_fit["ToA"].to_numpy(),
                                            yerr=toas_pre_fit["phase_err_cycle"].to_numpy(), init_parfile=init_parfile,
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
        post_mcmc_timdict = inject_free_params(init_parfile, med_vec, keys)
        source_label = "MCMC (posterior medians)"

        # plot residuals after MCMC run
        phase_residulas_post_fit = model_phases(toas_pre_fit["ToA"], post_mcmc_timdict)
        plot_residulas(toas_pre_fit, phase_residulas_post_fit, args.ephem_plot)

        # Write updated par file with new best-fit values
        patch_par_values(in_path=args.parfile, out_path=args.newparfile, new_values=post_mcmc_timdict,
                         uncertainties=uncertainties_max)
        print("---------------------------")
        print(f"Wrote new timing model to {args.newparfile} using {source_label} values")

        # Measure some statistical terms
        rms_residual_cycle = rms_residual(phase_residulas_post_fit)
        nbr_free_param = len(keys)
        simple_stats = chi2_fit(phase_residulas_post_fit, toas_pre_fit["phase_err_cycle"].to_numpy(), nbr_free_param)
        print("Statistics of new best-fit:")
        print(f"RMS residual in cycle = {rms_residual_cycle}")
        print(f"RMS residual in seconds = {rms_residual_cycle * (1/F0)} (assuming F0 = {F0} from input .par file)")
        print(f"Chi2 = {simple_stats['chi2']} for {simple_stats['dof']} dof")
        print(f"reduced Chi2 = {simple_stats['redchi2']}\n")

        # Append statistics to par file
        stats_forparfile = {"CHI2R": simple_stats['redchi2'], "NTOA": len(toas_pre_fit["ToA"]),
                            "TRES": rms_residual_cycle * (1/F0) * 1.0e6, "CHI2R_DOF": simple_stats['dof']}
        patch_statistics(in_path=args.newparfile, out_path=args.newparfile, new_stats=stats_forparfile)
        print(f"Appended best-fit statistical properties to {args.newparfile} par file\n")

    else:
        # Plot MLE fit residuals
        phase_residulas_post_fit = model_phases(toas_pre_fit["ToA"], timdict_best)
        plot_residulas(toas_pre_fit, phase_residulas_post_fit, args.ephem_plot)

        source_label = "Maximum Likelihood Estimation"
        # write the MEL results into a new (or override existing one) .par file
        patch_par_values(in_path=args.parfile, out_path=args.newparfile, new_values=timdict_best)
        print("---------------------------")
        print(f"Wrote new timing model to {args.newparfile} using {source_label} values\n")

        # Measure some statistical terms
        rms_residual_cycle = rms_residual(phase_residulas_post_fit)
        nbr_free_param = len(keys)
        simple_stats = chi2_fit(phase_residulas_post_fit, toas_pre_fit["phase_err_cycle"].to_numpy(), nbr_free_param)
        print("Statistics of new best-fit:")
        print(f"RMS residual in cycle = {rms_residual_cycle}")
        print(f"RMS residual in seconds = {rms_residual_cycle * (1/F0)} (assuming F0 = {F0} from input .par file)")
        print(f"Chi2 = {simple_stats['chi2']} for {simple_stats['dof']} dof")
        print(f"reduced Chi2 = {simple_stats['redchi2']}")

        # Append statistics to par file
        stats_forparfile = {"CHI2R": simple_stats['redchi2'], "NTOA": len(toas_pre_fit["ToA"]),
                            "TRES": rms_residual_cycle * (1/F0) * 1.0e6, "CHI2R_DOF": simple_stats['dof']}
        patch_statistics(in_path=args.newparfile, out_path=args.newparfile, new_stats=stats_forparfile)
        print(f"Appended best-fit statistical properties to {args.newparfile} par file\n")


if __name__ == '__main__':
    main()
