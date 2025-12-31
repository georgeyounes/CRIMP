"""
Simple module that performs a fit to the pulse ToAs
Models that are allowed currently are taylor expansion (F0, F1, F2, ..., F12), glitches, and WAVE
Similar to Tempo2 and PINT, a flag following each of those parameters in a .par file will indicate whether they are
fixed (0 or missing flag) or free to vary (1)

Can be called via command line as "fittoas"

Note on the fitting: we optimize for the error in the timing model parameters and not the model parameters themselves,
hence, we fit the phase-residuals
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import argparse

import emcee
import corner

from crimp.readtimingmodel import (patch_par_values, patch_statistics, patch_miscellaneous, get_parameter_value,
                                   ReadTimingModel)
from crimp.timfile import readtimfile, PulseToAs
from crimp.utilities_fittoas import *
from crimp.calcphase import calcphase

# Log config
############
from crimp.logging_utils import get_logger

logger = get_logger(__name__)


# -------- Starting point --------
def load_toas_for_fit(tim_file_df: pd.DataFrame, parfile: dict, t_start: float | None = None,
                      t_stop: float | None = None,
                      t_mjd_phasewrap: float | None = None, mode: str = "add") -> pd.DataFrame:
    """
    Prepare the TOAs from a .tim file
    :param tim_file_df: .tim TOA dataframe (e.g., a .tim file read with timefile.py)
    :type tim_file_df: pd.DataFrame
    :param parfile: dictionary of timing solution (e.g., from ReadTimingModel)
    :type parfile: dict
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
    # "get_parameter_value" reads in the value whether the parfile dict has:
    # - only value 'F0': 9.8765 (i.e., first return dictionary in ReadTimingModel)
    # - or a dict of {'value': 9.8765, 'flag': 1} keys (i.e., third return dictionary in ReadTimingModel)
    #  In both cases, F0 = 9.8765
    F0 = get_parameter_value(parfile['F0'])

    # Store .tim pulse TOA dataframe as an object of PulseToAs
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
    # Check if TRACK exists in par input, is = -2, and if pn is part of the .tim file
    if "TRACK" in parfile and get_parameter_value(parfile["TRACK"]) == -2 and "pn" in pt.df.columns:
        phases -= pt.df["pn"]
        logger.info(f"Found TRACK -2 in {parfile} and -pn (pulse number) flag in .tim file - using those")
    else:
        phases = ((phases + 0.5) % 1.0) - 0.5  # keep things between [-0.5, 0.5)
        logger.info(f"Phase folding between [-0.5, 0.5)")
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


# -------- MCMC through emcee --------
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
        mu = model_phase_residuals(x, init_parfile, theta, keys)
        # gaussian_nll is negative hence the - sign in front (as required by emcee.EnsembleSampler)
        return lp - gaussian_nll(np.asarray(y - np.mean(y)), np.asarray(mu), np.asarray(yerr))

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
        fig.savefig(corner_pdf + '.pdf', format='pdf', dpi=300)
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

    # Plot data - first normalize given phaseshifts to mean - everything we do will be normalized to mean
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
        plt.close(fig)
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
    parser.add_argument("-mc", "--mcmc", help="Run emcee to sample posteriors",
                        default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-st", "--mcmc-steps", type=int, default=10000,
                        help="Number of MCMC steps (default = 10000)")
    parser.add_argument("-bu", "--mcmc-burn", type=int, default=500,
                        help="Burn-in steps to discard when saving flat chain (default=500)")
    parser.add_argument("-wa", "--mcmc-walkers", type=int, default=32,
                        help="Number of walkers (default=32)")
    parser.add_argument("-cp", "--corner_plot", type=str, default=None,
                        help="Path to save Corner plot PDF (default=None)")
    parser.add_argument("-ch", "--chain-npy", type=str, default=None,
                        help="Path to save full chain as .npy (default=None)")
    parser.add_argument("-fl", "--flat-npy", type=str, default=None,
                        help="Path to save flattened post burn-in samples as .npy (default=None)")
    # Plotting residuals
    parser.add_argument("-rp", "--residual_plot", help="Plot of pre- and post-fit residuals",
                        type=str, default=None)

    # Parse all provided arguments
    args = parser.parse_args()

    ###################################
    # reading par file with flag values
    init_parfile_withflags = ReadTimingModel(args.parfile).readfulltimingmodel()[2]
    F0 = get_parameter_value(init_parfile_withflags['F0'])

    ####################
    # reading pulse TOAs
    tim_file_df = readtimfile(args.timfile_path, comment='C')
    toas_pre_fit = load_toas_for_fit(tim_file_df, init_parfile_withflags, args.t_start,
                                     args.t_end, args.t_mjd, args.mode)

    ############################
    # Validate .par timing model
    validate_parfile(init_parfile_withflags)

    START = toas_pre_fit["ToA"].min()
    FINISH = toas_pre_fit["ToA"].max()
    misc_keys = {"START": START, "FINISH": FINISH}

    ########################################################################################
    # MCMC posteriors if desired (currently only way to get uncertainties on fit parameters)
    if args.mcmc:
        keys = list_fit_keys(init_parfile_withflags)
        # check for the prior .yaml file
        if args.mcmc and args.init_yaml is None:
            parser.error("-iy (or --init_yaml) is required when -mc (or --mcmc) is used")
        # Initilize priors
        prior = initguess_prior_from_yaml(args.init_yaml)
        # Run MCMC with emcee
        print("Running MCMC with emcee...")
        sampler, flat, summaries = run_mcmc(x=toas_pre_fit["ToA"], y=toas_pre_fit["phase"],
                                            yerr=toas_pre_fit["phase_err_cycle"], init_parfile=init_parfile_withflags,
                                            keys=keys, prior=prior, steps=args.mcmc_steps, burn=args.mcmc_burn,
                                            walkers=args.mcmc_walkers, corner_pdf=args.corner_plot,
                                            chain_npy=args.chain_npy, flat_npy=args.flat_npy, progress=True)

        print("Posterior summaries(median - / + 1σ approx using 16th / 84th percentiles):")
        uncertainties_max = {}  # define before loop
        for name, s in summaries.items():
            print(f"  {name}: {s['median']:.8e} -{s['minus']:.2e} +{s['plus']:.2e}")
            # convert to 1-sided uncertainty - choose max (conservative approach)
            uncertainties_max[name] = max(s["minus"], s["plus"])

        # prepare medians in keys order
        med_vec = np.array([summaries[name]['median'] for name in keys], dtype=float)
        _, post_mcmc_timdict = inject_free_params(init_parfile_withflags, med_vec, keys)

        source_label = "MCMC (posterior medians)"

        # plot residuals after MCMC run
        phase_residulas_post_fit = model_phase_residuals(toas_pre_fit["ToA"], init_parfile_withflags, med_vec, keys)
        plot_residulas(toas_pre_fit, phase_residulas_post_fit, args.residual_plot)

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
                                               toas_pre_fit["phase_err_cycle"], init_parfile_withflags, args.init_yaml)

        ########################
        # MLE of above objective
        if any("wave" in s.lower() for s in keys):
            if any("glep_" in s.lower() for s in keys):
                logger.warning(f"Fitting for glitch epoch and waves simultaneously is discouraged. Either "
                               f"fix any GELP in your model or make sure the initial guess is close to the "
                               f"true value. Utilizing MCMC is reasonable but may be slow to converge.")
            res = minimize(nll, p0, method='BFGS', options={"maxiter": int(1e5)}, tol=1.0e-16, jac='3-point')
        else:
            res = minimize(nll, p0, method='Nelder-Mead', options={"maxiter": int(1e5)})

        ###############################################
        # Recover best-fit timing dict according to MLE
        timing_dict_fit, timing_dict_full = inject_free_params(tmpl_parfile, res.x, keys)

        # Plot MLE fit residuals - first model residuals according to best-fit model
        phase_residulas_post_fit = model_phase_residuals(toas_pre_fit["ToA"], tmpl_parfile, res.x, keys)
        plot_residulas(toas_pre_fit, phase_residulas_post_fit, args.residual_plot)

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
