"""
Module to generate local [F0, F1] ephemerides in a moving average fashion,
a typical analysis strategy for pulsars

Can be run via command line as "localephemerides"
"""
import pandas as pd
from scipy.optimize import minimize
import numpy as np
import argparse
import matplotlib.pyplot as plt

from crimp.readtimingmodel import ReadTimingModel
from crimp.ephemIntegerRotation import ephemIntegerRotation
from crimp.fit_toas import model_phases, make_nll, bestfit_timing_dict, plot_residulas, Prior, run_mcmc
from crimp.timfile import readtimfile, PulseToAs


def generate_local_ephemerides(
        tim_file, parfile,
        interval_days=90,
        jump_days=15,
        t_start=None,
        t_end=None,
        min_interval=60,
        debug_with_plots=False,
        outputfile='local_ephemerides',
        ephem_plot=None
):
    """
    Generate local F0, F1 every interval_days with a jump of jump_days,
    stopping at glitch epochs and resuming after each glitch.

    :param tim_file: .tim file of TOAs
    :type tim_file: str
    :param parfile: .par file of timing solution (the one used to generate the input .tim file works best)
    :type parfile: str
    :param interval_days: length of time interval (in days) for each F0, F1 calculation
    :type interval_days: float
    :param jump_days: length of time (in days) to shift the interval_days by
    :type jump_days: float
    :param t_start: optional start time (MJD) - otherwise start from first TOA
    :type t_start: float
    :param t_end: optional stop time (MJD) - otherwise stop at last TOA
    :type t_end: float
    :param min_interval: skip if start and stop TOAs within an interval is less than min_interval (in days)
    :type min_interval: float
    :param debug_with_plots: plot ToAs of each interval and posterior corner F0, F1 plot (use for debugging)
    :type min_interval: bool
    :param outputfile: name of output .txt file of local F0, F1, time and their uncertainties
    :type outputfile: str | None
    :param ephem_plot: name of output .pdf plot of local F0, F1 vs time
    :type outputfile: str | None
    :return: local_ephem_final_df: dataframe of F0, F1, and MJD of each time interval with uncertainties
    :rtype: pd.DataFrame
    """

    # Reading essential timing parameters
    parfile_params, _, _ = ReadTimingModel(parfile).readfulltimingmodel()
    PEPOCH_global = parfile_params["PEPOCH"]
    F0_global = parfile_params["F0"]
    F1_global = parfile_params["F1"]
    glitch_substring = "GLEP_"
    glitch_epochs = sorted([value for key, value in parfile_params.items() if glitch_substring in key])

    # Reading TOAs from .tim file
    tim_file_df = readtimfile(tim_file)
    pt = PulseToAs(tim_file_df)
    # Keep things ordered and reindex to 0..N-1
    pt.df = pt.df.sort_values('pulse_ToA').reset_index(drop=True)
    # Build array of TOAs
    toas = pd.to_numeric(pt.df['pulse_ToA'], errors='coerce').to_numpy(dtype=float)
    perr = pd.to_numeric(pt.df['pulse_ToA_err'], errors='coerce').to_numpy(dtype=float)
    phase = np.asarray(model_phases(toas, parfile), dtype=float)

    toa_df = pd.DataFrame({
        'ToA': toas,
        'phase': phase,
        'phase_err_cycle': (perr / 1e6) * F0_global,  # assuming pulse_ToA_err in microseconds
    })

    # Start and stop times
    if t_start is None:
        t_start = np.min(toas)
    if t_end is None:
        t_end = np.max(toas)

    local_ephem_final = []
    current_start = t_start
    eps = 1e-5  # small offset to move past glitch
    int_counter = 0
    while current_start < t_end:
        current_end = min(current_start + interval_days, t_end)
        interval_mid = current_start + (current_end - current_start) / 2.0

        # Check if this interval crosses a glitch
        crossing_glitch = None
        for g in glitch_epochs:
            if (g > current_start) and (g < current_end):
                crossing_glitch = g
                break

        if crossing_glitch:
            # Truncate interval before glitch
            current_end = crossing_glitch
            interval_mid = (current_start + current_end) / 2.0

        # Select ToAs in this interval - if less than 4 ToAs in interval or interval less than min_interval, skip
        mask = (toa_df['ToA'] >= current_start) & (toa_df['ToA'] <= current_end)
        df_interval = toa_df.loc[mask]
        n_toa_interval = len(df_interval)
        span_days = (df_interval['ToA'].max() - df_interval['ToA'].min()) if n_toa_interval > 0 else 0.0
        if (n_toa_interval >= 4) and (span_days > min_interval):
            # ---- Fit the corresponding subset of ToAs - from fit_toas.py ----
            # Get F0 at mid-interval (note that I am trying to maintain the same phases for ease of fitting)
            ephem_tmp = ephemIntegerRotation(interval_mid, parfile)
            F0_mid = ephem_tmp['freq_intRotation']
            interval_mid = ephem_tmp['Tmjd_intRotation']

            # Build a dictionary par file that fit_toas will understand
            # Pretty simple {key_name: {value, flag}}
            # Only caveat is that F0 to F12 must be provided - fill in F0, F1, and then padd 0 afterward
            fit_keys = ["PEPOCH"] + [f"F{i}" for i in range(0, 13)]
            # Define the non-zero values for the first three keys
            base_values = [interval_mid, F0_mid, F1_global]
            base_flags = [0, 1, 1]
            # For the remaining keys (F2–F12), fill with zeros and flag=0
            n_extra = len(fit_keys) - len(base_values)
            p0 = base_values + [0.0] * n_extra
            p0_flag = base_flags + [0] * n_extra
            # Build the dictionary
            init_par_file_dict = {
                k: {"value": np.float64(v), "flag": f}
                for k, v, f in zip(fit_keys, p0, p0_flag)
            }

            # Build the objective - from fit_toas.py
            nll, p0, fit_keys, tmpl = make_nll(df_interval['ToA'], df_interval["phase_err_cycle"], init_par_file_dict)

            # MLE of above objective - from fit_toas.py
            res = minimize(nll, p0, method="Nelder-Mead", options={"maxiter": int(1e6)})
            # Run a simple and localized mcmc to get uncertainties - from fit_toas.py
            bounds = {
                'F0': (F0_mid - 1 / (interval_days * 86400), F0_mid + 1 / (interval_days * 86400)),
                'F1': (F1_global - 1 / (interval_days * 86400) ** 2, F1_global + 1 / (interval_days * 86400) ** 2),
            }
            # Instantiate the Prior object - from fit_toas.py
            priors = Prior(bounds=bounds)

            if debug_with_plots is True:
                # Recover timing dict from best MLE fit
                timdict_bestfit = bestfit_timing_dict(res, fit_keys, tmpl)
                # Calculate phases according to this best-fit model
                phase_residulas_post_fit = model_phases(df_interval["ToA"], timdict_bestfit)
                # Plot pre- and post-fit TOAs
                plot_residulas(df_interval, phase_residulas_post_fit)
                # For corner plot below
                corner_plot = f"corner_{int_counter}.pdf"
                int_counter += 1
            else:
                corner_plot = None
            _, _, summaries = run_mcmc(x=df_interval['ToA'].to_numpy(),
                                       yerr=df_interval["phase_err_cycle"].to_numpy(),
                                       init_parfile=init_par_file_dict,
                                       keys=fit_keys, prior=priors, steps=1000, burn=100,
                                       walkers=24, progress=False, corner_pdf=corner_plot)

            F0_fit = summaries['F0']['median']
            # Convert to symmetric uncertainties - should be okay
            F0_fit_err = 0.5 * (summaries['F0']['plus'] + summaries['F0']['minus'])
            F1_fit = summaries['F1']['median']
            # Convert to symmetric uncertainties - should be okay
            F1_fit_err = 0.5 * (summaries['F1']['plus'] + summaries['F1']['minus'])

            local_ephem_final.append({
                'TOA_MJD_ref': interval_mid,
                'TOA_MJD_ref_err': (current_end - current_start) / 2.0,
                'F0': F0_fit,
                'F0_err': F0_fit_err,
                'F1': F1_fit,
                'F1_err': F1_fit_err,
            })

        # If we crossed a glitch, jump just after it
        if crossing_glitch:
            current_start = crossing_glitch + eps
            continue

        # Otherwise move forward by jump_days
        current_start += jump_days

    # Normalize F0 (subtract the global linear trends)
    local_ephem_final_df = pd.DataFrame(local_ephem_final)
    F0_local_based_on_parfile = F0_global + F1_global * ((local_ephem_final_df['TOA_MJD_ref'] - PEPOCH_global) * 86400)
    local_ephem_final_df['F0'] -= F0_local_based_on_parfile

    # Create CSV of local ephemerides
    if outputfile is not None:
        local_ephem_final_df.to_csv(f"{outputfile}.txt", sep='\t', index=True, header=True, mode='x')

    # Creating a plot of the local ephemerides
    if ephem_plot is not None:
        plot_local_ephemerides(local_ephem_final_df, glitch_epochs, ephem_plot)

    return local_ephem_final_df


def plot_local_ephemerides(local_df, glitches=None, plotname=None):
    """
    Plot local ephemerides (F0 and F1) with uncertainties.
    local_df must contain columns:
    ['TOA_MJD_ref', 'TOA_MJD_ref_err', 'F0', 'F0_err', 'F1', 'F1_err']
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k',
                            sharex=True, sharey=False,
                            gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1]})
    axs = axs.ravel()

    ##################
    # Formatting axs[0] (F0)
    axs[0].tick_params(axis='both', labelsize=14)
    axs[0].xaxis.offsetText.set_fontsize(14)
    axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axs[0].yaxis.offsetText.set_fontsize(14)
    axs[0].set_ylabel(r'$\,\mathrm{Frequency\ (Hz)}$', fontsize=14)

    axs[0].errorbar(local_df['TOA_MJD_ref'], local_df['F0'],
                    xerr=local_df['TOA_MJD_ref_err'], yerr=local_df['F0_err'],
                    fmt='o', color='k', ecolor='gray', elinewidth=1.5,
                    capsize=0, markersize=6, label=r'$F$', alpha=0.7)

    axs[0].grid(True, linestyle='--', alpha=0.3)

    ##################
    # Formatting axs[1] (F1)
    axs[1].tick_params(axis='both', labelsize=14)
    axs[1].xaxis.offsetText.set_fontsize(14)
    axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axs[1].yaxis.offsetText.set_fontsize(14)
    axs[1].set_xlabel(r'$\,\mathrm{Time\ (MJD)}$', fontsize=14)
    axs[1].set_ylabel(r'$\,\mathrm{\dot{F}\ (Hz\,s^{-1})}$', fontsize=14)

    axs[1].errorbar(local_df['TOA_MJD_ref'], local_df['F1'],
                    xerr=local_df['TOA_MJD_ref_err'], yerr=local_df['F1_err'],
                    fmt='o', color='k', ecolor='gray', elinewidth=1.5,
                    capsize=0, markersize=6, label=r'$\dot{F}$', alpha=0.7)

    axs[1].grid(True, linestyle='--', alpha=0.3)

    ##################
    # Add vertical glitch lines if provided
    if glitches is not None and len(glitches) > 0:
        for g in glitches:
            axs[0].axvline(g, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            axs[1].axvline(g, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

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
    parser = argparse.ArgumentParser(description="Module to generate local [F0, F1] ephemerides in a moving average fashion")
    parser.add_argument("timfile", help=".tim TOA file", type=str)
    parser.add_argument("parfile", help="A tempo2 .par file", type=str)
    parser.add_argument("-id", "--interval_days", help="Length of separate time interval (days)", type=float,
                        default=90.)
    parser.add_argument("-jd", "--jump_days", help="Shift time interval by (days)", type=float, default=15.)
    parser.add_argument("-ts", "--t_start", help="Start from (MJD)", type=float, default=None)
    parser.add_argument("-te", "--t_end", help="Stop at (MJD)", type=float, default=None)
    parser.add_argument("-mi", "--min_interval",
                        help="Minimum time between first and last ToA in each interval (MJD)", type=float, default=60)
    parser.add_argument("-dp", "--debug_with_plots",
                        help="For debugging purposes - plot ToAs in each interval, and corner plot of F1, F0",
                        default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-of", "--outputfile", help="Name of output .txt file of local ephemerides",
                        type=str, default="local_ephemerides")
    parser.add_argument("-ep", "--ephem_plot", help="Plot of local ephemerides", type=str, default=None)
    args = parser.parse_args()

    generate_local_ephemerides(args.timfile, args.parfile, args.interval_days, args.jump_days,
                               args.t_start, args.t_end, args.min_interval, args.debug_with_plots,
                               args.outputfile, args.ephem_plot)


if __name__ == '__main__':
    main()
