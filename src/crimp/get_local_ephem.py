"""
Module to generate local [F0, F1] ephemerides in a moving average fashion,
a typical analysis strategy for pulsars

Can be run via command line as "localephemerides"

Logger is very thing on information - could be made more useful
"""
import pandas as pd
import numpy as np
import argparse

from crimp.readtimingmodel import ReadTimingModel
from crimp.ephemIntegerRotation import ephemIntegerRotation

from crimp.fit_toas import (load_toas_for_fit, model_phases, inject_free_params, plot_residulas,
                            Prior, run_mcmc, list_fit_keys, chi2_fit)
from crimp.plot_local_ephem import plot_local_ephemerides
from crimp.timfile import readtimfile

# Log config
############
from crimp.logging_utils import get_logger, configure_logging
logger = get_logger(__name__)


def generate_local_ephemerides(
        tim_file, parfile,
        interval_days=90,
        jump_days=15,
        t_start=None,
        t_end=None,
        min_interval=45,
        debug_with_plots=False,
        outputfile='local_ephemerides',
        ephem_plot=None,
        clobber=False
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
    :param clobber: override existing .txt file of local F0, F1? Default: False
    :type clobber: bool
    :return: local_ephem_final_df: dataframe of F0, F1, and MJD of each time interval with uncertainties
    :rtype: pd.DataFrame
    """

    logger.info('\n Running generate_local_ephemerides with input parameters: '
                '\n tim_file: ' + tim_file +
                '\n parfile: ' + parfile +
                '\n interval_days: ' + str(interval_days) +
                '\n jump_days: ' + str(jump_days) +
                '\n t_start: ' + str(t_start) +
                '\n t_end: ' + str(t_end) +
                '\n min_interval: ' + str(min_interval) +
                '\n debug_with_plots: ' + str(debug_with_plots) +
                '\n outputfile: ' + str(outputfile) +
                '\n ephem_plot: ' + str(ephem_plot) +
                '\n clobber: ' + str(clobber) + '\n')

    # Reading essential timing parameters
    parfile_params, _, _ = ReadTimingModel(parfile).readfulltimingmodel()
    PEPOCH_global = parfile_params["PEPOCH"]
    F0_global = parfile_params["F0"]
    F1_global = parfile_params["F1"]
    glitch_substring = "GLEP_"
    glitch_epochs = sorted([value for key, value in parfile_params.items() if glitch_substring in key])

    # Reading TOAs from .tim file
    toa_df = readtimfile(tim_file)

    # Start and stop times
    if t_start is None:
        t_start = np.min(toa_df['pulse_ToA'])
    if t_end is None:
        t_end = np.max(toa_df['pulse_ToA'])

    local_ephem_final = []
    current_start = t_start
    eps = 1e-5  # small offset to move past glitch
    int_counter = 0  # interval counter for bookkeeping

    while current_start < t_end:
        # current end according to user defined criteria
        current_end = min(current_start + interval_days, t_end)

        # Select ToAs between current_start and current_end
        mask = (toa_df['pulse_ToA'] >= current_start) & (toa_df['pulse_ToA'] <= current_end)
        toa_df_interval = toa_df.loc[mask]

        if toa_df_interval.empty:
            # skip interval if no data but current_start < t_end is still true (e.g., long gap)
            current_start += jump_days
            continue

        # redefine current_start and current_end to match the exact days of TOA
        current_start = toa_df_interval['pulse_ToA'].min()
        current_end = toa_df_interval['pulse_ToA'].max()

        # Check if this interval crosses a glitch
        crossing_glitch = None
        for g in glitch_epochs:
            if (g > current_start) and (g < current_end):
                crossing_glitch = g
                break

        # If we crossed a glitch indeed - refilter and redefine everything
        if crossing_glitch:
            # Truncate interval before glitch
            current_end = crossing_glitch
            # re-filter data according to glitch_epoch
            mask = toa_df_interval['pulse_ToA'] <= current_end
            toa_df_interval = toa_df_interval.loc[mask]
            # Redefine end more precisely
            current_end = toa_df_interval['pulse_ToA'].max()

        # Exact time of mid-interval
        interval_mid = current_start + (current_end - current_start) / 2.0
        # Number of TOAs in this interval
        n_toa_interval = len(toa_df_interval)
        # Length of time interval
        span_days = (current_end - current_start) if n_toa_interval > 0 else 0.0
        # if length of time interval > min_interval and number of TOA in interval is >= 4 then continue
        if (n_toa_interval >= 4) and (span_days > min_interval):
            # ---- Fit the corresponding subset of ToAs - from fit_toas.py ----
            # Get F0 at mid-interval (simply to be as close as possible to correct solution for ease of fitting)
            ephem_tmp = ephemIntegerRotation(interval_mid, parfile)
            F0_mid = ephem_tmp['freq_intRotation']
            F1_mid = ephem_tmp['freqdot_intRotation']
            interval_mid = ephem_tmp['Tmjd_intRotation']

            # Build a dictionary par file that fit_toas will understand
            # Pretty simple {key_name: {value, flag}}
            # Only caveat is that F0 to F12 must be provided - fill in F0, F1, and then padd 0 afterward
            all_keys = ["PEPOCH"] + [f"F{i}" for i in range(0, 13)]
            # Define the non-zero values for the first three keys
            base_values = [interval_mid, F0_mid, F1_mid]  # F0_mid
            base_flags = [0, 1, 1]
            # For the remaining keys (F2–F12), fill with zeros and flag=0
            n_extra = len(all_keys) - len(base_values)
            p0 = base_values + [0.0] * n_extra
            p0_flag = base_flags + [0] * n_extra
            # Build the dictionary
            init_par_file_dict = {
                k: {"value": np.float64(v), "flag": f}
                for k, v, f in zip(all_keys, p0, p0_flag)
            }

            # keys to fit for, list of ['F0', 'F1']
            fit_keys = list_fit_keys(init_par_file_dict)
            # Build the priors
            bounds = {
                'F0': (-10 / (span_days * 86400), 10 / (span_days * 86400)),
                'F1': (-100 / (span_days * 86400) ** 2, 100 / (span_days * 86400) ** 2),
            }
            # Instantiate the Prior object - from fit_toas.py
            priors = Prior(bounds=bounds, initial_guess={})

            if debug_with_plots is True:
                corner_plot = f"corner_interval_{int_counter}.pdf"
            else:
                corner_plot = None

            # Create phases according to the above initial timing model
            toas_to_fit = load_toas_for_fit(toa_df_interval, init_par_file_dict)
            _, _, summaries = run_mcmc(x=toas_to_fit['ToA'], y=toas_to_fit['phase'],
                                       yerr=toas_to_fit["phase_err_cycle"], init_parfile=init_par_file_dict,
                                       keys=fit_keys, prior=priors, steps=1000, burn=100, walkers=24,
                                       progress=False, corner_pdf=corner_plot)

            # Recover timing dict from best mcmc fit
            med_vec = np.array([summaries[name]['median'] for name in fit_keys], dtype=float)
            post_mcmc_timdict_fit, post_mcmc_timdict = inject_free_params(init_par_file_dict, med_vec, fit_keys)
            phase_residulas_post_fit = model_phases(toas_to_fit['ToA'], post_mcmc_timdict_fit)

            if debug_with_plots is True:
                # plot residuals after MCMC run
                plot_residulas(toas_to_fit, phase_residulas_post_fit, plotname=f"residuals_interval_{int_counter}")
                int_counter += 1

            # Extract best-fit F0, F1 and their uncertainties
            # For error, convert to symmetric, take max of +/- q34 around q50
            F0_fit = post_mcmc_timdict['F0']
            F0_fit_err = max(summaries['F0']['plus'], summaries['F0']['minus'])
            F1_fit = post_mcmc_timdict['F1']
            F1_fit_err = max(summaries['F1']['plus'], summaries['F1']['minus'])

            # Get statistics of best-fit for interval
            nbr_free_param = 2
            simple_stats = chi2_fit(toas_to_fit['phase'], phase_residulas_post_fit,
                                    toas_to_fit["phase_err_cycle"], nbr_free_param)

            local_ephem_final.append({
                'TOA_MJD_ref': interval_mid,
                'TOA_MJD_ref_err': span_days / 2.0,
                'F0': F0_fit,
                'F0_err': F0_fit_err,
                'F1': F1_fit,
                'F1_err': F1_fit_err,
                'CHI2R': simple_stats['redchi2'],
                'DOF': simple_stats['dof']
            })

        # If we crossed a glitch, jump just after it
        if crossing_glitch:
            current_start = crossing_glitch + eps
            continue

        # Otherwise move forward by jump_days
        current_start += jump_days

    # Normalize F0 (subtract the global linear trends)
    if not local_ephem_final:
        logger.warning('No interval made the criteria - try decreasing min_interval and/or increasing interval_days '
                       'returning empty dataframe')
        return pd.DataFrame(local_ephem_final)

    local_ephem_final_df = pd.DataFrame(local_ephem_final)
    F0_local_based_on_parfile = F0_global + F1_global * ((local_ephem_final_df['TOA_MJD_ref'] - PEPOCH_global) * 86400)
    local_ephem_final_df['F0'] -= F0_local_based_on_parfile

    # Create CSV of local ephemerides
    if outputfile is not None and clobber is False:
        local_ephem_final_df.to_csv(f"{outputfile}.txt", sep='\t', index=True, header=True, mode='x')
    elif outputfile is not None and clobber is True:
        local_ephem_final_df.to_csv(f"{outputfile}.txt", sep='\t', index=True, header=True)
    else:
        print(f"No output text file created for local ephemerides.")

    # Creating a plot of the local ephemerides
    if ephem_plot is not None:
        plot_local_ephemerides(local_ephem_final_df, glitch_epochs, ephem_plot)
    else:
        print(f"No local ephemerides plot create.")

    return local_ephem_final_df


def main():
    parser = argparse.ArgumentParser(
        description="Module to generate local [F0, F1] ephemerides in a moving average fashion")
    parser.add_argument("timfile", help=".tim TOA file",
                        type=str)
    parser.add_argument("parfile", help="A tempo2 .par file",
                        type=str)
    parser.add_argument("-id", "--interval_days", help="Length of separate time interval (days)",
                        type=float, default=90.)
    parser.add_argument("-jd", "--jump_days", help="Shift time interval by (days)",
                        type=float, default=15.)
    parser.add_argument("-ts", "--t_start", help="Start from (MJD)",
                        type=float, default=None)
    parser.add_argument("-te", "--t_end", help="Stop at (MJD)",
                        type=float, default=None)
    parser.add_argument("-mi", "--min_interval",
                        help="Minimum time between first and last ToA in each interval (MJD)",
                        type=float, default=45)
    parser.add_argument("-dp", "--debug_with_plots",
                        help="For debugging purposes - plot ToAs in each interval, and corner plot of F1, F0",
                        default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-of", "--outputfile",
                        help="Name of output .txt file of local ephemerides (default='local_ephemerides'.txt)",
                        type=str, default="local_ephemerides")
    parser.add_argument("-ep", "--ephem_plot", help="Plot of local ephemerides (default=None)",
                        type=str, default=None)
    parser.add_argument("-cl", "--clobber",
                        help="Override output .txt file of local ephemerides (default=False)",
                        default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="WARNING if absent, -v: INFO, -vv: DEBUG")
    args = parser.parse_args()

    # Configure the log-file
    v = min(args.verbose, 2)  # cap -vv
    console_level = ("WARNING", "INFO", "DEBUG")[v]  # WARNING if --verbose is absent, INFO if -v, DEBUG if -vv

    log_file = f"{args.outputfile}.log" if args.outputfile is not None else "local_ephemerides.log"
    configure_logging(console_level=console_level, file_path=log_file, file_level="INFO", force=True)

    cli_logger = get_logger(__name__)
    cli_logger.info("\nCLI starting")

    generate_local_ephemerides(args.timfile, args.parfile, args.interval_days, args.jump_days,
                               args.t_start, args.t_end, args.min_interval, args.debug_with_plots,
                               args.outputfile, args.ephem_plot, args.clobber)


if __name__ == '__main__':
    main()
