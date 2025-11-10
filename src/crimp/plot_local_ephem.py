"""
Simple script to plot local ephemerides
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse


def read_local_ephemerides(localephem: str, t_start: float | None = None, t_end: float | None = None):
    """
    Read local ephemerides .txt file (build with get_local_ephem.py or CLT "localephemerides")
    :param localephem: .txt file of local ephemerides
    :type localephem: str
    :param t_start: start time of interest
    :type t_start: float | None
    :param t_end: end time of interest
    :type t_end: float | None
    :return: dataframe of local ephemerides
    :rtype: pandas.DataFrame
    """
    localephem_df_tmp = pd.read_csv(localephem, sep=r'\s+', comment='#', header=0)

    # filter between t_start and t_end
    if t_start is None:
        t_start = localephem_df_tmp["TOA_MJD_ref"].min()
    if t_end is None:
        t_end = localephem_df_tmp["TOA_MJD_ref"].max()

    mask = ((localephem_df_tmp["TOA_MJD_ref"] >= t_start) & (localephem_df_tmp["TOA_MJD_ref"] <= t_end))
    localephem_df = localephem_df_tmp.loc[mask].reset_index(drop=True)

    return localephem_df


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

    markers, caps, bars = axs[0].errorbar(local_df['TOA_MJD_ref'], local_df['F0'],
                                          xerr=local_df['TOA_MJD_ref_err'], yerr=local_df['F0_err'],
                                          fmt='o', color='k', ecolor='gray', elinewidth=1.5,
                                          capsize=2, markersize=6, label=r'$F$', alpha=0.7)

    # Set separate alpha value for bar and cap
    [bar.set_alpha(0.35) for bar in bars]
    [cap.set_alpha(0.35) for cap in caps]

    axs[0].grid(True, linestyle='--', alpha=0.3)

    ##################
    # Formatting axs[1] (F1)
    axs[1].tick_params(axis='both', labelsize=14)
    axs[1].xaxis.offsetText.set_fontsize(14)
    axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axs[1].yaxis.offsetText.set_fontsize(14)
    axs[1].set_xlabel(r'$\,\mathrm{Time\ (MJD)}$', fontsize=14)
    axs[1].set_ylabel(r'$\,\mathrm{\dot{F}\ (Hz\,s^{-1})}$', fontsize=14)

    markers, caps, bars = axs[1].errorbar(local_df['TOA_MJD_ref'], local_df['F1'],
                                          xerr=local_df['TOA_MJD_ref_err'], yerr=local_df['F1_err'],
                                          fmt='o', color='k', ecolor='gray', elinewidth=1.5,
                                          capsize=2, markersize=6, label=r'$\dot{F}$', alpha=0.7)

    # Set separate alpha value for bar and cap
    [bar.set_alpha(0.35) for bar in bars]
    [cap.set_alpha(0.35) for cap in caps]

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
    parser = argparse.ArgumentParser(description="Plot local ephemerides")
    parser.add_argument("localephem",
                        help=".txt file of local ephemerides built with get_local_ephem.py or CLT localephemerides",
                        type=str)
    parser.add_argument("-ts", "--t_start", help="Start from (MJD)",
                        type=float, default=None)
    parser.add_argument("-te", "--t_end", help="Stop at (MJD)",
                        type=float, default=None)
    parser.add_argument("-gl", "--glitches", help="MJDs of glitches (where dashed lines will be plotted)",
                        type=float, nargs='+', default=None,)
    parser.add_argument("-ep", "--ephem_plot", help="Plot of local ephemerides (default=None)",
                        type=str, default=None)
    args = parser.parse_args()

    # Reading .txt local ephemerides file
    localephem_df = read_local_ephemerides(args.localephem, args.t_start, args.t_end)

    # Creating the plot
    plot_local_ephemerides(localephem_df, glitches=args.glitches, plotname=args.ephem_plot)


if __name__ == '__main__':
    main()
