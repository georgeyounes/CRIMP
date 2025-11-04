"""
Simple module to create some pulse profile plots, e.g., 2d count vs phase, 3d phase-energy diagram,
waterfall plot, multi row subplots of energy vs phase (at different times if desired)

Can be run through the command line as "pulseprofile_plots"
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.ndimage import gaussian_filter
import yaml

from crimp.eventfile import EvtFileOps
from crimp.calcphase import calcphase
from crimp.binphases import binphases


def prep_for_plotting(eventfile: str, parfile: str, enelow=0, enehigh=100, t_start=None, t_end=None):
    """
    - reading event data via EvtFileOps
    - filtering by energy/time
    - computing phase via calcphase(parfile)
    - binning
    """
    EF = EvtFileOps(eventfile)
    time_energy_phase_df = EF.build_time_energy_df().filtenergy(eneLow=enelow,
                                                                eneHigh=enehigh).filttime(t_start, t_end).time_energy_df

    # Exposure from GTIs (seconds)
    # GTIs returned as (header, array[[start, stop],...]) where units are MJD
    gti = EF.readGTI()[1]
    gti = update_gti(gti, t_start, t_end)

    # Fold data using timing model
    TIMEMJD = time_energy_phase_df['TIME'].to_numpy()
    _, cycleFoldedPhases = calcphase(TIMEMJD, parfile)

    time_energy_phase_df['foldedphases'] = cycleFoldedPhases

    return time_energy_phase_df, gti


def update_gti(GTI, tstart, tend):
    # Restructuring GTI if tstart and tend are provided
    if tstart is None and tend is None:
        return GTI

    elif tstart is not None and tend is None:
        includegti_idx = (GTI[:, 1] > tstart)
        GTI = GTI[includegti_idx]
        if tstart > GTI[0, 0]:
            GTI[0, 0] = tstart
        return GTI

    elif tstart is None and tend is not None:
        includegti_idx = (GTI[:, 0] < tend)
        GTI = GTI[includegti_idx]
        if tend < GTI[-1, -1]:
            GTI[-1, -1] = tend
        return GTI

    elif tstart is not None and tend is not None:
        # Fixing tstart
        includegti_idx = (GTI[:, 1] > tstart)
        GTI = GTI[includegti_idx]
        if tstart > GTI[0, 0]:
            GTI[0, 0] = tstart
        # Fixing tend
        includegti_idx = (GTI[:, 0] < tend)
        GTI = GTI[includegti_idx]
        if tend < GTI[-1, -1]:
            GTI[-1, -1] = tend
        return GTI


def plotting_pp(time_energy_phase_df, nbrbins=100, plotname: str | None = None):
    """
    Plot the pulse profile
    """
    # Create binned profile
    binnedProfile = binphases(time_energy_phase_df['foldedphases'], nbrbins)
    binnedProfile["ctRate"] = binnedProfile["ctsBins"]
    binnedProfile["ctRateErr"] = binnedProfile["ctsBinsErr"]

    ppBins = binnedProfile["ppBins"]
    ctRate = binnedProfile["ctRate"] / binnedProfile["ctRate"].mean()
    ctRateErr = binnedProfile["ctRateErr"] / binnedProfile["ctRate"].mean()

    # Initiating plot
    fig, ax1 = plt.subplots(1, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')

    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlabel(r'$\,\mathrm{Phase\,(cycles)}$', fontsize=12)
    ax1.set_ylabel(r'$\,\mathrm{Normalized\,rate}$', fontsize=12)
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))
    ax1.xaxis.offsetText.set_fontsize(12)
    ax1.yaxis.offsetText.set_fontsize(12)

    if np.max(ppBins) > 1:  # Plotting a second cycle for cauchy or vonmises (cycle = 2*pi)
        secondCycle = 2 * np.pi
    else:  # Plotting a second cycle for fourier (cycle = 1)
        secondCycle = 1

    # Creating two cycles and plotting PP
    ppBins_plt = np.append(ppBins, ppBins + secondCycle)
    ctRate_plt = np.append(ctRate, ctRate)
    ctRateErr_plt = np.append(ctRateErr, ctRateErr)

    ax1.errorbar(ppBins_plt, ctRate_plt, yerr=ctRateErr_plt, fmt='ok', zorder=10)
    ax1.step(ppBins_plt, ctRate_plt, 'k+-', where='mid', zorder=10)

    ax1.set_xlim(0.0, 2 * secondCycle)

    #############################
    # Finishing touches
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.5)
        ax1.tick_params(width=1.5)

    fig.tight_layout()

    if plotname is None:
        plt.show()
    else:
        fig.savefig(str(plotname) + '.pdf', format='pdf', dpi=300, bbox_inches="tight")
    return


def plotting_phase_energy(time_energy_phase_df, nphasebins=64, nenergybins=24,
                          smooth_sigma: float | list | None = 0.5,
                          plotname: str | None = None):
    """
    Plot a phase-energy diagram:
    x-axis = phase (cycles), y-axis = energy, colormap = min-max normalized count rate
    Optionally apply a mild Gaussian smoothing to reduce noise
    """
    # Extract needed arrays
    phases = time_energy_phase_df['foldedphases'].to_numpy()
    energies = time_energy_phase_df['PI'].to_numpy()

    # Bin edges
    phase_edges = np.linspace(0.0, 1, nphasebins + 1)
    e_min, e_max = np.nanmin(energies), np.nanmax(energies)
    energy_edges = np.logspace(np.log10(e_min), np.log10(e_max), nenergybins + 1)

    # 2D histogram: counts per (phase, energy) bin
    H, xedges, yedges = np.histogram2d(phases, energies, bins=[phase_edges, energy_edges])

    # Transpose so rows correspond to energy bins
    H_T = H.T

    # Min-Max normalization of each energy row
    row_min = H_T.min(axis=1, keepdims=True)
    row_max = H_T.max(axis=1, keepdims=True)
    rate = (H_T - row_min) / (row_max - row_min)

    # Optional Gaussian smoothing
    if smooth_sigma is not None:
        # Apply smoothing in both phase (x) and energy (y) directions
        # smooth_sigma can be a float or a list/tuple [sigma_energy, sigma_phase]
        if isinstance(smooth_sigma, list):
            smooth_sigma = tuple(smooth_sigma)
        rate = gaussian_filter(rate, sigma=smooth_sigma, mode='nearest')

    # Plot
    fig, ax = plt.subplots(1, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')

    pcm = ax.pcolormesh(xedges, yedges, rate, shading='auto')

    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel(r'$\,\mathrm{Phase\,(cycles)}$', fontsize=12)
    ax.set_ylabel(r'$\,\mathrm{Energy}$', fontsize=12)
    ax.xaxis.offsetText.set_fontsize(12)
    ax.yaxis.offsetText.set_fontsize(12)
    ax.set_yscale("log")

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.tick_params(width=1.5)

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(r'$\,\mathrm{Min-Max\,scaling}$', fontsize=12)

    fig.tight_layout()

    if plotname is None:
        plt.show()
    else:
        fig.savefig(str(plotname) + '.pdf', format='pdf', dpi=300, bbox_inches="tight")
    return


def plotting_phase_time(time_energy_phase_df, nphasebins=32, ntimebins=12,
                        smooth_sigma: float | list | None = 0.5,
                        plotname: str | None = None):
    """
    Plot a phase-time diagram:
    x-axis = phase (cycles), y-axis = time (MJD), colormap = min-max normalized count rate
    Optionally apply a Gaussian smoothing to reduce noise
    """
    # Extract needed arrays
    phases = time_energy_phase_df['foldedphases'].to_numpy()
    times = time_energy_phase_df['TIME'].to_numpy()

    # Bin edges
    phase_edges = np.linspace(0.0, 1, nphasebins + 1)
    t_min, t_max = np.nanmin(times), np.nanmax(times)
    time_edges = np.linspace(t_min, t_max, ntimebins + 1)

    # 2D histogram - and transpose so that rows correspond to energy bins
    H, xedges, yedges = np.histogram2d(phases, times, bins=[phase_edges, time_edges])
    H_T = H.T

    # Min–Max normalization per row (with NaN for flat rows)
    row_min = H_T.min(axis=1, keepdims=True)
    row_max = H_T.max(axis=1, keepdims=True)
    denom = row_max - row_min
    # Initialize output as NaN
    rate = np.full_like(H_T, np.nan, dtype=float)
    # Compute normalized values where denom != 0 - time slice which does not have any data
    np.divide(H_T - row_min, denom, out=rate, where=denom != 0)

    # Optional Gaussian smoothing
    if smooth_sigma is not None:
        # Apply smoothing in both phase (x) and energy (y) directions
        # smooth_sigma can be a float or a list/tuple [sigma_energy, sigma_phase]
        if isinstance(smooth_sigma, list):
            smooth_sigma = tuple(smooth_sigma)
        rate = gaussian_filter(rate, sigma=smooth_sigma, mode='nearest')

    # Plot
    fig, ax = plt.subplots(1, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')

    pcm = ax.pcolormesh(xedges, yedges, rate, shading='auto')

    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel(r'$\,\mathrm{Phase\,(cycles)}$', fontsize=12)
    ax.set_ylabel(r'$\,\mathrm{Time\,(MJD)}$', fontsize=12)
    ax.xaxis.offsetText.set_fontsize(12)
    ax.yaxis.offsetText.set_fontsize(12)
    ax.set_yscale("log")

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
        ax.tick_params(width=1.5)

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(r'$\,\mathrm{Min-Max\,\,Scaling}$', fontsize=12)

    fig.tight_layout()

    if plotname is None:
        plt.show()
    else:
        fig.savefig(str(plotname) + '.pdf', format='pdf', dpi=300, bbox_inches="tight")
    return


def plotting_pp_grid(time_energy_phase_df,
                     n_timebins=6,
                     n_energybins=6,
                     nbrbins=(20, 24, 24, 24, 20, 16),  # can be int or list/array length n_energybins
                     plotname: str | None = None):
    """
    Subplots of pulse profiles:
    rows = time bins, columns = energy bins.
    - Two phase cycles, includes uncertainties (like plotting_pp)
    - Top row titles show energy ranges: "Emin - Emax keV"
    - Right-side y-axis (last column) shows time ranges: "Tmin - Tmax MJD" (integers)
    - Per-energy-channel binning if nbrbins is iterable
    - Tight spacing between subplots
    """
    # Extract arrays
    phases = time_energy_phase_df['foldedphases'].to_numpy()
    times = time_energy_phase_df['TIME'].to_numpy()
    energies = time_energy_phase_df['PI'].to_numpy()

    # Time bin edges
    t_min, t_max = np.nanmin(times), np.nanmax(times)
    time_edges = np.linspace(t_min, t_max, n_timebins + 1)

    # Energy bin edges (log)
    e_min, e_max = np.nanmin(energies), np.nanmax(energies)
    e_min = max(e_min, np.nextafter(0, 1))  # guard for logspace
    energy_edges = np.logspace(np.log10(e_min), np.log10(e_max), n_energybins + 1)

    # Per-energy-channel binning
    if np.isscalar(nbrbins):
        nbrbins_list = [int(nbrbins)] * n_energybins
    else:
        nbrbins_list = list(nbrbins)
        if len(nbrbins_list) != n_energybins:
            raise ValueError("If nbrbins is iterable, its length must equal n_energybins.")

    # Figure & axes
    fig, axes = plt.subplots(n_timebins, n_energybins,
                             figsize=(3.8 * n_energybins, 2.9 * n_timebins),
                             dpi=80, facecolor='w', edgecolor='k', squeeze=False)

    # First pass: compute panel data and global y-lims
    ymin_all, ymax_all = np.inf, -np.inf
    panels = []  # (i, j, ppBins_plt, ct_norm_plt, ct_err_norm_plt)

    for i in range(n_timebins):
        t0, t1 = time_edges[i], time_edges[i + 1]
        for j in range(n_energybins):
            e0, e1 = energy_edges[j], energy_edges[j + 1]
            ax = axes[i, j]

            sel = (times >= t0) & (times < t1) & (energies >= e0) & (energies < e1)
            if not np.any(sel):
                ax.set_visible(False)
                panels.append((i, j, None, None, None))
                continue

            # Bin phases for this tile (per-energy bin count)
            nb = int(nbrbins_list[j])
            binned = binphases(phases[sel], nb)

            ppBins = binned["ppBins"]
            ct = binned["ctsBins"].astype(float)
            ctErr = binned["ctsBinsErr"].astype(float) if "ctsBinsErr" in binned else np.sqrt(np.maximum(ct, 1.0))

            mean_ct = ct.mean()
            if mean_ct <= 0:
                ax.set_visible(False)
                panels.append((i, j, None, None, None))
                continue

            # Normalize by mean
            ct_norm = ct / mean_ct
            ct_err_norm = ctErr / mean_ct

            # Two cycles (match plotting_pp)
            secondCycle = 2 * np.pi if np.max(ppBins) > 1 else 1.0
            ppBins_plt = np.append(ppBins, ppBins + secondCycle)
            ct_norm_plt = np.append(ct_norm, ct_norm)
            ct_err_norm_plt = np.append(ct_err_norm, ct_err_norm)

            panels.append((i, j, ppBins_plt, ct_norm_plt, ct_err_norm_plt))

            ymin_all = min(ymin_all, np.nanmin(ct_norm))
            ymax_all = max(ymax_all, np.nanmax(ct_norm))

    # Uniform y-lims
    if not np.isfinite(ymin_all) or not np.isfinite(ymax_all):
        ymin_all, ymax_all = 0.85, 1.15
    else:
        pad = 0.05 * (ymax_all - ymin_all if ymax_all > ymin_all else 0.3)
        ymin_all = max(0.0, ymin_all - pad)
        ymax_all = ymax_all + pad

    # Helper for neat number formatting
    def fmt_e(x):
        # compact but readable; adjust if you prefer fixed decimals
        return f"{x:.2g}"

    # Second pass: draw + labels
    for (i, j, ppBins_plt, ct_norm_plt, ct_err_norm_plt) in panels:
        ax = axes[i, j]
        if ppBins_plt is None:
            # Still place column energy headers (top row) even if no data
            if i == 0:
                e0, e1 = energy_edges[j], energy_edges[j + 1]
                ax.set_title(f"{fmt_e(e0)} - {fmt_e(e1)} keV", fontsize=14, pad=4)
            if j == n_energybins - 1:
                t0, t1 = time_edges[i], time_edges[i + 1]
                axR = ax.twinx()
                axR.set_ylabel(f"{int(t0)} - {int(t1)} MJD", fontsize=14, rotation=270, labelpad=14)
                axR.set_yticks([])
                axR.tick_params(right=True, labelright=True, length=0)
            continue

        # Uncertainties + step (same as plotting_pp)
        ax.errorbar(ppBins_plt, ct_norm_plt, yerr=ct_err_norm_plt, fmt='ok', zorder=10)
        ax.step(ppBins_plt, ct_norm_plt, 'k+-', where='mid', zorder=10)

        # Limits & ticks
        ax.set_xlim(0.0, np.max(ppBins_plt))  # two cycles span
        ax.set_ylim(ymin_all, ymax_all)
        ax.tick_params(axis='both', labelsize=14, length=3)

        # Outer labels only
        if i == n_timebins - 1:
            ax.set_xlabel(r'Phase (cycles)', fontsize=14)
        else:
            ax.set_xticklabels([])
        if j == 0:
            ax.set_ylabel(r'Norm. rate', fontsize=14)
        else:
            ax.set_yticklabels([])

        # Column energy header on TOP ROW
        if i == 0:
            e0, e1 = energy_edges[j], energy_edges[j + 1]
            ax.set_title(f"{fmt_e(e0)} - {fmt_e(e1)} keV", fontsize=14, pad=4)

        # Row time label on RIGHT of LAST COLUMN
        if j == n_energybins - 1:
            t0, t1 = time_edges[i], time_edges[i + 1]
            axR = ax.twinx()
            axR.set_ylabel(f"{int(t0)} - {int(t1)} MJD", fontsize=14, rotation=270, labelpad=14)
            axR.set_yticks([])
            axR.tick_params(right=True, labelright=True, length=0)

        # Spines
        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_linewidth(1.2)

    # Tight spacing
    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    if plotname is None:
        plt.show()
    else:
        fig.savefig(str(plotname) + '.pdf', format='pdf', dpi=300, bbox_inches="tight")
    return


def plotting_pp_before_after(time_energy_phase_df,
                             t_mjd: float,
                             days_window: float | list | tuple = 7,
                             nbrbins: int = 48,
                             plotname: str | None = None):
    """
    Two stacked pulse profiles around a reference time t_mjd:
      - Top panel: [t_mjd - days_window, t_mjd]
      - Bottom panel: [t_mjd, t_mjd + days_window]
    Energy selection is assumed to be pre-filtered in time_energy_phase_df.
    Both panels use the same nbrbins, two phase cycles, and include uncertainties.
    """
    phases = time_energy_phase_df['foldedphases'].to_numpy()
    times = time_energy_phase_df['TIME'].to_numpy()

    # Parse days_window
    if isinstance(days_window, (list, tuple)):
        if len(days_window) != 2:
            raise ValueError("days_window must be a scalar or a pair (pre_half, post_half).")
        pre_half, post_half = float(days_window[0]), float(days_window[1])
    else:
        pre_half = post_half = float(days_window)

    # Define windows
    pre_t0, pre_t1 = t_mjd - pre_half, t_mjd
    post_t0, post_t1 = t_mjd, t_mjd + post_half
    windows = [(pre_t0, pre_t1), (post_t0, post_t1)]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k', squeeze=False)

    # First pass: compute binned profiles to get common y-lims
    ymin_all, ymax_all = np.inf, -np.inf
    panels = []  # (row, x_plot, y_plot, yerr_plot, (t0,t1))

    for row, (t0, t1) in enumerate(windows):
        sel = (times >= t0) & (times <= t1)
        ax = axes[row, 0]
        if not np.any(sel):
            ax.set_visible(False)
            panels.append((row, None, None, None, (t0, t1)))
            continue

        binned = binphases(phases[sel], nbrbins)
        ppBins = binned["ppBins"]
        ct = binned["ctsBins"].astype(float)
        ctErr = binned["ctsBinsErr"].astype(float) if "ctsBinsErr" in binned else np.sqrt(np.maximum(ct, 1.0))

        mean_ct = ct.mean()
        if mean_ct <= 0:
            ax.set_visible(False)
            panels.append((row, None, None, None, (t0, t1)))
            continue

        ct_norm = ct / mean_ct
        ct_err_norm = ctErr / mean_ct

        # Two cycles (match plotting_pp)
        secondCycle = 2 * np.pi if np.max(ppBins) > 1 else 1.0
        x_plot = np.append(ppBins, ppBins + secondCycle)
        y_plot = np.append(ct_norm, ct_norm)
        yerr_plot = np.append(ct_err_norm, ct_err_norm)

        panels.append((row, x_plot, y_plot, yerr_plot, (t0, t1)))

        ymin_all = min(ymin_all, np.nanmin(ct_norm))
        ymax_all = max(ymax_all, np.nanmax(ct_norm))

    # Common y-lims
    if not np.isfinite(ymin_all) or not np.isfinite(ymax_all):
        ymin_all, ymax_all = 0.85, 1.15
    else:
        pad = 0.05 * (ymax_all - ymin_all if ymax_all > ymin_all else 0.3)
        ymin_all = max(0.0, ymin_all - pad)
        ymax_all = ymax_all + pad

    # Second pass: draw
    for row, x_plot, y_plot, yerr_plot, (t0, t1) in panels:
        ax = axes[row, 0]
        if x_plot is None:
            continue

        ax.errorbar(x_plot, y_plot, yerr=yerr_plot, fmt='ok', zorder=10)
        ax.step(x_plot, y_plot, 'k+-', where='mid', zorder=10)

        ax.set_xlim(0.0, np.max(x_plot))
        ax.set_ylim(ymin_all, ymax_all)
        ax.tick_params(axis='both', labelsize=12)

        # Labels
        if row == 1:
            ax.set_xlabel(r'$\,\mathrm{Phase\,(cycles)}$', fontsize=12)
        else:
            ax.set_xticklabels([])
        ax.set_ylabel(r'$\,\mathrm{Normalized\,rate}$', fontsize=12)

        # Minimal title with integer MJD range
        ax.set_title(f"{int(t0)} - {int(t1)} MJD", fontsize=12, pad=4)

        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_linewidth(1.5)
            ax.tick_params(width=1.5)

    fig.tight_layout()

    if plotname is None:
        plt.show()
    else:
        fig.savefig(str(plotname) + '.pdf', format='pdf', dpi=300, bbox_inches="tight")
    return


def run_plots_from_yaml(config_path: str, time_energy_phase_df):
    """
    Read a YAML file that lists plots to generate and run them in order.
    Each item: {type: <key in _PLOT_REGISTRY>, params: {...kwargs for that function...}}
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    plots = cfg.get("plots", [])
    if not isinstance(plots, list):
        raise ValueError("YAML must contain a top-level 'plots' list.")

    for idx, item in enumerate(plots, 1):
        if not isinstance(item, dict):
            print(f"[WARN] plots[{idx}] is not a mapping; skipping")
            continue
        ptype = item.get("type")
        params = item.get("params", {}) or {}

        fn = _PLOT_REGISTRY.get(ptype)
        if fn is None:
            print(f"[WARN] Unknown plot type '{ptype}'; skipping")
            continue

        try:
            fn(time_energy_phase_df, **params)
        except TypeError as e:
            print(f"[WARN] Failed to run plot '{ptype}' with params {params}: {e}")


# map short names -> plotting functions
_PLOT_REGISTRY = {
    "pp": plotting_pp,
    "phase_energy": plotting_phase_energy,
    "phase_time": plotting_phase_time,
    "pp_grid": plotting_pp_grid,
    "before_after": plotting_pp_before_after,
}


def main():
    parser = argparse.ArgumentParser(description="BLA")
    parser.add_argument("eventfile", help="Event file", type=str)
    parser.add_argument("parfile", help="A tempo2 .par file", type=str)
    parser.add_argument("yamlconfig", help="YAML file listing plots to generate and "
                                           "corresponding input parameters", type=str, default=None)
    parser.add_argument("-el", "--enelow", help="Low energy filter in event file, default=0.3", type=float, default=0.3)
    parser.add_argument("-eh", "--enehigh", help="High energy filter in event file, default=10", type=float, default=10)
    parser.add_argument("-ts", "--tstart", help="Consider events from tstart (MJD)", type=float, default=40000)
    parser.add_argument("-te", "--tend", help="Stop at events earlier than tend (MJD)", type=float, default=70000)
    parser.add_argument("-op", "--outputplot", help="Name of output plot file", type=str, default=None)
    args = parser.parse_args()

    time_energy_phase_df, _ = prep_for_plotting(
        args.eventfile, args.parfile, args.enelow, args.enehigh, args.tstart, args.tend
    )

    run_plots_from_yaml(args.yamlconfig, time_energy_phase_df)


if __name__ == '__main__':
    main()
