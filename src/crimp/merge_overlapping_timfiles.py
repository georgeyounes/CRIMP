"""
Merge multiple .tim files that include pulse numbers (-pn) by using overlapping TOAs as anchors.
Mainly utilized to keep track of pulse numbering in noisy magnetars/pulsars as one derives
phase-coherent solutions for short, overlapping time intervals

.tim files must have the -pn flag. Any two consecutive .tim files must have one TOA in common
TOA matching happens after rounding to 12 decimal points (in MJD). After matching first overlapping TOA,
ALL remaining, overlapping TOAs must have matching pulse numbers (otherwise exits with an error).

Can be run through the CLI tool mergeoverlappingtims
Note that individual .tim files can be provided or a .txt file with a list of .tim files on separate lines.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from crimp.timfile import readtimfile, PulseToAs

from crimp.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class MergeConfig:
    """Configuration for merging .tim files."""
    toa_round_decimals: int = 12  # fixed by design


def _toa_key(series: pd.Series, ndp: int) -> pd.Series:
    """
    Make a stable key for matching TOAs across files.
    We round to ndp decimals to be robust to minor formatting differences.
    """
    # Keep as float; rounding to 12 decimals is usually safe for NICER-style MJDs
    return series.round(ndp)


def _ensure_required_columns(df: pd.DataFrame, timfile: str) -> None:
    if "pulse_ToA" not in df.columns:
        raise ValueError(f"{timfile}: missing required column 'pulse_ToA'")
    if "pn" not in df.columns:
        raise ValueError(
            f"{timfile}: missing required pulse number column 'pn'. "
            "Make sure every TOA line has '-pn <int>'."
        )


def _read_one_tim(timfile: str) -> pd.DataFrame:
    df = readtimfile(timfile, skiprows=1)  # "FORMAT 1" header line
    _ensure_required_columns(df, timfile)

    # enforce numeric pn
    df["pn"] = pd.to_numeric(df["pn"], errors="raise").astype(np.int64)

    # sort by TOA (ascending)
    df = df.sort_values("pulse_ToA").reset_index(drop=True)
    return df


def _expand_inputs(inputs: list[str]) -> list[str]:
    """
    Expand inputs into a list of .tim files.

    Each input can be a:
      - .tim filename
      - .txt list file containing one .tim filename per line (like `ls *.tim > files.txt`)

    Input order is preserved, and we do minimal guessing:
      - If suffix is .txt: treat as list file
      - Else: treat as a tim path
    This step could be switched to type checking with python-magic - will be implemented in a later iteration
    """
    timfiles: list[str] = []

    for item in inputs:
        p = Path(item)

        if p.suffix.lower() == ".txt":
            if not p.exists():
                raise FileNotFoundError(f"List file not found: {item}")

            lines = p.read_text().splitlines()
            for ln in lines:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                timfiles.append(ln)
        else:
            timfiles.append(item)

    # basic validation
    if len(timfiles) < 2:
        raise ValueError("Need at least two .tim files to merge.")

    missing = [t for t in timfiles if not Path(t).exists()]
    if missing:
        raise FileNotFoundError("Missing .tim files:\n  " + "\n  ".join(missing))

    return timfiles


def _compute_shift_for_next(prev: pd.DataFrame, nxt: pd.DataFrame, cfg: MergeConfig) -> int:
    """
    Compute the integer shift to apply to nxt['pn'] so that the FIRST overlap matches prev.
    """
    prev_key = _toa_key(prev["pulse_ToA"], cfg.toa_round_decimals)
    nxt_key = _toa_key(nxt["pulse_ToA"], cfg.toa_round_decimals)

    overlap = pd.Index(prev_key).intersection(pd.Index(nxt_key))
    if overlap.empty:
        raise ValueError("No overlapping TOAs found between consecutive files.")

    # Anchor on FIRST overlap (earliest TOA)
    anchor_toa = float(np.min(overlap.to_numpy(dtype=float)))

    prev_pn = int(prev.loc[prev_key == anchor_toa, "pn"].iloc[0])
    nxt_pn = int(nxt.loc[nxt_key == anchor_toa, "pn"].iloc[0])

    shift = prev_pn - nxt_pn
    return shift


def _validate_overlap(prev: pd.DataFrame, nxt_shifted: pd.DataFrame, cfg: MergeConfig) -> None:
    """
    After shifting, every overlapping TOA must have identical pn.
    """
    prev_key = _toa_key(prev["pulse_ToA"], cfg.toa_round_decimals)
    nxt_key = _toa_key(nxt_shifted["pulse_ToA"], cfg.toa_round_decimals)

    overlap = pd.Index(prev_key).intersection(pd.Index(nxt_key))
    if overlap.empty:
        raise ValueError("No overlapping TOAs found (unexpected).")

    # Build maps TOA->pn for overlaps (should be 1-to-1 for typical tim files)
    prev_map = (
        prev.assign(_k=prev_key)
        .loc[lambda d: d["_k"].isin(overlap), ["_k", "pn"]]
        .drop_duplicates(subset="_k")
        .set_index("_k")["pn"]
    )
    nxt_map = (
        nxt_shifted.assign(_k=nxt_key)
        .loc[lambda d: d["_k"].isin(overlap), ["_k", "pn"]]
        .drop_duplicates(subset="_k")
        .set_index("_k")["pn"]
    )

    # Align and compare
    joined = prev_map.to_frame("pn_prev").join(nxt_map.to_frame("pn_next"), how="inner")
    bad = joined[joined["pn_prev"] != joined["pn_next"]]
    if not bad.empty:
        ex = bad.head(10)
        raise ValueError(
            "Overlap validation failed: overlapping TOAs have inconsistent pulse numbers after shifting.\n"
            f"First mismatches:\n{ex}"
        )


def _merge_two(prev_merged: pd.DataFrame, nxt: pd.DataFrame, cfg: MergeConfig) -> pd.DataFrame:
    """
    Merge prev_merged with nxt by shifting nxt pn and dropping duplicated TOAs.
    """
    shift = _compute_shift_for_next(prev_merged, nxt, cfg)

    nxt2 = nxt.copy()
    nxt2["pn"] = (nxt2["pn"] + shift).astype(np.int64)

    _validate_overlap(prev_merged, nxt2, cfg)

    # Combine, then drop duplicates by TOA key (keep earlier row)
    prev_key = _toa_key(prev_merged["pulse_ToA"], cfg.toa_round_decimals)
    nxt_key = _toa_key(nxt2["pulse_ToA"], cfg.toa_round_decimals)

    prev2 = prev_merged.copy()
    prev2["_toa_key"] = prev_key
    nxt2["_toa_key"] = nxt_key

    out = pd.concat([prev2, nxt2], ignore_index=True)
    out = out.sort_values("pulse_ToA").drop_duplicates(subset="_toa_key", keep="first")
    out = out.drop(columns=["_toa_key"]).reset_index(drop=True)

    logger.info("Applied shift %+d and merged (now %d TOAs).", shift, len(out))
    return out


def merge_tim_files(timfiles_or_listfiles: list[str], cfg: MergeConfig | None = None) -> pd.DataFrame:
    """
    Merge a sequence of .tim files into a single DataFrame with consistent pulse numbering.

    :param timfiles_or_listfiles: list of .tim files and/or a .txt list file(s) containing .tim filenames
    :type timfiles_or_listfiles: list[str]
    :param cfg: MergeConfig
    :type cfg: MergeConfig | None
    :return: merged ToAs DataFrame
    :rtype: pd.DataFrame
    """
    cfg = MergeConfig() if cfg is None else cfg
    timfiles = _expand_inputs(timfiles_or_listfiles)

    logger.info("Merging %d .tim files...", len(timfiles))
    df = _read_one_tim(timfiles[0])

    for tf in timfiles[1:]:
        nxt = _read_one_tim(tf)
        df = _merge_two(df, nxt, cfg)

    return df


def write_merged_tim(df: pd.DataFrame, outprefix: str, clobber: bool = False) -> None:
    """
    Write merged DataFrame to <outprefix>.tim using your existing PulseToAs writer.
    """
    PulseToAs(df).writetimfile(outprefix, clobber=clobber)


def main():
    parser = argparse.ArgumentParser(description="Merge .tim files with pulse numbers (-pn) using overlapping "
                                                 "TOAs as anchors.")
    parser.add_argument("timfiles", nargs="+",
                        help="Either .tim files, or .txt list file with one .tim filename per line.",
                        type=str)
    parser.add_argument("-ot", "--outputtim", help="Output prefix <outputtim>.tim (default=all_merged)",
                        type=str, default="all_merged")
    parser.add_argument("-cl", "--clobber", help="Override output .tim file? (default=False)",
                        default=False,
                        action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    merged = merge_tim_files(args.timfiles)
    write_merged_tim(merged, args.outputtim, clobber=args.clobber)

    logger.info("Wrote %s.tim (%d TOAs)", args.outputtim, len(merged))


if __name__ == "__main__":
    main()
