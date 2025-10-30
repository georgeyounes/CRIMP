"""
readtimingmodel.py is a module that reads in a .par file. It is made to be as
compatible with tempo2 and PINT as possible. It reads in all glitches, frequency
derivatives, and any wave functions, which are typically used to align pulses in
noisy systems, i.e., magnetars and isolated neutron stars. Returns a dictionary
'timModParam' of parameter values, 'timModFlags' parameter flags (for fitting
purposes), and a nested dictionary 'timModBoth' of both. As a reminder this tool
does not yet accomodate IFUNC, proper motion, binary systems, or parallax. These
*may* get included in later versions, likely in that order
"""

import sys
import numpy as np
import re
from typing import Any, Optional

sys.dont_write_bytecode = True


class ReadTimingModel:
    """
    A class to read in a .par file into dictionaries.

    Returns:
      - readtaylorexpansion(): (values, flags, both)
      - readglitches():        (values, flags, both)
      - readwaves():           (values, flags, both)  # flags only for WAVE_OM
      - readfulltimingmodel(): (timModParams, timModFlags, timModBoth),
      - readstatistics(): (stats)  # dict of CHI2R, NTOA, and TRES
    """

    def __init__(self, timMod: str):
        self.timMod = timMod

    # ---------- helpers ----------
    @staticmethod
    def _parse_value_and_flag(tokens) -> tuple[np.float64, int]:
        """
        tokens: sequence of strings where tokens[0] is the numeric value,
                tokens[1] (optional) is a 0/1 flag.
        Returns (value, flag) with default flag=0 if absent or unparsable.
        """
        val = np.float64(tokens[0])
        flag = 0
        if len(tokens) > 1:
            try:
                fi = int(float(tokens[1]))
                if fi in (0, 1):
                    flag = fi
            except (ValueError, OverflowError):
                flag = 0
        return val, flag

    # ---------- Taylor expansion ----------
    def readtaylorexpansion(self):
        """
        Read Taylor expansion parameters (PEPOCH, F0..F12).
        Returns:
          timModParamTE (values-only dict),
          timModFlagsTE (flags-only dict),
          timModBothTE  (nested dict with {"value", "flag"})
        """
        keys = ["PEPOCH"] + [f"F{i}" for i in range(0, 13)]
        values: dict[str, Any] = {k: np.float64(0) for k in keys}
        flags: dict[str, int] = {k: 0 for k in keys}
        both: dict[str, dict[str, Any]] = {k: {"value": np.float64(0), "flag": 0} for k in keys}

        with open(self.timMod, "r") as data_file:
            for raw in data_file:
                li = raw.lstrip()
                toks = li.split()
                if not toks:
                    continue
                name = toks[0]
                if name in keys and len(toks) >= 2:
                    val, flg = self._parse_value_and_flag(toks[1:])
                    values[name] = val
                    flags[name] = flg
                    both[name] = {"value": val, "flag": flg}

        timModParamTE = values
        return timModParamTE, flags, both

    # ---------- Glitches ----------
    def readglitches(self):
        """
        Read glitch parameters.
        Returns:
          timModParamGlitches (values-only dict),
          timModFlagsGlitches (flags-only dict),
          timModBothGlitches  (nested dict with {"value", "flag"})
        """
        # first, find glitch ids
        glitch_ids = []
        with open(self.timMod, "r") as f:
            for raw in f:
                li = raw.lstrip()
                if li.startswith("GLEP_"):
                    m = re.match(r"GLEP_(\S+)", li)
                    if m:
                        glitch_ids.append(m.group(1))

        timModParamGlitches: dict[str, Any] = {}
        timModFlagsGlitches: dict[str, int] = {}
        timModBothGlitches: dict[str, dict[str, Any]] = {}

        if not glitch_ids:
            return timModParamGlitches, timModFlagsGlitches, timModBothGlitches

        # defaults
        bases = ["GLEP_", "GLPH_", "GLF0_", "GLF1_", "GLF2_", "GLF0D_", "GLTD_"]
        default_vals = {
            "GLEP_": np.float64(0),
            "GLPH_": np.float64(0),
            "GLF0_": np.float64(0),
            "GLF1_": np.float64(0),
            "GLF2_": np.float64(0),
            "GLF0D_": np.float64(0),
            "GLTD_": np.float64(1),  # to avoid a divide by 0
        }

        # initialize
        for jj in glitch_ids:
            for base in bases:
                k = f"{base}{jj}"
                timModParamGlitches[k] = default_vals[base]
                timModFlagsGlitches[k] = 0
                timModBothGlitches[k] = {"value": default_vals[base], "flag": 0}

        # fill from file
        with open(self.timMod, "r") as f:
            for raw in f:
                li = raw.lstrip()
                toks = li.split()
                if not toks:
                    continue
                name = toks[0]
                # check any of our glitch keys
                for jj in glitch_ids:
                    for base in bases:
                        key = f"{base}{jj}"
                        if name == key and len(toks) >= 2:
                            val, flg = self._parse_value_and_flag(toks[1:])
                            timModParamGlitches[key] = val
                            timModFlagsGlitches[key] = flg
                            timModBothGlitches[key] = {"value": val, "flag": flg}

        return timModParamGlitches, timModFlagsGlitches, timModBothGlitches

    # ---------- Waves ----------
    def readwaves(self):
        """
        Read wave parameters.
        Flags are parsed only for WAVE_OM
        Returns:
          timModParamWaves (values-only dict),
          timModFlagsWaves (flags-only dict; only key present is WAVE_OM if found),
          timModBothWaves  (nested dict; WAVE_OM has {"value","flag"}, others have flag=None)
        """
        timModParamWaves: dict[str, Any] = {}
        timModFlagsWaves: dict[str, int] = {}
        timModBothWaves: dict[str, dict[str, Any]] = {}

        # Collect harmonics
        wave_harmonics = set()
        with open(self.timMod, "r") as f:
            for raw in f:
                li = raw.lstrip()
                toks = li.split()
                if not toks:
                    continue
                if toks[0].startswith("WAVE"):
                    m = re.match(r"WAVE(\d+)$", toks[0])
                    if m:
                        wave_harmonics.add(int(m.group(1)))

        # pass to read WAVEEPOCH and WAVE_OM
        with open(self.timMod, "r") as f:
            for raw in f:
                li = raw.lstrip()
                toks = li.split()
                if not toks:
                    continue
                if toks[0] == "WAVEEPOCH" and len(toks) >= 2:
                    val = np.float64(toks[1])
                    timModParamWaves["WAVEEPOCH"] = val
                    timModBothWaves["WAVEEPOCH"] = {"value": val, "flag": None}
                elif toks[0] == "WAVE_OM" and len(toks) >= 2:
                    val, flg = self._parse_value_and_flag(toks[1:])
                    timModParamWaves["WAVE_OM"] = val
                    timModFlagsWaves["WAVE_OM"] = flg
                    timModBothWaves["WAVE_OM"] = {"value": val, "flag": flg}

        # harmonics (value-only A/B pair, no flags)
        for j in sorted(wave_harmonics):
            with open(self.timMod, "r") as f:
                for raw in f:
                    li = raw.lstrip()
                    if li.startswith(f"WAVE{j} ") or li.startswith(f"WAVE{j}\t"):
                        toks = li.split()
                        if len(toks) >= 3:
                            A = np.float64(toks[1])
                            B = np.float64(toks[2])
                            timModParamWaves[f"WAVE{j}"] = {"A": A, "B": B}
                            timModBothWaves[f"WAVE{j}"] = {"value": {"A": A, "B": B}, "flag": None}
                        break

        return timModParamWaves, timModFlagsWaves, timModBothWaves

    # ---------- Full model ----------
    def readfulltimingmodel(self):
        """
        Read full .par timing model.

        Returns (in order):
          timModParams : dict  # values only,
          timModFlags  : dict  # the 0/1 flags for the same keys,
          timModBoth   : dict  # nested {"value": ..., "flag": ...}
        """
        te_vals, te_flags, te_both = self.readtaylorexpansion()
        gl_vals, gl_flags, gl_both = self.readglitches()
        wv_vals, wv_flags, wv_both = self.readwaves()

        timModParams = {**te_vals, **gl_vals, **wv_vals}
        timModFlags = {**te_flags, **gl_flags, **wv_flags}
        timModBoth = {**te_both, **gl_both, **wv_both}

        return timModParams, timModFlags, timModBoth

    # ---------- Statistics ----------
    def readstatistics(self):
        """
        Read CHI2R, NTOA, and TRES values from the .par file.
        Returns:
            dict (with None for missing values)
        """
        stats = {"CHI2R": None, "CHI2R_DOF": None, "NTOA": None, "TRES": None}

        with open(self.timMod, "r") as f:
            for raw in f:
                toks = raw.strip().split()
                if not toks:  # empty line - simply skip
                    continue
                key = toks[0].upper()
                if key == "CHI2R":
                    try:  # if a key is written in the file without a value - don't crah
                        stats["CHI2R"] = float(toks[1])
                        if len(toks) > 2:
                            stats["CHI2R_DOF"] = int(toks[2])
                    except (ValueError, IndexError):
                        pass
                elif key == "NTOA":
                    try:
                        stats["NTOA"] = int(toks[1])
                    except (ValueError, IndexError):
                        pass
                elif key == "TRES":
                    try:
                        stats["TRES"] = float(toks[1])
                    except (ValueError, IndexError):
                        pass
        return stats

    def readmiscellaneous(self):
        """
        Read miscellaneous keywords in .par file - none if it doesn't existent
        Returns:
            dict (with None for missing values)
        """
        misc_schema = {
            "PSR": str,
            "RAJ": str,
            "DECJ": str,
            "POSEPOCH": float,
            "DMEPOCH": float,
            "START": float,
            "FINISH": float,
            "TZRMJD": float,
            "TZRFRQ": float,
            "TZRSITE": str,
            "CLK": str,
            "UNITS": str,
            "EPHEM": str

        }

        misc_keys = {k: None for k in misc_schema}

        with open(self.timMod, "r") as f:
            for raw in f:
                toks = raw.strip().split()
                if not toks:
                    continue
                key = toks[0].upper()
                if key in misc_schema:
                    try:
                        misc_keys[key] = misc_schema[key](toks[1])
                    except (IndexError, ValueError):
                        pass

        return misc_keys


def patch_par_values(
        in_path: str,
        out_path: str,
        *,
        new_values: dict[str, float],
        float_fmt: str = ".15g",
        uncertainties: Optional[dict[str, float]] = None,
        uncertainty_fmt: str = ".6g",
) -> None:
    """
    Write new .par file with new_values (and uncertainties) for certain parameters.
    If an uncertainty already exists after a flag, it will be REPLACED (not duplicated).
    Uncertainties are only written when a flag is present.
    """

    # KEY  <ws> VALUE [<ws> FLAG] [<ws> UNCERTAINTY] <tail>
    # Capture optional FLAG and optional UNCERTAINTY separately.
    line_re = re.compile(
        r"""
        ^([A-Za-z][A-Za-z0-9_]*)      # key
        (\s+)                         # whitespace before value
        (\S+)                         # value
        (?:                           # optional flag group
            (\s+)                     #   whitespace before flag
            ([01])                    #   flag
        )?
        (?:                           # optional uncertainty group (a single numeric token)
            \s+
            ([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
        )?
        (.*)$                         # tail (comments or anything else after)
        """,
        re.VERBOSE,
    )

    with open(in_path, "r") as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        core = line.rstrip("\n")
        m = line_re.match(core)
        if not m:
            out_lines.append(line)
            continue

        key, ws_val, old_val, ws_flag, flag, old_unc, tail = m.groups()

        # If this key isn't being updated, write the line back unchanged
        if key not in new_values:
            out_lines.append(line)
            continue

        new_val = format(float(new_values[key]), float_fmt)

        # No flag: just replace the value; do not add/modify uncertainty
        if flag is None:
            new_line = f"{key}{ws_val}{new_val}{tail}"
            out_lines.append(new_line + "\n")
            continue

        # With flag: decide which uncertainty to write (if any)
        if uncertainties is not None and key in uncertainties:
            # Replace any existing uncertainty with the new one
            unc_str = format(float(uncertainties[key]), uncertainty_fmt)
            new_line = f"{key}{ws_val}{new_val}{ws_flag}{flag} {unc_str}{tail}"
        else:
            # Keep existing uncertainty if it was present; otherwise write none
            if old_unc is not None:
                new_line = f"{key}{ws_val}{new_val}{ws_flag}{flag} {old_unc}{tail}"
            else:
                new_line = f"{key}{ws_val}{new_val}{ws_flag}{flag}{tail}"

        out_lines.append(new_line + "\n")

    with open(out_path, "w") as f:
        f.writelines(out_lines)


def patch_statistics(in_path: str, out_path: str, new_stats: dict[str, float]) -> None:
    """
    Update CHI2R, NTOA, and TRES lines in a .par file using new_stats dict.
    Missing keys in new_stats are ignored. Missing lines are appended at the end.
    """
    with open(in_path, "r") as f:
        lines = f.readlines()

    out_lines = []
    found = {"CHI2R": False, "NTOA": False, "TRES": False}

    for line in lines:
        toks = line.strip().split()
        if not toks:
            out_lines.append(line)
            continue
        key = toks[0].upper()

        if key == "CHI2R" and "CHI2R" in new_stats:
            chi2_str = f"{new_stats['CHI2R']}"
            if "CHI2R_DOF" in new_stats and new_stats["CHI2R_DOF"] is not None:
                out_lines.append(f"CHI2R          {chi2_str} {int(new_stats['CHI2R_DOF'])}\n")
            else:
                out_lines.append(f"CHI2R          {chi2_str}\n")
            found["CHI2R"] = True
        elif key == "NTOA" and "NTOA" in new_stats:
            out_lines.append(f"NTOA           {int(new_stats['NTOA'])}\n")
            found["NTOA"] = True
        elif key == "TRES" and "TRES" in new_stats:
            out_lines.append(f"TRES           {new_stats['TRES']}\n")
            found["TRES"] = True
        else:
            out_lines.append(line)

    # append missing ones
    for key in ["CHI2R", "NTOA", "TRES"]:
        if not found[key] and key in new_stats and new_stats[key] is not None:
            if key == "CHI2R":
                chi2_str = f"{new_stats['CHI2R']}"
                if "CHI2R_DOF" in new_stats and new_stats["CHI2R_DOF"] is not None:
                    out_lines.append(f"\nCHI2R          {chi2_str} {int(new_stats['CHI2R_DOF'])}")
                else:
                    out_lines.append(f"\nCHI2R          {chi2_str}\n")
            elif key == "NTOA":
                out_lines.append(f"\nNTOA           {int(new_stats['NTOA'])}")
            elif key == "TRES":
                out_lines.append(f"\nTRES           {new_stats['TRES']}")

    with open(out_path, "w") as f:
        f.writelines(out_lines)


def patch_miscellaneous(in_path: str, out_path: str, new_misc: dict[str, object]) -> None:
    """
    Update or append miscellaneous keys in a .par file using new_misc dict.

    Behavior:
      - Existing keys are updated in place.
      - Missing keys are appended at the end in the order they appear in new_misc.
      - Keys with None values are skipped entirely.
      - The original order and formatting of unrelated lines are preserved.
    """
    with open(in_path, "r") as f:
        lines = f.readlines()

    # Normalize keys for case-insensitive matching but preserve input order
    new_misc_norm = {k.upper(): v for k, v in new_misc.items()}
    found = {k.upper(): False for k, v in new_misc.items() if v is not None}
    out_lines = []

    for line in lines:
        toks = line.strip().split()
        if not toks:
            out_lines.append(line)
            continue

        key = toks[0].upper()
        if key in new_misc_norm and new_misc_norm[key] is not None:
            val = new_misc_norm[key]
            out_lines.append(f"{key:<15}{val}\n")
            found[key] = True
        else:
            out_lines.append(line)

    # Append missing ones in the same order as provided in new_misc
    for key, val in new_misc.items():
        if val is not None and not found.get(key.upper(), False):
            out_lines.append(f"\n{key.upper():<15}{val}")

    with open(out_path, "w") as f:
        f.writelines(out_lines)
