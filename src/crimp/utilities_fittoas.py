"""
Some utility functions for fit_toas.py modules
"""

import re
import numpy as np
from dataclasses import dataclass
import yaml
import copy

from crimp.calcphase import Phases


def list_fit_keys(parfile: dict):
    """
    Given a parfile dictionary of {param1: {value: , flag: }, ...}, list the keys with flag = 1
    If WAVES (whitening model) is present, and WAVE_OM has flag = 1, then all WAVE harmonics
    e.g., {WAVE1: {A: , B: }, ...} will be added to keysas WAVE1_A, WAVE1_B, etc.
    """
    keys = [k for k, v in parfile.items()
            if isinstance(v, dict)
            and "value" in v
            and "flag" in v
            and v["flag"] == 1]

    # Special behavior for WAVE_OM
    if "WAVE_OM" in parfile and parfile["WAVE_OM"].get("flag") == 1:
        # Remove WAVE_OM itself if it was added
        keys = [k for k in keys if k != "WAVE_OM"]
        # Add all WAVE1, WAVE2, ... keys
        wave_keys = [
            f"{k}_{suffix}"
            for k in parfile.keys()
            if re.match(r"^WAVE(\d+)?$", k)
            for suffix in ["A", "B"]]
        keys.extend(wave_keys)
    return keys


def extract_free_params(parfile, yaml_initialguesses: str | None = None):
    """
    Extract free parameter vector from a YAML file of initial guesses or as all free parameters set to 0
    :param parfile: timing solution as a dictionary
    :type parfile: dict
    :param yaml_initialguesses: YAML file containing initial guesses
    :type yaml_initialguesses: str | None
    :return: Array of the values of free parameters return[0], and their corresponding key names return[1]
    :rtype: tuple
    """
    keys = list_fit_keys(parfile)

    # if YAML file provided initialize parameters from it
    if yaml_initialguesses is not None:
        p0 = initialize_params(keys, yaml_initialguesses)
    else:
        p0 = np.zeros(len(keys), dtype=float)

    return p0, keys


def inject_free_params(parfile, pvec: np.ndarray, keys: list):
    """
    Inject new values for free parameter
    :param parfile: timing solution as a dictionary
    :type parfile: dict
    :param pvec: vector of new values to be injected into the parfile
    :type pvec: np.ndarray
    :param keys: list of free parameter keys
    :type keys: list
    :return: parfile dictionary for fitting purposes (values changed in place) return[0], parfile dictionary where
    injected values are assumed deltas and were added to the original fit parameters return[1]
    :rtype: tuple
    """
    parfile_dict_fit = {}
    parfile_dict_full = {}

    def is_scalar_param(vv):
        return isinstance(vv, dict) and "value" in vv and not isinstance(vv["value"], dict)

    def is_glep(name):
        return bool(re.match(r"^GLEP_\d+$", name))

    def is_gltd(name):
        return bool(re.match(r"^GLTD_\d+$", name))

    def is_waveep(name):
        return name == "WAVEEPOCH"

    def is_waveom(name):
        return name == "WAVE_OM"

    def is_wave_ab(name):
        return bool(re.match(r"^WAVE\d+_[AB]$", name))

    # Seed outputs with defaults
    for k, v in parfile.items():
        if is_scalar_param(v):
            base = v["value"]
            if k == "PEPOCH" or is_glep(k) or is_waveep(k) or is_waveom(k):
                parfile_dict_fit[k] = base
            else:
                parfile_dict_fit[k] = 0.0
            parfile_dict_full[k] = base
        else:
            parfile_dict_fit[k] = copy.deepcopy(v)
            parfile_dict_full[k] = copy.deepcopy(v)

    # Apply overrides from keys/pvec
    for k, val in zip(keys, pvec):
        # Do not override these EPOCHS and never fit for them
        if k == "PEPOCH" or is_waveep(k) or is_waveom(k):
            continue

        # Dealing with WAVE model
        if is_wave_ab(k):
            base_name, coeff = k.rsplit("_", 1)  # e.g. WAVE1_A
            if base_name not in parfile:  # just in case but should not happen
                raise KeyError(f"Parameter '{base_name}' not found in parfile.")
            base_coeff = parfile[base_name]["value"][coeff]
            delta = val
            parfile_dict_fit[base_name]["value"][coeff] = delta
            parfile_dict_full[base_name]["value"][coeff] = base_coeff - delta
            continue

        # Move on from EPOCHs and WAVE model
        if k not in parfile:  # Again, should not happen but just in case
            raise KeyError(f"Parameter '{k}' not found in parfile.")

        v = parfile[k]
        base = v["value"]
        delta = val

        parfile_dict_fit[k] = delta
        if is_glep(k):  # Fit for actual epoch
            parfile_dict_full[k] = delta
        elif is_gltd(k):  # Glitch exponential term is increment of what is in the initial .par file
            parfile_dict_full[k] = base + delta
        else:  # Same as above for the frequency terms (note the negative sign since we work in phase space)
            parfile_dict_full[k] = base - delta

    return parfile_dict_fit, parfile_dict_full


def validate_parfile(parfile):
    """
    Validate a flags-only initial timing model:
      - every scalar param is {"value": float-like, "flag": 0 or 1}
      - nested structures (e.g., WAVE1 dicts) are passed
      - ensure at least one flag==1
      - If fit WAVES, ensures that no other parameter has flag = 1 (WAVES cannot be fit with any other model)
    Raises ValueError with a (hopefully) helpful message if invalid
    :param parfile: timing solution as a .par file
    :type parfile: dict
    """
    if not isinstance(parfile, dict):
        raise ValueError("Initial timing model must be a dict")

    n_fit = 0
    for k, v in parfile.items():

        if k == "WAVEEPOCH" or re.match(r"^WAVE(\d+)$", k):
            # Skip WAVEEPOCH
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


# -------- Objective factory (closure) --------
def gaussian_nll(y, mu, sigma):
    """
    Gaussian negative log-likelihood
    """
    r = (y - mu) / sigma
    nll = 0.5 * np.sum(r**2 + np.log(2.0 * np.pi * sigma**2))
    return nll


def make_nll(x, y, y_err, parfile, yaml_init=None):
    """
    Creates the NLL
    :param x: time array in MJD
    :type x: np.ndarray
    :param y: Phase residuals
    :type y: np.ndarray
    :param y_err: Uncertainties on phase residuals
    :type y_err: np.ndarray
    :param parfile: timing solution as a .par file
    :type parfile: dict
    :param yaml_init: YAML file containing initial guesses
    :type yaml_init: str | None
    Returns:
      nll(pvec) -> scalar NLL
      p0 (initial vector)
      keys (fit parameter names)
      parfile (the same validated parfile you passed in)
    """
    validate_parfile(parfile)

    p0, keys = extract_free_params(parfile, yaml_init)

    def nll(pvec):
        # Inject new parameters in parfile during minimization process
        mu = model_phase_residuals(x, parfile, pvec, keys)
        return gaussian_nll(y - np.mean(y), mu, y_err)

    return nll, p0, keys, parfile


def initialize_params(keys: list[str], yaml_init: str) -> np.ndarray:
    """
    Initialize the parameters in keys as given in a YAML file
    Return a 1D NumPy array of initial parameter values in the same order as keys
    :param keys: List of parameter names (order will be preserved)
    :param yaml_init: Path to YAML file defining initial guesses
    :return: np.ndarray 1D array of parameter values ordered as in `keys`
    """
    # Read initial guesses (unlike the name says, no need to have priors if you are not attempting MCMC
    # initial guesses will suffice - see initguess_prior_from_yaml for more info
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


def model_phase_residuals(x_mjd, timmodel, pvec, keys: list[str]) -> np.ndarray:
    """
    Calculate mean-subtracted phase residuals according to a timing model - for fitting purposes only
    Use the module calcphase.py for all other purposes
    """
    newparfile_dict_fit, newparfile_dict_full = inject_free_params(timmodel, pvec, keys)
    if all("wave" in s.lower() for s in keys):
        # WAVES need special treatment since it requires F0
        newparfile_dict_fit['F0'] = newparfile_dict_full['F0']
        phases = Phases(x_mjd, newparfile_dict_fit).waves()
    elif not any("wave" in s.lower() for s in keys):
        # Calculate phases according to new injected parameters
        delta_phases_te = Phases(x_mjd, newparfile_dict_fit).taylorexpansion()
        delta_phases_gl = Phases(x_mjd, newparfile_dict_fit).glitches()
        delta_phases_waves = Phases(x_mjd, newparfile_dict_full).waves()  # We are not fitting for waves
        phases = delta_phases_te + delta_phases_gl + delta_phases_waves
    else:
        delta_phases_te = Phases(x_mjd, newparfile_dict_fit).taylorexpansion()
        delta_phases_gl = Phases(x_mjd, newparfile_dict_fit).glitches()
        # Special treatment for WAVES
        newparfile_dict_fit['F0'] = newparfile_dict_full['F0']
        delta_phases_waves = Phases(x_mjd, newparfile_dict_fit).waves()  # Here we fit for waves
        phases = delta_phases_te + delta_phases_gl + delta_phases_waves
    phases -= np.mean(phases)
    return phases


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
