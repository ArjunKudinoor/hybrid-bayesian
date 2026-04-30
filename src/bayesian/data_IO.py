#!/usr/bin/env python
'''
Module related to reading and writing of tables of observables into numpy arrays

CORE FUNCTIONALITY:
===================
Read design/prediction/data tables (.dat files) into structured dictionary format:
 - initialize_observables_dict_from_tables() -- main entry point for loading data
 - read/write_dict_to_h5() -- HDF5 serialization for observables dictionary
 - predictions_matrix_from_h5() -- construct prediction matrix (design_points × observable_bins)
 - design_array_from_h5() -- extract design points
 - data_array_from_h5() -- extract experimental data with systematic correlations
 - data_dict_from_h5() -- dictionary format for experimental data
 - observable_dict_from_matrix() -- convert stacked matrix to per-observable dict
 - observable_matrix_from_dict() -- convert per-observable dict to stacked matrix
 - observable_label_to_keys() -- parse observable identifiers
 - sorted_observable_list_from_dict() -- enforce consistent observable ordering

SYSTEMATIC UNCERTAINTY SUPPORT (August - November 2025, Jingyu Zhang):
========================================================================

FOUR OPERATIONAL MODES:
-----------------------

+------------------+----------------------+-------------------------+---------------------+
| Mode             | Configuration        | Intra-Observable        | Cross-Observable    |
+------------------+----------------------+-------------------------+---------------------+
| Fallback         | (no systematics)     | None                    | None                |
| Legacy (STAT)    | sum:length:strength  | Exp. decay corr.        | Not possible        |
| Advanced (NEW)   | name:tag             | Full correlation        | Via group tags      |
| Expert           | external_covariance  | User-defined            | User-defined        |
+------------------+----------------------+-------------------------+---------------------+

MODE DETAILS:

1. Fallback Mode - No Systematics
   Physics: Incorrect (ignores systematic uncertainties entirely)
   Status: Backward compatibility only
   When active: No correlation_manager in config
   Covariance: C = C_emulator + diag(σ_stat²)

2. Legacy Mode - Summed Systematics 
   Config format: sys_data: ['sum:cor_length:cor_strength']
   Algorithm: Sum in quadrature → apply exponential decay correlation within observable
   Covariance: C = C_emulator + diag(σ_stat²) + C_sys_correlated_within_obs
   
3. Advanced Mode - Individual Systematics (Recommended)
   Config format: sys_data: ['jec:alice', 'taa:global', 'tracking:uncor']
   Algorithm: Track each systematic → correlate via group tags
   Cross-observable: Same tag → fully correlated, different tags → uncorrelated
   Special: 'uncor' tag → diagonal contribution only
   Status: Recommended for all new precision measurements
   Covariance: C = C_emulator + diag(σ_stat²) + C_sys_cross_obs
   
4. Expert Mode - External Covariance
   Config format: external_covariance: 'path/to/matrix.txt'
   Algorithm: User provides complete experimental covariance
   Replaces: Both statistical and systematic uncertainties
   Responsibility: User ensures physical validity
   Status: Expert feature with minimal validation
   Covariance: C = C_emulator + C_external

FILE FORMAT AND PARSING:
------------------------
.dat File Structure:
  # Label xmin xmax y y_err_stat sys_jec sys_taa ...
  Data columns read by np.loadtxt (Label skipped):
    col 0: xmin, col 1: xmax, col 2: y, col 3: y_err_stat
    col 4+: systematic sources (sys_jec, sys_taa, etc.)

Config → Data Mapping:
  Config specifies: 'jec:alice' (name + correlation tag)
  .dat file contains: 'sys_jec' (base name only)
  Mapping: strip tag from config → match base name in file
  Result: sys_jec column → jec:alice systematic with correlation tag

CONFIGURATION EXAMPLES:
-----------------------

Legacy Mode (summed with correlation):
  observable_list:
    - observable: 'jet_pt_alice'
      sys_data: ['sum:10:0.8']  # cor_length=10 bins, cor_strength=0.8

Advanced Mode (individual with tags):
  observable_list:
    - observable: 'jet_pt_alice'
      sys_data: ['jec:alice', 'taa:global']  # JEC specific, TAA global
    - observable: 'jet_pt_cms'
      sys_data: ['jec:cms', 'taa:global']    # TAA correlated via 'global' tag

Expert Mode (external covariance):
  observable_list:
    - external_covariance: 'path/to/covariance_matrix.txt'

CLOSURE TEST SUPPORT:
---------------------
Generate pseudodata from validation design points for closure tests:
- Proper shape handling across all operational modes
- Systematic uncertainties copied from experimental data
- Validates emulator performance against known inputs

DATA STRUCTURE:
---------------
observables['Data'][obs_label]['y']              -- measurement values
                                ['y_err_stat']    -- statistical uncertainties
                                ['systematics']   -- dict of systematic arrays 
                                ['xmin'], ['xmax'] -- bin edges for plotting

observables['Prediction'][obs_label]['y']        -- theory predictions
                                    ['y_err_stat'] -- statistical uncertainties
                                    ['systematics'] -- systematic uncertainties 
                                    ['xmin'], ['xmax'] -- copied from Data

observables['correlation_manager']                -- SystematicCorrelationManager 
observables['external_covariance']                -- User-provided covariance (Expert mode)

DESIGN NOTES:
-------------
- "Design" key contains parameters, "Design_indices" contains point indices
- Systematics stored as separate columns, NOT combined into total uncertainty
- Empty systematics dict maintained for backward compatibility
- Correlation tags only in config files, NOT in .dat files
- Base systematic names in .dat files unchanged (e.g., s_jec, s_taa)

For systematic correlation structure details, see systematic_correlation.py
For covariance visualization, see plot_covariance.py

.. codeauthor:: J.Mulligan
.. codeauthor:: R.Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
.. codeauthor:: Jingyu Zhang <jingyu.zhang@cern.ch>, Vanderbilt
'''

from __future__ import annotations

from bayesian.systematic_correlation import SystematicCorrelationManager
import fnmatch
import os
import logging
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

import attrs
import numpy as np
import numpy.typing as npt
from silx.io.dictdump import dicttoh5, h5todict


logger = logging.getLogger(__name__)


####################################################################################################################
# UTILITY FUNCTIONS
####################################################################################################################

def _recursive_defaultdict():
    '''
    Create a nested defaultdict

    :return recursive defaultdict
    '''
    return defaultdict(_recursive_defaultdict)


def _validate_and_flatten_array(array: np.ndarray, name: str) -> np.ndarray:
    """
    Validate array is 1D or flattenable to 1D.
    
    Args:
        array: Input array
        name: Name for error messages
        
    Returns:
        1D array
        
    Raises:
        ValueError: If array cannot be flattened to 1D
    """
    if array.ndim == 1:
        return array
    
    if array.ndim == 2 and (array.shape[0] == 1 or array.shape[1] == 1):
        result = array.flatten()
        logger.debug(f"{name}: Converted from 2D {array.shape} to 1D {result.shape}")
        return result
    
    raise ValueError(
        f"{name}: Expected 1D array, got shape {array.shape}. "
        f"Cannot automatically flatten."
    )


def _generate_pseudodata(prediction_data: np.ndarray, exp_uncertainty: np.ndarray, 
                         pseudodata_index: int, observable_label: str) -> np.ndarray:
    """
    Generate pseudodata from validation predictions.
    
    Args:
        prediction_data: Prediction matrix (n_bins, n_design_points)
        exp_uncertainty: Experimental uncertainties (n_bins,)
        pseudodata_index: Design point index to use
        observable_label: Observable name for error messages
        
    Returns:
        Pseudodata array (n_bins,)
    """
    if prediction_data.ndim != 2:
        raise ValueError(
            f"Prediction data for {observable_label} has unexpected shape: {prediction_data.shape}. "
            f"Expected 2D array (n_bins, n_design_points)"
        )
    
    if prediction_data.shape[1] <= pseudodata_index:
        raise ValueError(
            f"Validation prediction data not available for observable {observable_label}. "
            f"Available design points: {prediction_data.shape[1]}, "
            f"Requested index: {pseudodata_index}"
        )
    
    prediction_central_value = prediction_data[:, pseudodata_index]
    pseudodata = prediction_central_value + np.random.normal(loc=0., scale=exp_uncertainty)
    
    if pseudodata.ndim != 1:
        raise ValueError(
            f"Generated pseudodata for {observable_label} has unexpected shape: {pseudodata.shape}. "
            f"Expected 1D array."
        )
    
    return pseudodata


####################################################################################################################
# SYSTEMATIC UNCERTAINTY HELPER FUNCTIONS
####################################################################################################################

_COORDINATE_COLUMNS = {'xmin', 'xmax', 'y'}
_STAT_COLUMN_ALIASES = ('stat', 'y_err_stat', 'y_err')


def _strip_asymmetric_suffix(name):
    """Strip trailing ',low' or ',high' from a column label, returning base name."""
    if name.endswith(',low'):
        return name[:-4]
    if name.endswith(',high'):
        return name[:-5]
    return name


def _parse_data_header(filepath):
    """
    Parse the '# Label ...' header of a Data .dat file.

    Supports two conventions for uncertainty columns:
      - Symmetric:  a single column named e.g. 'y_err_stat', 'sys_jec'
      - Asymmetric: a pair 'name,low' and 'name,high'; these are grouped under
                    the shared base name 'name'

    Coordinate columns (xmin, xmax, y) are skipped. Column indices are in the
    numpy array produced by np.loadtxt (i.e. 'Label' token does not consume an
    index).

    Returns:
        Dict mapping base_name → list of column indices (1 or 2 entries).
    """
    grouped = {}
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                if line.startswith('#') and 'label' in line.lower() and 'xmin' in line.lower():
                    tokens = line.strip('#').strip().split()
                    data_col_index = 0
                    for tok in tokens:
                        lower = tok.lower()
                        if lower == 'label':
                            continue
                        if lower in _COORDINATE_COLUMNS:
                            data_col_index += 1
                            continue
                        base = _strip_asymmetric_suffix(tok)
                        grouped.setdefault(base, []).append(data_col_index)
                        data_col_index += 1
                    break

                if line_num > 10:
                    break

    except Exception as e:
        logger.warning(f"Could not parse header for {filepath}: {e}")

    return grouped


def _read_grouped_columns(filepath, grouped_columns):
    """
    Read grouped columns from a Data file.

    For each base name, loads the listed columns and reduces multi-column
    entries (asymmetric low/high pairs) by elementwise maximum.

    Returns:
        Dict mapping base_name → 1D numpy array.
    """
    if not grouped_columns:
        return {}

    try:
        full_data = np.loadtxt(filepath, ndmin=2)
    except Exception as e:
        logger.error(f"Failed to load data from {filepath}: {e}")
        return {}

    result = {}
    for base, indices in grouped_columns.items():
        valid = [i for i in indices if i < full_data.shape[1]]
        if not valid:
            logger.warning(
                f"Column '{base}' at indices {indices} not in {filepath} "
                f"(only {full_data.shape[1]} columns)"
            )
            continue
        if len(valid) == 1:
            result[base] = full_data[:, valid[0]]
        else:
            stacked = np.stack([full_data[:, i] for i in valid], axis=0)
            result[base] = np.max(stacked, axis=0)
            logger.debug(f"  Reduced {len(valid)} columns for '{base}' via elementwise max")

    return result

def _read_theory_systematics(table_dir, model, observable_name, theory_systematics):
    """
    Read theory systematic uncertainty files for model predictions.
    
    Theory systematics represent uncertainties in the theoretical model itself
    (e.g., scale variations, PDF uncertainties) as opposed to experimental
    measurement uncertainties (which are in sys_data).
    
    Expected file format:
        {table_dir}/Prediction/Prediction__{model}__{observable_name}__systs_{sys_name}.dat
    
    Each file should have shape (n_bins, n_design_points) matching prediction values.
    
    Args:
        table_dir: Base table directory
        model: Model name (e.g., 'exponential')
        observable_name: Full observable label
        theory_systematics: List of systematic names from config's sys_theory
        
    Returns:
        Dict mapping systematic names to arrays of shape (n_bins, n_design_points).
        Returns empty dict if no theory systematics configured or files not found.
        
    Example:
        >>> theory_syst = _read_theory_systematics(
        ...     'tables/', 'exponential', '5020__PbPb__hadron__pt_ch_cms____0-5', 
        ...     ['scale', 'pdf']
        ... )
        >>> # Returns: {'scale': array(...), 'pdf': array(...)}
    """
    theory_syst_data = {}
    prediction_dir = os.path.join(table_dir, 'Prediction')
    base_filename = f'Prediction__{model}__{observable_name}'
    
    if not theory_systematics:
        return theory_syst_data
    
    for sys_name in theory_systematics:
        syst_filepath = os.path.join(prediction_dir, f'{base_filename}__systs_{sys_name}.dat')
        
        if os.path.exists(syst_filepath):
            try:
                syst_array = np.loadtxt(syst_filepath, ndmin=2)
                theory_syst_data[sys_name] = syst_array
                logger.info(f"Loaded theory systematic '{sys_name}' for {observable_name}")
            except Exception as e:
                logger.warning(f"Failed to load theory systematic {sys_name}: {e}")
        else:
            logger.debug(f"Theory systematic file not found: {syst_filepath}")
    
    if theory_systematics and not theory_syst_data:
        logger.warning(
            f"No theory systematic files found for {observable_name}. "
            f"Expected files: {[f'{base_filename}__systs_{s}.dat' for s in theory_systematics]}"
        )
    
    return theory_syst_data

def _read_external_covariance(filepath):
    """
    Read external covariance matrix from file.
    
    Expert feature - minimal validation, user is responsible for correctness.
    Validation includes: 2D, square, symmetry check, eigenvalue check.
    
    Args:
        filepath: Path to covariance matrix file
        
    Returns:
        Covariance matrix or None if loading fails
    """
    try:
        logger.info(f"Reading external covariance from {filepath}")
        cov = np.loadtxt(filepath)
        
        if cov.ndim != 2:
            raise ValueError(f"Covariance must be 2D, got shape {cov.shape}")
        
        if cov.shape[0] != cov.shape[1]:
            raise ValueError(f"Covariance must be square, got shape {cov.shape}")
        
        if not np.allclose(cov, cov.T, rtol=1e-5):
            logger.warning("External covariance is not symmetric")
        
        eigenvals = np.linalg.eigvalsh(cov)
        if np.any(eigenvals < -1e-8):
            logger.warning(f"External covariance has negative eigenvalues: min={np.min(eigenvals):.2e}")
        
        logger.info(f"External covariance loaded: shape={cov.shape}, trace={np.trace(cov):.2e}")
        
        return cov
        
    except Exception as e:
        logger.error(f"Failed to read external covariance: {e}")
        return None

def _read_external_stat_covariance(filepath, n_bins, obs_label):
    """
    Load per-observable external statistical covariance matrix.
    
    Args:
        filepath: Path to covariance matrix file
        n_bins: Expected number of bins for this observable
        obs_label: Observable label (for error messages)
    
    Returns:
        Covariance matrix of shape (n_bins, n_bins)
    
    Raises:
        ValueError: If file cannot be loaded or has wrong shape
    """
    try:
        cov_matrix = np.loadtxt(filepath)
    except Exception as e:
        raise ValueError(f"Failed to load external_stat_cov for {obs_label} from {filepath}: {e}")
    
    # Validate shape
    if cov_matrix.shape != (n_bins, n_bins):
        raise ValueError(
            f"External stat covariance for {obs_label} has shape {cov_matrix.shape}, "
            f"expected ({n_bins}, {n_bins})"
        )
    
    # Validate symmetry
    if not np.allclose(cov_matrix, cov_matrix.T):
        logger.warning(f"External stat covariance for {obs_label} is not symmetric. Symmetrizing.")
        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
    
    # Validate positive semi-definite
    eigvals = np.linalg.eigvalsh(cov_matrix)
    if np.any(eigvals < -1e-10):
        raise ValueError(
            f"External stat covariance for {obs_label} is not positive semi-definite. "
            f"Minimum eigenvalue: {eigvals.min()}"
        )
    
    logger.info(f"Loaded external statistical covariance for {obs_label} from {filepath}")
    return cov_matrix

def _sum_systematics_quadrature(systematics_dict):
    """
    Sum systematics in quadrature: σ_total = √(Σ σᵢ²).
    
    Used when config specifies 'sum:...' to combine multiple systematic
    sources into a single combined systematic uncertainty.
    
    Args:
        systematics_dict: Dict of systematic arrays {name: array}
                         e.g., {'jec': [0.1, 0.2], 'taa': [0.05, 0.08]}
    
    Returns:
        Array of summed systematic uncertainties
    """
    if not systematics_dict:
        logger.warning("No systematics to sum - returning empty array")
        return np.array([])
    
    arrays = list(systematics_dict.values())
    
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) > 1:
        logger.error(f"Systematic arrays have different lengths: {lengths}")
        raise ValueError(f"Cannot sum systematics with different lengths: {lengths}")
    
    sum_squared = sum(arr**2 for arr in arrays)
    summed = np.sqrt(sum_squared)
    
    logger.info(f"Summed {len(arrays)} systematics in quadrature")
    logger.debug(f"  Result: {len(summed)} bins, mean uncertainty = {np.mean(summed):.4f}")
    
    return summed

def _filter_systematics_by_config(systematic_data, config_systematics):
    """
    Filter systematics dictionary based on config specification.
    
    Maps configuration systematic names (with correlation tags) to 
    data file systematic names (without tags). Correlation tags
    define correlation structure but aren't present in data files.
    
    Mapping examples:
        Config 'jec:cms' → Data file 'jec' or 'sys_jec'
        Config 'taa:global' → Data file 'taa' or 'sys_taa'
    
    Args:
        systematic_data: Dict of systematic arrays from file.
                        Keys are base names like 'jec', 'taa', 'sys_jec', 'sys_taa'
        config_systematics: List of systematic names from config.
                           May include correlation tags like 'jec:cms', 'taa:global'
        
    Returns:
        Dict of filtered systematics with base names as keys
        
    Example:
        >>> data = {'sys_jec': array([0.1, 0.2]), 'sys_taa': array([0.05, 0.08])}
        >>> config = ['jec:cms', 'taa:global']
        >>> result = _filter_systematics_by_config(data, config)
        >>> # Returns: {'jec': array([0.1, 0.2]), 'taa': array([0.05, 0.08])}
    """
    if not config_systematics:
        logger.debug("No systematic filtering needed - config_systematics is empty")
        return {}
    
    filtered_systematics = {}
    
    for sys_full_name in config_systematics:
        # Extract base name from full name (remove correlation tag)
        if ':' in sys_full_name:
            base_sys_name, _ = sys_full_name.split(':', 1)
            logger.debug(f"Mapping config '{sys_full_name}' → base name '{base_sys_name}'")
        else:
            base_sys_name = sys_full_name
            logger.debug(f"Config systematic '{sys_full_name}' has no correlation tag")
        
        # Look for base name in systematic_data
        # Data files may have 'jec' or 'sys_jec' format
        if base_sys_name in systematic_data:
            filtered_systematics[base_sys_name] = systematic_data[base_sys_name]
            logger.debug(f"  Found '{base_sys_name}' in data")
        elif f'sys_{base_sys_name}' in systematic_data:
            # Handle sys_ prefix if present in data file
            filtered_systematics[base_sys_name] = systematic_data[f'sys_{base_sys_name}']
            logger.debug(f"  Found 'sys_{base_sys_name}' in data, mapped to '{base_sys_name}'")
        else:
            logger.warning(
                f"Systematic '{sys_full_name}' specified in config but not found in data. "
                f"Looked for: '{base_sys_name}' or 'sys_{base_sys_name}'"
            )
    
    logger.debug(f"Filtered {len(filtered_systematics)}/{len(config_systematics)} systematics from config")
    
    return filtered_systematics

def _parse_config_observables(analysis_config, correlation_groups=None):
    """
    Parse observable configuration for systematic support.
    Handles both old and new formats, and detects external covariance mode.
    
    :param analysis_config: Analysis configuration dictionary
    :return: Tuple of (parsed_observables_list, correlation_manager, external_cov_file)
    """
    correlation_manager = SystematicCorrelationManager()
    
    # Check for external covariance file
    external_cov_file = analysis_config.get('external_covariance_file', None)
    
    if external_cov_file:
        logger.info(f"External covariance mode enabled: {external_cov_file}")
        logger.info("Systematic uncertainties (sys_data) will be ignored")
    
    # Parse observables similar to existing implementation
    parsed_observables = []
    
    try:
        for emulation_group_settings in analysis_config["parameters"]["emulators"].values():
            observable_config_list = emulation_group_settings.get('observable_list', [])
            
            for obs_config in observable_config_list:
                if isinstance(obs_config, str):
                    # Old format: just observable name - no systematics
                    parsed_observables.append((obs_config, [], []))
                elif isinstance(obs_config, dict) and 'observable' in obs_config:
                    # New format with correlation tags
                    obs_name = obs_config['observable']
                    # Ignore sys_data if external covariance is used
                    sys_data = [] if external_cov_file else obs_config.get('sys_data', [])
                    sys_theory = obs_config.get('sys_theory', [])
                    
                    # Extract per-observable external stat cov (if specified)
                    external_stat_cov = obs_config.get('external_stat_cov', None)
                    
                    # Append with external_stat_cov info
                    parsed_observables.append((obs_name, sys_data, sys_theory, external_stat_cov))
                else:
                    logger.warning(f"Unrecognized observable config format: {obs_config}")
                    
    except KeyError as e:
        logger.error(f"Config structure issue: {e}")
        logger.error("Expected structure: analysis_config['parameters']['emulators'][group]['observable_list']")
        return parsed_observables, correlation_manager, None
    
    # Parse the correlation configuration (only if no external covariance)
    if not external_cov_file:
        correlation_manager.parse_configuration(parsed_observables)

        # Parse correlation_groups section
        if correlation_groups:
            logger.info(f"Found correlation_groups with {len(correlation_groups)} group tags")
            correlation_manager.set_correlation_parameters(correlation_groups)
        else:
            logger.warning("No correlation_groups provided (using default full correlation)")

        logger.info(f"Created correlation manager with {len(correlation_manager.get_all_systematic_names())} systematics")
    else:
        logger.info("Skipping systematic correlation parsing (external covariance mode)")
    
    # Return BOTH the parsed observables list, correlation manager, AND external_cov_file
    return parsed_observables, correlation_manager, external_cov_file

####################################################################################################################
# DATA LOADING FUNCTIONS - CORE INTERFACE
####################################################################################################################

def data_array_from_h5(
    output_dir: str | Path,
    filename: str,
    pseudodata_index: int = -1,
    observable_filter: ObservableFilter | None = None
) -> dict[str, Any]:
    """
    Load experimental data array from observables.h5 with systematic correlation support.
    
    Main entry point that routes to appropriate loading function based on
    what's available in the observables dict.
    
    Args:
        output_dir: Directory containing observables.h5
        filename: Filename (typically 'observables.h5')
        pseudodata_index: Index for closure test (-1 for experimental data)
        observable_filter: Optional filter for observables
        
    Returns:
        Dict with keys: y, y_err_stat, y_err_syst, systematic_names, observable_ranges,
        and optionally: correlation_manager or external_covariance
    """
    observables = read_dict_from_h5(output_dir, filename, verbose=False)

    # Route to appropriate handler based on available data
    external_cov = observables.get('external_covariance', None)
    if external_cov is not None:
        logger.info("External covariance mode: building data without systematics")
        return _data_array_from_h5_external_cov(observables, external_cov, pseudodata_index, observable_filter)

    correlation_manager_data = observables.get('correlation_manager', None)
    if correlation_manager_data is not None:
        try:
            correlation_manager = SystematicCorrelationManager.from_dict(correlation_manager_data)
            logger.info("Using correlation-aware systematic handling")
            return _data_array_from_h5_with_correlations(
                observables, correlation_manager, pseudodata_index, observable_filter
            )
        except Exception as e:
            logger.warning(f"Failed to deserialize correlation manager: {e}")
    
    return _data_array_from_h5_nosys(output_dir, filename, pseudodata_index, observable_filter)


####################################################################################################################
# DATA LOADING FUNCTIONS - MODE-SPECIFIC IMPLEMENTATIONS
####################################################################################################################

def _data_array_from_h5_nosys(output_dir, filename, pseudodata_index: int = -1, 
                              observable_filter=None):
    '''
    Load data array without systematic correlations (Fallback mode).
    
    Args:
        output_dir: Directory containing observables.h5
        filename: Filename (typically 'observables.h5')
        pseudodata_index: Index for closure test (-1 for experimental data)
        observable_filter: Optional filter
        
    Returns:
        Dict with basic data structure
    '''
    observables = read_dict_from_h5(output_dir, filename, verbose=False)
    sorted_observable_list = sorted_observable_list_from_dict(observables, observable_filter=observable_filter)

    if pseudodata_index < 0:
        data_dict = observables['Data']
    else:
        data_dict = observables['Prediction_validation']
        exp_data_dict = observables['Data']
        
        for observable_label in sorted_observable_list:
            exp_uncertainty = exp_data_dict[observable_label]['y_err_stat']
            prediction_data = data_dict[observable_label]['y']
            
            pseudodata = _generate_pseudodata(prediction_data, exp_uncertainty, 
                                             pseudodata_index, observable_label)
            
            data_dict[observable_label]['y'] = pseudodata
            data_dict[observable_label]['y_err_stat'] = exp_uncertainty

    data = {'y': [], 'y_err_stat': []}
    
    for observable_label in sorted_observable_list:
        data['y'].extend(data_dict[observable_label]['y'])
        data['y_err_stat'].extend(data_dict[observable_label]['y_err_stat'])

    data['y'] = np.array(data['y'])
    data['y_err_stat'] = np.array(data['y_err_stat'])
    
    logger.info(f'Data loading complete (no systematics): {data["y"].shape[0]} features')

    return data


def _data_array_from_h5_external_cov(observables, external_cov, pseudodata_index, 
                                     observable_filter):
    """
    Load data array using external covariance matrix (Expert mode).
    
    Args:
        observables: Observables dictionary from h5 file
        external_cov: External covariance matrix
        pseudodata_index: Index for closure test (-1 for experimental data)
        observable_filter: Optional filter
        
    Returns:
        Data structure with external covariance
    """
    sorted_observable_list = sorted_observable_list_from_dict(observables, observable_filter=observable_filter)
    
    if not sorted_observable_list:
        logger.warning("No observables passed the filter.")
        return {
            'y': np.array([]), 
            'y_err_stat': np.array([]), 
            'external_covariance': external_cov
        }
    
    if pseudodata_index < 0:
        data_dict = observables['Data']
        logger.info("Loading experimental data (external covariance mode)")
    else:
        logger.info(f"Generating pseudodata from validation design point {pseudodata_index}")
        data_dict = observables['Prediction_validation']
        exp_data_dict = observables['Data']
        
        for observable_label in sorted_observable_list:
            exp_uncertainty = exp_data_dict[observable_label]['y_err_stat']
            prediction_data = data_dict[observable_label]['y']
            
            pseudodata = _generate_pseudodata(prediction_data, exp_uncertainty,
                                             pseudodata_index, observable_label)
            
            data_dict[observable_label]['y'] = pseudodata
            data_dict[observable_label]['y_err_stat'] = exp_uncertainty
    
    data = {
        'y': [],
        'y_err_stat': [],
        'y_err_syst': np.array([]).reshape(0, 0),
        'systematic_names': [],
        'observable_ranges': [],
        'external_covariance': external_cov
    }
    
    current_feature_index = 0
    
    for observable_label in sorted_observable_list:
        y_values = data_dict[observable_label]['y']
        n_bins = len(y_values)
        
        start_idx = current_feature_index
        end_idx = current_feature_index + n_bins
        data['observable_ranges'].append((start_idx, end_idx, observable_label))
        
        data['y'].extend(y_values)
        data['y_err_stat'].extend([0.0] * n_bins)
        
        current_feature_index = end_idx
    
    data['y'] = np.array(data['y'])
    data['y_err_stat'] = np.array(data['y_err_stat'])
    
    n_features = len(data['y'])
    if external_cov.shape != (n_features, n_features):
        raise ValueError(
            f"External covariance shape {external_cov.shape} doesn't match "
            f"n_features={n_features} from observables"
        )
    
    logger.info(f"Data loading complete (external covariance mode):")
    logger.info(f"  Features: {n_features}")
    logger.info(f"  Observables: {len(data['observable_ranges'])}")
    
    return data


def _data_array_from_h5_with_correlations(observables, correlation_manager, 
                                          pseudodata_index, observable_filter):
    """
    Load data array with systematic correlation support (Legacy or Advanced mode).
    
    Args:
        observables: Observables dictionary from h5 file
        correlation_manager: SystematicCorrelationManager instance
        pseudodata_index: Index for closure test (-1 for experimental data)
        observable_filter: Optional filter
        
    Returns:
        Data structure with systematic correlations
    """
    sorted_observable_list = sorted_observable_list_from_dict(observables, observable_filter=observable_filter)
    
    if not sorted_observable_list:
        logger.warning("No observables passed the filter.")
        return {
            'y': np.array([]), 
            'y_err_stat': np.array([]), 
            'y_err_syst': np.array([]).reshape(0, 0)
        }

    if pseudodata_index < 0:
        data_dict = observables['Data']
    else:
        data_dict = observables['Prediction_validation']
        exp_data_dict = observables['Data']
        
        for observable_label in sorted_observable_list:
            exp_uncertainty = exp_data_dict[observable_label]['y_err_stat']
            prediction_data = data_dict[observable_label]['y']
            
            pseudodata = _generate_pseudodata(prediction_data, exp_uncertainty,
                                             pseudodata_index, observable_label)
            
            data_dict[observable_label]['y'] = pseudodata
            data_dict[observable_label]['y_err_stat'] = exp_uncertainty
            data_dict[observable_label]['systematics'] = exp_data_dict[observable_label]['systematics']

            if 'external_stat_cov_matrix' in exp_data_dict[observable_label]:
                data_dict[observable_label]['external_stat_cov_matrix'] = exp_data_dict[observable_label]['external_stat_cov_matrix']

    all_systematic_names = correlation_manager.get_all_systematic_names()
    
    data = {
        'y': [],
        'y_err_stat': [],
        'y_err_syst': None,
        'systematic_names': all_systematic_names,
        'observable_ranges': [],
        'correlation_manager': correlation_manager
    }
    
    current_feature_index = 0
    systematic_uncertainty_list = []
    
    for observable_label in sorted_observable_list:
        y_values = _validate_and_flatten_array(data_dict[observable_label]['y'], observable_label)
        y_err_stat_values = _validate_and_flatten_array(
            data_dict[observable_label]['y_err_stat'], 
            f"{observable_label}_stat"
        )

        n_bins = len(y_values)
        start_idx = current_feature_index
        end_idx = current_feature_index + n_bins
        data['observable_ranges'].append((start_idx, end_idx, observable_label))
        
        data['y'].extend(y_values)
        data['y_err_stat'].extend(y_err_stat_values)
        
        obs_systematics = data_dict[observable_label].get('systematics', {})
        expected_systematics = correlation_manager.get_systematic_names_for_observable(observable_label)
        
        obs_syst_matrix = np.zeros((n_bins, len(all_systematic_names)))
        
        for sys_full_name in expected_systematics:
            if ':' in sys_full_name:
                base_sys_name, _ = sys_full_name.split(':', 1)
            else:
                base_sys_name = sys_full_name

            if base_sys_name.startswith('sum_'):
                base_sys_name = 'sum'
            
            if sys_full_name in all_systematic_names:
                sys_idx = all_systematic_names.index(sys_full_name)
                
                if base_sys_name in obs_systematics:
                    obs_syst_matrix[:, sys_idx] = obs_systematics[base_sys_name]
                    logger.debug(f"  Mapped {base_sys_name} → {sys_full_name} for {observable_label}")
                else:
                    logger.warning(f"  Systematic {base_sys_name} not found in data for {observable_label}")
            else:
                logger.warning(f"  Systematic {sys_full_name} not in global list")
        
        systematic_uncertainty_list.append(obs_syst_matrix)
        current_feature_index = end_idx

    data['y'] = np.array(data['y'])
    data['y_err_stat'] = np.array(data['y_err_stat'])
    
    if systematic_uncertainty_list:
        try:
            data['y_err_syst'] = np.vstack(systematic_uncertainty_list)
        except ValueError as e:
            for i, mat in enumerate(systematic_uncertainty_list):
                logger.error(f"  Matrix {i}: {mat.shape}")
            raise ValueError(f"Shape mismatch in systematic uncertainty stacking: {e}")
    else:
        data['y_err_syst'] = np.array([]).reshape(len(data['y']), 0)

    # Collect per-observable external stat covariances
    data['per_observable_external_stat_cov'] = {}
    for observable_label in sorted_observable_list:
        ext_cov = data_dict[observable_label].get('external_stat_cov_matrix', None)
        if ext_cov is not None:
            # Find observable ranges
            for start, end, obs_name in data['observable_ranges']:
                if obs_name == observable_label:
                    data['per_observable_external_stat_cov'][observable_label] = {
                        'matrix': ext_cov,
                        'start': start,
                        'end': end
                    }
                    logger.info(f"Stored external stat cov for {observable_label} (bins {start}-{end})")
                    break
    
    correlation_manager.register_observable_ranges(data['observable_ranges'])
    correlation_manager.resolve_bin_counts(data['observable_ranges'])
    
    logger.info(f"Data loading complete:")
    logger.info(f"  Features: {data['y'].shape[0]}")
    logger.info(f"  Systematic uncertainties: {data['y_err_syst'].shape[1]} sources")
    logger.info(f"  Observables: {len(data['observable_ranges'])}")
    
    warnings = correlation_manager.validate_configuration()
    for warning in warnings:
        logger.warning(f"  Correlation validation: {warning}")
    
    return data


####################################################################################################################
# MAIN INITIALIZATION
####################################################################################################################


def initialize_observables_dict_from_tables(
    table_dir: str | Path,
    analysis_config: dict[str, Any],
    parameterization: str,
    correlation_groups: dict[str, str] | None = None
) -> dict[str, Any]:
    '''
    Initialize observables dictionary from .dat files with systematic uncertainty support.
    CORE FUNCTIONALITY:
    ==================
    Initialize from .dat files into a dictionary of numpy arrays:
      - Loop through all observables in the table directory for the given model and parameterization
      - Include only those observables that:
         - Have sqrts, centrality specified in the analysis_config
         - Whose filename contains a string from analysis_config observable_list
      - Apply optional cuts to the x-range of the predictions and data (e.g. pt_hadron>10 GeV)
      - Separate out the design/predictions with indices in the validation set
      - Parse and store systematic uncertainties with correlation information
    
    Note: All data points are the ratio of AA/pp
    
    :param str table_dir: directory where tables are located
    :param dict analysis_config: dictionary of analysis configuration
    :param str parameterization: name of qhat parameterization
    
    :return: Dictionary with the following enhanced structure:
    :rtype: dict
    
    RETURN STRUCTURE:
    ================
    observables['Data'][observable_label]['y'] -- measurement values
                                         ['y_err_stat'] -- statistical uncertainties (renamed from 'y_err')
                                         ['systematics']['jec'] -- JEC systematic uncertainties 
                                         ['systematics']['taa'] -- TAA systematic uncertainties 
                                         ['systematics'][...] -- other systematic uncertainties 
                                         ['xmin'] -- bin lower edge (used for plotting)
                                         ['xmax'] -- bin upper edge (used for plotting)
    
    observables['Prediction'][observable_label]['y'] -- theory prediction values
                                               ['y_err_stat'] -- statistical uncertainties (renamed from 'y_err')
                                               ['systematics'] -- systematic uncertainties dict 
                                               ['xmin'] -- bin lower edge (copied from Data)
                                               ['xmax'] -- bin upper edge (copied from Data)
    
    observables['Prediction_validation'][observable_label] -- same structure as Prediction
    
    observables['Design'][parameterization] -- design points for given parameterization
    observables['Design_indices'][parameterization] -- indices of design points included
    observables['Design_validation'][parameterization] -- design points for validation set  
    observables['Design_indices_validation'][parameterization] -- indices of validation design points
    
    observables['correlation_manager'] -- SystematicCorrelationManager instance (NEW)
                                        -- Contains correlation structure from config parsing
                                        -- Used for correlation-aware covariance calculations
    
    OBSERVABLE LABEL CONVENTION:
    ===========================
    observable_label = f'{sqrts}__{system}__{observable_type}__{observable}__{subobservable}__{centrality}'
    
    Example: '5020__PbPb__hadron__pt_ch_cms____0-5'
    
    CONFIGURATION FORMATS:
    =====================
    OLD FORMAT (still supported):
    observable_list: ['5020__PbPb__hadron__pt_ch_cms____0-5']
    
    NEW FORMAT (with systematic correlations):
    observable_list:
      - observable: '5020__PbPb__hadron__pt_ch_cms____0-5'
        sys_data: ['jec:cms', 'taa:5020']  # correlation tags define systematic correlations
        sys_theory: []                     # theory systematics (future feature)
    
    SYSTEMATIC CORRELATION TAGS:
    ===========================
    - 'jec:cms' -- JEC systematic correlated within CMS measurements only
    - 'jec:alice' -- JEC systematic correlated within ALICE measurements only  
    - 'taa:5020' -- TAA systematic correlated across all 5.02 TeV measurements
    - 'lumi:uncor' -- Luminosity systematic uncorrelated (diagonal)
    - Custom tags supported: 'group1', 'experiment_a', etc.
    
    NOTE: Correlation tags are only in config files, not in .dat files
    NOTE: Base systematic names in .dat files (s_jec, s_taa) remain unchanged
    
    DESIGN NOTES:
    ============
    - The "Design" key contains actual parameters, "Design_indices" contains design point indices
    - As of August 2023, the "Design" key doesn't pass around the parameterization
    - Systematic uncertainties are stored as separate columns, not combined into total uncertainty
    - Empty systematics dict maintained for observables without systematic uncertainties (backward compatibility)
    '''
    logger.info('Including the following observables:')

    # We will construct a dict containing all observables
    observables = _recursive_defaultdict()

    # We separate out the validation indices specified in the config
    validation_range = analysis_config['validation_indices']
    validation_indices = range(validation_range[0], validation_range[1])

    #----------------------
    # Read experimental data
    data_dir = os.path.join(table_dir, 'Data')

    parsed_observables, correlation_manager, external_cov_file = _parse_config_observables(
                                            analysis_config,
                                            correlation_groups=correlation_groups
    )

    # Build config map including external_stat_cov
    systematic_config_map = {}
    for obs_name, sys_data_list, sys_theory_list, external_stat_cov in parsed_observables:
        systematic_config_map[obs_name] = (sys_data_list, sys_theory_list, external_stat_cov)

    def _lookup_systematic_config(file_obs_label):
        """Find the YAML observable entry that matches a file-derived label.

        The YAML uses short labels like '5020__PbPb__hadron__pt_ch_cms', while
        the loader produces file-derived labels with centrality suffixes such as
        '5020__PbPb__hadron__pt_ch_cms____0-5'. Match by substring containment
        (same convention as ObservableFilter.accept_observable). When multiple
        YAML keys are contained in the file label, prefer the longest one.
        """
        match = None
        for cfg_key in systematic_config_map:
            if cfg_key in file_obs_label and (match is None or len(cfg_key) > len(match)):
                match = cfg_key
        return systematic_config_map[match] if match is not None else ([], [], None)

    if external_cov_file:
        external_cov_path = os.path.join(table_dir, external_cov_file)
        external_cov = _read_external_covariance(external_cov_path)
        
        if external_cov is not None:
            observables['external_covariance'] = external_cov
            logger.info(f"Loaded external covariance: shape {external_cov.shape}")
        else:
            raise ValueError(f"Failed to load external covariance from {external_cov_path}")

    if correlation_manager.get_all_systematic_names():

        logger.info(f"Adding correlation manager with {len(correlation_manager.get_all_systematic_names())} systematics")
        observables['correlation_manager'] = correlation_manager.to_dict()
    else:
        logger.info("No systematic correlations found in config")

    for filename in os.listdir(data_dir):
        # Skip files that don't match the expected Data file pattern
        if not filename.startswith('Data__'):
            logger.debug(f"Skipping non-data file in Data directory: {filename}")
            continue

        if _accept_observable(analysis_config, filename):

            filepath = os.path.join(data_dir, filename)
            data = np.loadtxt(filepath, ndmin=2)
            data_entry = {}
            data_entry['xmin'] = data[:,0]
            data_entry['xmax'] = data[:,1]
            data_entry['y'] = data[:,2]

            # Parse header to discover stat / systematic columns. Supports both
            # symmetric columns (e.g. 'y_err_stat', 'sys_jec') and asymmetric
            # pairs (e.g. 'stat,low'+'stat,high'); pairs are reduced by max.
            grouped_columns = _parse_data_header(filepath)
            all_columns = _read_grouped_columns(filepath, grouped_columns)

            y_err_stat = None
            for alias in _STAT_COLUMN_ALIASES:
                if alias in all_columns:
                    y_err_stat = all_columns[alias]
                    break
            if y_err_stat is None:
                # Header parsing failed or no recognized stat column -- fall back
                # to the legacy convention of column index 3.
                y_err_stat = data[:, 3]
            data_entry['y_err_stat'] = y_err_stat

            systematic_data = {k: v for k, v in all_columns.items()
                               if k not in _STAT_COLUMN_ALIASES}

            observable_label, _ = _filename_to_labels(filename)

            sys_data_list, sys_theory_list, external_stat_cov_file = _lookup_systematic_config(observable_label)

            # Check if this observable has per-observable external stat cov
            external_stat_cov_matrix = None
            if external_stat_cov_file is not None and external_cov_file is None:
                # Only load if global external cov is NOT set
                n_bins = len(data_entry['y'])

                # Construct full path (resolve relative to table_dir)
                if not os.path.isabs(external_stat_cov_file):
                    external_stat_cov_full_path = os.path.join(table_dir, external_stat_cov_file)
                else:
                    external_stat_cov_full_path = external_stat_cov_file

                try:
                    external_stat_cov_matrix = _read_external_stat_covariance(
                        external_stat_cov_full_path, n_bins, observable_label
                    )
                except Exception as e:
                    logger.error(f"Failed to load external_stat_cov for {observable_label}: {e}")
                    external_stat_cov_matrix = None

            # Store in data_entry
            data_entry['external_stat_cov_file'] = external_stat_cov_file
            data_entry['external_stat_cov_matrix'] = external_stat_cov_matrix

            if sys_data_list:
                # Handle 'sum' configurations - check if this observable wants summed systematics
                for sys_config in sys_data_list:
                    if sys_config.startswith('sum'):
                        # This observable wants summed systematics
                        logger.info(f"Observable '{observable_label}' requests summed systematics")

                        if systematic_data:
                            # Sum all available systematics in quadrature
                            summed_sys = _sum_systematics_quadrature(systematic_data)

                            # Replace individual systematics with single summed one
                            logger.info(f"  Replaced {len(systematic_data)} individual systematics with 1 summed systematic")
                            systematic_data = {'sum': summed_sys}
                        else:
                            logger.warning(f"  No systematic columns found to sum for '{observable_label}'")
                            # Create empty sum systematic to maintain structure
                            systematic_data = {}

                        # Only one 'sum' directive should exist per observable
                        break

                filtered_systematics = _filter_systematics_by_config(systematic_data, sys_data_list)
                data_entry['systematics'] = filtered_systematics
            else:
                data_entry['systematics'] = {}
            
            observables['Data'][observable_label] = data_entry

            # ORIGINAL: Validation check
            if 0 in data_entry['y']:
                msg = f'{filename} has value=0'
                raise ValueError(msg)

    #----------------------
    # Read design points
    design_points_to_exclude = analysis_config.get('design_points_to_exclude', [])
    design_dir = os.path.join(table_dir, 'Design')
    for filename in os.listdir(design_dir):

        if _filename_to_labels(filename)[1] == parameterization:
            # Explanation of variables:
            #  - design_point_parameters: The parameters of the design points, with one per design point.
            #  - design_points: List of the design points. If everything is filled out, this is just the trivial range(n_design_points).
            #                   However, we have to handle this carefully because sometimes they are missing.

            # Shape: (n_design_points, n_parameters)
            design_point_parameters = np.loadtxt(os.path.join(design_dir, filename), ndmin=2)

            # Separate training and validation sets into separate dicts
            design_points = _read_design_points_from_design_dat(table_dir, parameterization)
            training_indices, training_design_points, validation_indices, validation_design_points = _split_training_validation_indices(
                design_points=design_points,
                validation_indices=validation_indices,
                design_points_to_exclude=design_points_to_exclude,
            )

            observables['Design'] = design_point_parameters[training_indices]
            observables['Design_indices'] = training_design_points
            observables['Design_validation'] = design_point_parameters[validation_indices]
            observables['Design_indices_validation'] = validation_design_points

    #----------------------
    # Read predictions and uncertainty
    prediction_dir = os.path.join(table_dir, 'Prediction')
    for filename in os.listdir(prediction_dir):

        if 'values' in filename and parameterization in filename:
            if _accept_observable(analysis_config, filename):

                filename_prediction_values = filename
                filename_prediction_errors = filename.replace('values', 'errors')
                observable_label, _ = _filename_to_labels(filename_prediction_values)

                prediction_values = np.loadtxt(os.path.join(prediction_dir, filename_prediction_values), ndmin=2)
                prediction_errors = np.loadtxt(os.path.join(prediction_dir, filename_prediction_errors), ndmin=2)

                # Check that the observable is in the data dict
                if observable_label not in observables['Data']:
                    data_keys = observables['Data'].keys()
                    msg = f'{observable_label} not found in observables[Data]: {data_keys}'
                    raise ValueError(msg)

                # Check that data and prediction have the same size.
                # If the published data file covers more bins than the simulation predicts
                # (typical when HEPData provides a wider pT range than the model can cover),
                # drop the lowest-pT data bins to align. The convention is HEPData-sorted
                # bins (ascending pT) and the prediction covering the high-pT tail.
                data_size = observables['Data'][observable_label]['y'].shape[0]
                prediction_size = prediction_values.shape[0]
                if data_size > prediction_size:
                    n_drop = data_size - prediction_size
                    obs_data = observables['Data'][observable_label]
                    kept_xmin = obs_data['xmin'][n_drop] if obs_data['xmin'].size > n_drop else None
                    kept_xmax = obs_data['xmax'][-1] if obs_data['xmax'].size else None
                    logger.info(
                        f"Auto-aligning '{observable_label}': data has {data_size} bins, "
                        f"prediction has {prediction_size}. Dropping the lowest-pT "
                        f"{n_drop} data bin(s) to match (kept x range [{kept_xmin}, {kept_xmax}])."
                    )
                    for key, val in list(obs_data.items()):
                        if hasattr(val, 'shape') and val.shape and val.shape[0] == data_size:
                            obs_data[key] = val[n_drop:] if val.ndim == 1 else val[n_drop:, ...]
                        elif isinstance(val, dict):
                            # nested dicts (e.g. 'systematics'): trim per-systematic arrays
                            for sk, sv in val.items():
                                if hasattr(sv, 'shape') and sv.shape and sv.shape[0] == data_size:
                                    val[sk] = sv[n_drop:]
                    data_size = prediction_size

                if data_size != prediction_size:
                    msg = f'({filename_prediction_values}) has different shape ({prediction_size}) than Data ({data_size}) -- before cuts.'
                    raise ValueError(msg)

                # Apply cuts to the prediction values and errors (as well as data dict)
                # We do this by construct a mask of bins (rows) to keep
                cuts = analysis_config.get('cuts', {})
                for obs_key, cut_range in cuts.items():
                    if obs_key in observable_label:
                        x_min, x_max = cut_range
                        obs_data = observables['Data'][observable_label]
                        mask = (x_min <= obs_data['xmin']) & (obs_data['xmax'] <= x_max)
                        prediction_values = prediction_values[mask, :]
                        prediction_errors = prediction_errors[mask, :]
                        n_data = obs_data['y'].shape[0]
                        for key, val in list(obs_data.items()):
                            if hasattr(val, 'shape') and val.shape and val.shape[0] == n_data:
                                obs_data[key] = val[mask] if val.ndim == 1 else val[mask, ...]
                            elif isinstance(val, dict):
                                # nested dicts (e.g. 'systematics'): mask per-systematic arrays
                                for sk, sv in val.items():
                                    if hasattr(sv, 'shape') and sv.shape and sv.shape[0] == n_data:
                                        val[sk] = sv[mask]

                # Check that data and prediction have the same size
                data_size = observables['Data'][observable_label]['y'].shape[0]
                prediction_size = prediction_values.shape[0]
                if data_size != prediction_size:
                    msg = f'({filename_prediction_values}) has different shape ({prediction_size}) than Data ({data_size}) -- after cuts.'
                    raise ValueError(msg)

                # Separate training and validation sets into separate dicts
                design_points = _read_design_points_from_predictions_dat(
                    prediction_dir=prediction_dir,
                    filename_prediction_values=filename_prediction_values,
                )
                training_indices, _, validation_indices, _ = _split_training_validation_indices(
                    design_points=design_points,
                    validation_indices=validation_indices,
                    design_points_to_exclude=design_points_to_exclude,
                )

                _, sys_theory_list, _ = _lookup_systematic_config(observable_label)
                model_name = analysis_config.get('model_name', 'exponential')
                theory_systematics = _read_theory_systematics(table_dir, model_name, observable_label, sys_theory_list)
                filtered_theory_systematics = _filter_systematics_by_config(theory_systematics, sys_theory_list)

                if cuts:
                    for obs_key, cut_range in cuts.items():
                        if obs_key in observable_label:
                            x_min, x_max = cut_range
                            mask = (x_min <= observables['Data'][observable_label]['xmin']) & (observables['Data'][observable_label]['xmax'] <= x_max)
                            for sys_name, sys_data in filtered_theory_systematics.items():
                                filtered_theory_systematics[sys_name] = sys_data[mask, :]

                # MODIFIED: Store predictions with systematic support
                observables['Prediction'][observable_label] = {
                    'xmin': observables['Data'][observable_label]['xmin'], 
                    'xmax': observables['Data'][observable_label]['xmax'], 
                    'y': np.take(prediction_values, training_indices, axis=1),
                    'y_err_stat': np.take(prediction_errors, training_indices, axis=1), 
                    'systematics': {sys_name: np.take(sys_data, training_indices, axis=1) 
                                   for sys_name, sys_data in filtered_theory_systematics.items()}  
                }

                observables['Prediction_validation'][observable_label] = {
                    'xmin': observables['Data'][observable_label]['xmin'],  
                    'xmax': observables['Data'][observable_label]['xmax'],   
                    'y': np.take(prediction_values, validation_indices, axis=1),
                    'y_err_stat': np.take(prediction_errors, validation_indices, axis=1),  
                    'systematics': {sys_name: np.take(sys_data, validation_indices, axis=1) 
                                   for sys_name, sys_data in filtered_theory_systematics.items()}  
                }

                # TODO: Do something about bins that have value=0?
                if 0 in prediction_values:
                    logger.warning(f'{filename_prediction_values} has value=0 at design points {np.where(prediction_values == 0)[1]}')

                # If no bins left, remove the observable
                if not np.any(observables['Prediction'][observable_label]['y']):
                    del observables['Prediction'][observable_label]
                    del observables['Prediction_validation'][observable_label]
                    del observables['Data'][observable_label]
                    logging.info(f'  Note: Removing {observable_label} from observables dict because no bins left after cuts')

    #----------------------
    # Print observables that we will use
    # NOTE: We don't need to pass the observable filter because we already filtered the observables via `_accept_observables``
    [logger.info(f'Accepted observable {s}') for s in sorted_observable_list_from_dict(observables['Prediction'])]

    return observables

####################################################################################################################
# HDF5 I/O
####################################################################################################################

def write_dict_to_h5(results, output_dir, filename, verbose=True):
    '''
    Write nested dictionary of ndarray to hdf5 file
    Note: all keys should be strings

    :param dict results: (nested) dictionary to write
    :param str output_dir: directory to write to
    :param str filename: name of hdf5 file to create (will overwrite)
    '''
    if verbose:
        logger.info("")
        logger.info(f'Writing results to {output_dir}/{filename}...')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dicttoh5(results, os.path.join(output_dir, filename), update_mode="modify")

    if verbose:
        logger.info('Done.')
        logger.info("")

def read_dict_from_h5(input_dir, filename, verbose=True):
    '''
    Read dictionary of ndarrays from hdf5
    Note: all keys should be strings

    :param str input_dir: directory from which to read data
    :param str filename: name of hdf5 file to read
    '''
    if verbose:
        logger.info("")
        logger.info(f'Loading results from {input_dir}/{filename}...')

    results = h5todict(os.path.join(input_dir, filename))

    if verbose:
        logger.info('Done.')
        logger.info("")

    return results


####################################################################################################################
# MATRIX OPERATIONS
####################################################################################################################

def predictions_matrix_from_h5(output_dir, filename, validation_set=False, observable_filter: ObservableFilter | None = None):
    '''
    Initialize predictions from observables.h5 file into a single 2D array:

    :param str output_dir: location of filename
    :param str filename: h5 filename (typically 'observables.h5')
    :param ObservableFilter observable_filter: (optional) filter to apply to the observables
    :return 2darray Y: matrix of predictions at all design points (design_point_index, observable_bins) i.e. (n_samples, n_features)
    '''

    # Initialize observables dict from observables.h5 file
    observables = read_dict_from_h5(output_dir, filename, verbose=False)

    # Sort observables, to keep well-defined ordering in matrix
    sorted_observable_list = sorted_observable_list_from_dict(observables, observable_filter=observable_filter)

    # Set dictionary key
    if validation_set:
        prediction_label = 'Prediction_validation'
    else:
        prediction_label = 'Prediction'

    # Loop through sorted observables and concatenate them into a single 2D array:
    #   (design_point_index, observable_bins) i.e. (n_samples, n_features)
    length_of_Y = 0
    for i,observable_label in enumerate(sorted_observable_list):
        values = observables[prediction_label][observable_label]['y'].T
        length_of_Y += values.shape[1]
        logger.info(f"{observable_label} shape: {values.shape}, length: {length_of_Y=}")
        if i==0:
            Y = values
        else:
            Y = np.concatenate([Y,values], axis=1)
    if length_of_Y == 0:
        raise ValueError(f"No observables found in the prediction file for {observable_filter}")
    logger.info(f'  Total shape of {prediction_label} data (n_samples, n_features): {Y.shape}')

    return Y

####################################################################################################################
def design_array_from_h5(output_dir, filename, validation_set=False):
    '''
    Initialize design array from observables.h5 file

    :param str output_dir: location of filename
    :param str filename: h5 filename (typically 'observables.h5')
    :return 2darray design: array of design points
    '''

    # Initialize observables dict from observables.h5 file
    observables = read_dict_from_h5(output_dir, filename, verbose=False)
    if validation_set:
        design = observables['Design_validation']
    else:
        design = observables['Design']
    return design

####################################################################################################################
def data_dict_from_h5(output_dir, filename, observable_table_dir=None):
    '''
    Initialize data dict from observables.h5 file

    :param str output_dir: location of filename
    :param str filename: h5 filename (typically 'observables.h5')
    :return dict data: dict of arrays of data points (columns of data[observable_label]: xmin xmax y y_err_stat)
    '''

    # Initialize observables dict from observables.h5 file
    observables = read_dict_from_h5(output_dir, filename, verbose=False)
    data = observables['Data']

    # Check that data matches original table (if observable_table_dir is specified)
    if observable_table_dir:
        data_table_dir = os.path.join(observable_table_dir, 'Data')
        for observable_label in observables['Data'].keys():
            data_table_filename = f'Data__{observable_label}.dat'
            data_table = np.loadtxt(os.path.join(data_table_dir, data_table_filename), ndmin=2)
            assert np.allclose(data[observable_label]['xmin'], data_table[:,0])
            assert np.allclose(data[observable_label]['xmax'], data_table[:,1])
            assert np.allclose(data[observable_label]['y'], data_table[:,2])
            assert np.allclose(data[observable_label]['y_err_stat'] , data_table[:,3])

    return data

####################################################################################################################
def observable_dict_from_matrix(Y, observables, cov=np.array([]), config=None, validation_set=False, observable_filter: ObservableFilter | None = None):
    '''
    Translate matrix of stacked observables to a dict of matrices per observable

    :param ndarray Y: 2D array: (n_samples, n_features)
    :param dict observables: dict
    :param ndarray cov: covariance matrix (n_samples, n_features, n_features)
    :param config EmulatorConfig: config object
    :param bool validation_set: (optional, only needed to check against table values)
    :param ObservableFilter observable_filter: (optional) filter to apply to the observables
    :return dict[ndarray] Y_dict: dict with ndarray for each observable
    '''

    Y_dict: dict[str, dict[str, npt.NDArray]] = {}
    Y_dict['central_value'] = {}
    if cov.any():
        Y_dict['cov'] = {}

    if validation_set:
        prediction_key = 'Prediction_validation'
    else:
        prediction_key = 'Prediction'

    # Loop through sorted list of observables and populate predictions into Y_dict
    # Also store variances (ignore off-diagonal terms here, for plotting purposes)
    #   (Note that in general there will be significant covariances between observables, induced by the PCA)
    sorted_observable_list = sorted_observable_list_from_dict(observables, observable_filter=observable_filter)
    current_bin = 0
    for observable_label in sorted_observable_list:
        n_bins = observables[prediction_key][observable_label]['y'].shape[0]
        Y_dict['central_value'][observable_label] = Y[:,current_bin:current_bin+n_bins]

        if cov.any():
            Y_dict['cov'][observable_label] = cov[:,current_bin:current_bin+n_bins,current_bin:current_bin+n_bins]
            assert Y_dict['central_value'][observable_label].shape == Y_dict['cov'][observable_label].shape[:-1]

        current_bin += n_bins

    # Check that the total number of bins is correct
    assert current_bin == Y.shape[1], f"{current_bin=}, {Y.shape[1]=}"

    # Check that prediction matches original table (if observable_table_dir, parameterization, validation_indices are specified)
    # If validation_set, select the validation indices; otherwise, select the training indices
    # NOTE: We cannot do this crosscheck if we've applied preprocessing because the prediction
    #       values may vary from the tables themselves (eg. due to smoothing).
    #       Similarly, if we have applied cuts to the x-range we cannot do the check.
    if config and "preprocessed" not in config.observables_filename and 'cuts' not in config.analysis_config:

        validation_range = config.analysis_config['validation_indices']
        validation_indices = list(range(validation_range[0], validation_range[1]))
        design_points = _read_design_points_from_design_dat(
            observable_table_dir=config.observable_table_dir,
            parameterization=config.parameterization,
        )
        training_indices_numpy, _, validation_indices_numpy, _ = _split_training_validation_indices(
            design_points=design_points,
            validation_indices=validation_indices,
            design_points_to_exclude=config.analysis_config.get('design_points_to_exclude', []),
        )
        if validation_set:
            indices_numpy = validation_indices_numpy
        else:
            indices_numpy = training_indices_numpy

        prediction_table_dir = os.path.join(config.observable_table_dir, 'Prediction')
        for observable_label in sorted_observable_list:
            prediction_table_filename = f'Prediction__{config.parameterization}__{observable_label}__values.dat'
            prediction_table = np.loadtxt(os.path.join(prediction_table_dir, prediction_table_filename), ndmin=2)
            prediction_table_selected = np.take(prediction_table, indices_numpy, axis=1).T
            assert np.allclose(Y_dict['central_value'][observable_label], prediction_table_selected), \
                               f"{observable_label} (design point 0) \n prediction: {Y_dict['central_value'][observable_label][0,:]} \n prediction (table): {prediction_table_selected[0,:]}"

    return Y_dict

####################################################################################################################
def observable_matrix_from_dict(Y_dict: dict[str, dict[str, npt.NDArray[np.float64]]], values_to_return: str = "central_value") -> npt.NDArray[np.float64]:
    """
    Translate dict of matrixes per observable to a matrix of stacked observables

    The observable keys should already be ordered, so we're free to trivially concatenate them

    :param dict[str, ndarray] Y_dict: dict with ndarray for each observable
    :param str values_to_return: (optional) which values to return. Default: "central_value"
    :return ndarray: 2D array: (n_samples, n_features)
    """
    matrix: npt.NDArray[np.float64] | None = None
    for observable_values in Y_dict[values_to_return].values():
        if matrix is None:
            matrix = np.array(observable_values, copy=True)
        else:
            matrix = np.concatenate([matrix, observable_values], axis=1)

    # Help out typing
    assert matrix is not None

    return matrix


####################################################################################################################
# UTILITIES
####################################################################################################################

def observable_label_to_keys(observable_label):
    '''
    Parse filename into individual keys

    :param str observable_label: observable label
    :return list of subobservables
    :rtype list
    '''

    observable_keys = observable_label.split('__')

    sqrts = observable_keys[0]
    system = observable_keys[1]
    observable_type = observable_keys[2]
    observable = observable_keys[3]
    subobserable = observable_keys[4]
    centrality = observable_keys[5]
    return sqrts, system, observable_type, observable, subobserable, centrality

def sorted_observable_list_from_dict(observables, observable_filter: ObservableFilter | None = None):
    '''
    Define a sorted list of observable_labels from the keys of the observables dict, to keep well-defined ordering in matrix

    :param dict observables: dictionary containing predictions/design/data (or any other dict with observable_labels as keys)
    :param ObservableFilter observable_filter: (optional) filter to apply to the observables
    :return list[str] sorted_observable_list: list of observable labels
    '''
    observable_keys = list(observables.keys())
    if 'Prediction' in observables.keys():
        observable_keys = list(observables['Prediction'].keys())

    # The correlation manager and other metadata should not be treated as observables
    special_keys = ['correlation_manager', 'Design', 'Design_validation', 'Prediction', 'Prediction_validation', 'Data']
    observable_keys = [k for k in observable_keys if k not in special_keys]

    if observable_filter is not None:
        # Filter the observables based on the provided filter
        observable_keys = [
            k for k in observable_keys if observable_filter.accept_observable(observable_name=k)
        ]

    # Sort observables, to keep well-defined ordering in matrix
    return _sort_observable_labels(observable_keys)

def _sort_observable_labels(unordered_observable_labels):
    '''
    Sort list of observable keys by observable_type, observable, subobservable, centrality, sqrts.
    TODO: Instead of a fixed sorting, we may want to allow the user to specify list of sort
          criteria to apply, e.g. list of regex to iteratively sort by.

    :param list[str] observable_labels: unordered list of observable_label keys
    :return list[str] sorted_observable_labels: sorted observable_labels
    '''

    # First, sort the observable_labels to ensure an unambiguous ordering
    ordered_observable_labels = sorted(unordered_observable_labels)

    # Get the individual keys from the observable_label
    x = [observable_label_to_keys(observable_label) for observable_label in ordered_observable_labels]

    # Sort by (in order): observable_type, observable, subobservable, centrality, sqrts
    sorted_observable_label_tuples = sorted(x, key=itemgetter(2,3,4,5,0))

    # Reconstruct the observable_key
    sorted_observable_labels = ['__'.join(x) for x in sorted_observable_label_tuples]

    return sorted_observable_labels

def _filename_to_labels(filename):
    '''
    Parse filename to return observable_label, parameterization

    :param str filename: filename to parse
    :return list of subobservables and parameterization
    :rtype (list, str)
    '''

    # Remove file suffix
    filename_keys = filename[:-4].split('__')

    # Get table type and return observable_label, parameterization
    data_type = filename_keys[0]

    if data_type == 'Data':

        observable_label = '__'.join(filename_keys[1:])
        parameterization = None

    elif data_type == 'Design':

        observable_label = None
        parameterization = filename_keys[1]

    elif data_type == 'Prediction':

        parameterization = filename_keys[1]
        observable_label = '__'.join(filename_keys[2:-1])

    return observable_label, parameterization

@attrs.define
class ObservableFilter:
    include_list: list[str]
    exclude_list: list[str] = attrs.field(factory=list)

    def accept_observable(self, observable_name: str) -> bool:
        """Accept observable from the provided list(s)

        :param str observable_name: Name of the observable to possibly accept.
        :return: bool True if the observable should be accepted.
        """
        # Select observables based on the input list, with the possibility of excluding some
        # observables with additional selection strings (eg. remove one experiment from the
        # observables for an exploratory analysis).
        observable_in_include_list_no_glob = any([observable_string in observable_name for observable_string in self.include_list])
        observable_in_exclude_list_no_glob = any([exclude in observable_name for exclude in self.exclude_list])
        # NOTE: We don't actually care about the name - just that it matches
        observable_in_include_list_glob = any(
            # NOTE: We add "*" around the observable because we have to match against the full string (especially given file extensions), and if we add
            #       them to existing strings, it won't disrupt it.
            [len(fnmatch.filter([observable_name], f"*{observable_string}*")) > 0 for observable_string in self.include_list if "*" in observable_string]
        )
        observable_in_exclude_list_glob = any(
            # NOTE: We add "*" around the observable because we have to match against the full string (especially given file extensions), and if we add
            #       them to existing strings, it won't disrupt it.
            [len(fnmatch.filter([observable_name], f"*{observable_string}*")) > 0 for observable_string in self.exclude_list if "*" in observable_string]
        )

        found_observable = (
            (observable_in_include_list_no_glob or observable_in_include_list_glob)
            and not
            (observable_in_exclude_list_no_glob or observable_in_exclude_list_glob)
        )

        # Helpful for cross checking when debugging
        if observable_in_exclude_list_no_glob or observable_in_exclude_list_glob:
            logger.debug(
                f"Excluding observable '{observable_name}' due to exclude list. {found_observable=},"
                f" {observable_in_include_list_no_glob=}, {observable_in_include_list_glob=}, {observable_in_exclude_list_no_glob=}, {observable_in_exclude_list_glob=}"
            )

        return found_observable

def _accept_observable(analysis_config, filename):
    '''
    Check if observable should be included in the analysis.
    MODIFIED: Handle new config format with systematic specifications
    It must:
      - Have sqrts,centrality specified in the analysis_config
      - Have a filename that contains a string from analysis_config observable_list

    :param dict analysis_config: dictionary of analysis configuration
    :param str filename: filename of table for the considered observable
    '''

    observable_label, _ = _filename_to_labels(filename)

    sqrts, _, _, _, _, centrality = observable_label_to_keys(observable_label)

    # Check sqrts
    if int(sqrts) not in analysis_config['sqrts_list']:
        return False

    # Check centrality
    centrality_min, centrality_max = centrality.split('-')
    
    # Validation
    # Provided a single centrality range - convert to a list of ranges
    centrality_ranges = analysis_config['centrality_range']
    if not isinstance(centrality_ranges[0], list):
        centrality_ranges = [list(centrality_ranges)]

    accepted_centrality = False
    for (selected_cent_min, selected_cent_max) in centrality_ranges:
        if float(centrality_min) >= selected_cent_min:
            if float(centrality_max) <= selected_cent_max:
                accepted_centrality = True
                # Bail out - no need to keep looping if it's already accepted
                break
    if not accepted_centrality:
        return False

    # Check observable - MODIFIED to handle new config format
    # Select observables based on the input list, with the possibility of excluding some
    # observables with additional selection strings (eg. remove one experiment from the
    # observables for an exploratory analysis).
    # NOTE: This is equivalent to EmulationConfig.observable_filter
    accept_observable = False
    global_observable_exclude_list = analysis_config.get("global_observable_exclude_list", [])
    
    for emulation_group_settings in analysis_config["parameters"]["emulators"].values():
        
        # Extract observable names from both old and new formats
        observable_list = emulation_group_settings['observable_list']
        include_list = []
        
        for obs_item in observable_list:
            if isinstance(obs_item, str):
                # Old format: just the observable name
                include_list.append(obs_item)
            elif isinstance(obs_item, dict) and 'observable' in obs_item:
                # New format: extract observable name from dict
                obs_name = obs_item['observable']
                include_list.append(obs_name)
        
        # Verify include_list contains only strings (safety check)
        for item in include_list:
            if not isinstance(item, str):
                raise ValueError(f"include_list must contain only strings, got: {type(item)} - {item}")
        
        observable_filter = ObservableFilter(
            include_list=include_list,  # Now properly extracted as strings
            exclude_list=emulation_group_settings.get("observable_exclude_list", []) + global_observable_exclude_list,
        )
        
        accept_observable = observable_filter.accept_observable(
            observable_name=filename,
        )
        # If it's accepted, return immediately
        if accept_observable:
            return accept_observable

    return accept_observable


####################################################################################################################
# DESIGN POINT HANDLING
####################################################################################################################

def _read_design_points_from_design_dat(
    observable_table_dir: Path | str,
    parameterization: str,
) -> npt.NDArray[np.int32]:
    """Retrieve the design points from the header of the design.dat file

    :param str observable_table_dir: location of table dir
    :param str parameterization: qhat parameterization type
    :return ndarray: design points in their original order in the file
    """
    # Get training set or validation set
    design_table_dir = os.path.join(observable_table_dir, 'Design')
    design_filename = f'Design__{parameterization}.dat'
    with open(os.path.join(design_table_dir, design_filename)) as f:
        for line in f.readlines():
            if 'Design point indices' in line:
                # dtype doesn't really matter here - it's not a limiting factor, so just take int32 as a default
                design_points = np.array(
                    [int(s) for s in line.split(':')[1].split()], dtype=np.int32
                )
                break

    # Validation
    assert len(design_points) == len(set(design_points)), "Design points are not unique! Check on the input file"

    return design_points


#---------------------------------------------------------------
def _read_design_points_from_predictions_dat(
    prediction_dir: Path | str,
    filename_prediction_values: str,
) -> npt.NDArray[np.int32]:
    """ Read design points from the header of a predictions *.dat file

    :param str prediction_dir: location of prediction dir
    :param str filename_prediction_values: name of the prediction values file
    :return ndarray: design points in their original order in the file
    """
    prediction_dir = Path(prediction_dir)
    len_design_point_label_str = len("design_point")
    with open(os.path.join(prediction_dir, filename_prediction_values)) as f:
        for line in f.readlines():
            if 'design_point' in line:
                # dtype doesn't really matter here - it's not a limiting factor, so just take int32 as a default
                # NOTE: This strips out the leading "design_point" text to extract the design point index
                design_points = np.array(
                    [int(s[len_design_point_label_str:]) for s in line.split('#')[1].split()], dtype=np.int32
                )
                break

    # Validation
    assert len(design_points) == len(set(design_points)), "Design points are not unique! Check on the input file"

    return design_points


#---------------------------------------------------------------
def _filter_design_points(
    indices: npt.NDArray[np.int64],
    design_points: npt.NDArray[np.int32],
    design_points_to_exclude: list[int],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]]:
    """ Filter design point indices (and design points themselves).

    :param ndarray indices: indices of the design points themselves to filter
    :param ndarray design_points: design points in their original order in the file
    :param list[int] design_points_to_exclude: list of design point indices to exclude (in the original design point)
    :return tuple[ndarray, ndarray]: filtered indices and design points
    """
    # Determine filter
    points_to_keep = np.isin(design_points, design_points_to_exclude, invert=True)
    # And apply
    indices = indices[points_to_keep]
    design_points = design_points[points_to_keep]
    return indices, design_points

#---------------------------------------------------------------
def _split_training_validation_indices(
    design_points: npt.NDArray[np.int32],
    validation_indices: list[int],
    design_points_to_exclude: list[int] | None = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32], npt.NDArray[np.int64], npt.NDArray[np.int32]]:
    ''' Get numpy indices of training and validation sets

    :param npt.NDArray[np.int32] design_points: list of design points (in their original order in the file).
    :param list[int] validation_indices: list of validation indices
    :param list[int] design_points_to_exclude: list of design point indices to exclude (in the original design point)
    :return tuple[npt.NDArray[np.int64], npt.NDArray[np.int32], npt.NDArray[np.int64], npt.NDArray[np.int32]]: numpy indices of training, design points, numpy indices of validation sets, validation design points
    '''
    # Determine the training and validation masks, providing indices for selecting
    # the relevant design points parameters and associated values
    training_mask = np.isin(design_points, validation_indices, invert=True)
    validation_mask = ~training_mask

    np_training_indices = np.where(training_mask)[0]
    np_validation_indices = np.where(validation_mask)[0]

    training_design_points = design_points[np_training_indices]
    validation_design_points = design_points[np_validation_indices]

    if design_points_to_exclude:
        # Determine which design points to keep, and then apply those masks to the indices and design points themselves:
        # For training
        np_training_indices, training_design_points = _filter_design_points(
            indices=np_training_indices,
            design_points=training_design_points,
            design_points_to_exclude=design_points_to_exclude,
        )
        # And validation
        np_validation_indices, validation_design_points = _filter_design_points(
            indices=np_validation_indices,
            design_points=validation_design_points,
            design_points_to_exclude=design_points_to_exclude,
        )

    # Most useful is to have the training and validation indices. However, we also sometimes
    # need the design points themselves (for excluding design points), so we return those as well
    return np_training_indices, training_design_points, np_validation_indices, validation_design_points