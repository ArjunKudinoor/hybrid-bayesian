"""
Systematic uncertainty correlation management for Bayesian inference

OVERVIEW:
=========
Manages correlation structure for systematic uncertainties in likelihood calculations.
Supports two independent systematic uncertainty systems that cannot be mixed:

1. LEGACY SYSTEM: Summed systematics with exponential decay (intra-observable only)
2. ADVANCED SYSTEM: Individual systematics with group tags (cross-observable capable)

Design principle: Simple, user-controlled correlation structure via configuration tags.
No physics assumptions about systematic sources - user defines correlations explicitly.

MAIN CLASSES:
=============
SystematicInfo: Data structure for individual systematic uncertainty information
SystematicCorrelationManager: Core class managing correlation groups and covariance calculation

KEY FEATURES:
=============
- Parse correlation specifications from configuration (e.g., 'jec:alice', 'taa:global')
- Build correlation groups based on user-defined tags (agnostic to tag meanings)
- Create correlation-aware covariance matrices for likelihood calculations
- Validation and debugging tools for correlation structure

TWO SYSTEMATIC UNCERTAINTY APPROACHES:
======================================

LEGACY MODE:
---------------------------------------------
Config format: 'sum:cor_length:cor_strength'
Algorithm:
  1. Sum all systematic sources in quadrature: σ_total = √(Σ σᵢ²)
  2. Apply exponential decay correlation within observable:
     ρ(i,j) = cor_strength × exp(-|i-j|/cor_length)
  3. No cross-observable correlation possible

Parameters:
  cor_length: Correlation length in bins
    -1 = fully correlated within observable
    >0 = exponential decay over cor_length bins
  cor_strength: Overall correlation strength [0, 1]
    0 = uncorrelated, 1 = fully correlated

Correlation structure:
  Block-diagonal: Each observable independent
  Intra-observable: Exponential decay based on bin separation

Use case:
  - Compatibility with original STAT repository
  - Global analyses where cross-observable correlations are negligible
  - Exploratory studies

ADVANCED MODE (recommended for precision measurements):
-------------------------------------------------------
Config format: 'name:group_tag' (e.g., 'jec:alice', 'taa:global', 'tracking:uncor')
Algorithm:
  1. Track each systematic source separately
  2. Group systematics by correlation tag
  3. Apply full correlation within groups across all observables

Group tags define correlation:
  Same tag → fully correlated across all observables
  Different tags → completely uncorrelated
  Special tag 'uncor' → diagonal (no correlation even within observable)

Examples:
  'jec:alice' + 'jec:cms' → JEC uncorrelated between experiments
  'taa:global' in all obs → TAA fully correlated everywhere
  'tracking:uncor' → no correlation at all

Advantage: Proper treatment of global systematics (TAA, luminosity, trigger efficiency)
  Global systematic can affect multiple observables with correct correlation

Use case:
  - Precision physics measurements for publication
  - Analyses with known global systematics
  - Multi-experiment combinations

USAGE EXAMPLE (Advanced Mode):
===============================

# 1. Configuration specifies correlation structure
observable_list:
  - observable: 'jet_pt_alice'
    sys_data: ['jec:alice', 'taa:global']      # JEC specific to ALICE, TAA global
  - observable: 'jet_pt_cms'
    sys_data: ['jec:cms', 'taa:global']        # Different JEC, same TAA

# 2. Create and configure correlation manager
from systematic_correlation import SystematicCorrelationManager

correlation_manager = SystematicCorrelationManager()
correlation_manager.parse_configuration(parsed_observables)
correlation_manager.register_observable_ranges(observable_ranges)

# 3. Calculate correlation-aware covariance matrix
systematic_cov = correlation_manager.create_systematic_covariance_matrix(
    systematic_uncertainties,  # shape: (n_features, n_systematics)
    systematic_names,          # list of 'name:tag' strings
    n_features                 # total bins across all observables
)

# Result: systematic_cov shape (n_features, n_features)
# - Diagonal blocks: individual systematic contributions within observables
# - Off-diagonal: cross-observable correlations from shared tags

IMPLEMENTATION NOTES:
=====================
- SystematicInfo stores metadata for each systematic source
- Correlation groups built during parse_configuration()
- Covariance matrix construction happens during MCMC initialization
- Exponential decay only applies to summed systematics
- Individual systematics always fully correlated within their observable
- Serialization/deserialization supported for HDF5 storage

For data loading and integration, see data_IO.py
For likelihood calculation, see log_posterior.py
For visualization, see plot_covariance.py

.. codeauthor:: Jingyu Zhang <jingyu.zhang@cern.ch>, Vanderbilt
"""

from __future__ import annotations

import logging
from collections import defaultdict

import attrs
import numpy as np

logger = logging.getLogger(__name__)


@attrs.define
class SystematicInfo:
    """
    Store information about a systematic uncertainty.

    Two types:
    1. Individual systematics: Use group tags for cross-observable correlation
       - Always fully correlated within observable
       - cor_length and cor_strength not used

    2. Summed systematics: Use cor_length/cor_strength for intra-observable correlation
       - No cross-observable correlation (no group tag)
       - cor_length and cor_strength define bin-to-bin correlation
    """

    base_name: str  # e.g., 'jec', 'taa', 'sum'
    correlation_tag: str  # e.g., 'alice', '5020' (empty string for sum)
    full_name: str  # e.g., 'jec:alice' or 'sum_observable_name'
    is_summed: bool = attrs.field(default=False)  # True if this is a summed systematic

    # Correlation parameters (ONLY used for summed systematics)
    cor_length: int = attrs.field(default=-1)  # -1 means all bins (only applies to sum)
    cor_strength: float = attrs.field(default=1.0)  # Only applies to sum

    # Derived properties. Initialized based on the other arguments - see below.
    is_uncorrelated: bool = attrs.field(init=False)  # True if tag is 'uncor'

    def __attrs_post_init__(self):
        """Validate systematic info after initialization."""
        self.is_uncorrelated = self.correlation_tag.lower() == "uncor"

        # Validation: individual systematics should not have correlation parameters
        if not self.is_summed and (self.cor_length != -1 or self.cor_strength != 1.0):
            logger.warning(
                f"Individual systematic {self.full_name} has cor_length/cor_strength "
                f"(length={self.cor_length}, strength={self.cor_strength}). "
                f"These parameters are ignored - individual systematics use full correlation "
                f"within observable and group tags for cross-observable correlation."
            )

        # Validation: summed systematics should not be uncorrelated
        if self.is_summed and self.is_uncorrelated:
            msg = (
                f"Summed systematic {self.full_name} cannot be uncorrelated. "
                f"Sum systematics combine multiple sources - use individual systematics with 'uncor' tag instead."
            )
            raise ValueError(msg)

        # Validation: correlation strength bounds
        if self.cor_strength < 0.0 or self.cor_strength > 1.0:
            logger.warning(
                f"Correlation strength {self.cor_strength} for {self.full_name} outside [0,1]. Clipping to valid range."
            )
            self.cor_strength = np.clip(self.cor_strength, 0.0, 1.0)


def parse_systematic_config(sys_config_string: str) -> dict:
    """
    Parse systematic configuration string.

    ALLOWED FORMATS:
    1. Individual systematic: 'name:group_tag'
       - Example: 'jec:alice', 'taa:5020'
       - Always fully correlated within observable (all bins)
       - Group tag controls cross-observable correlation

    2. Summed systematic: 'sum:cor_length:cor_strength' or 'sum'
       - Example: 'sum:10:0.8', 'sum'
       - cor_length and cor_strength control intra-observable correlation
       - No cross-observable correlation

    DISABLED FORMATS (will raise ValueError):
     'sum:group_tag:...'              - sum cannot have group tags
     'name:tag:cor_length:cor_strength' - individual systematics cannot have correlation params

    Args:
        sys_config_string: Configuration string from config file

    Returns:
        Dictionary with keys:
            - 'type': 'individual' or 'sum'
            - 'name': base systematic name
            - 'group_tag': correlation group tag (empty for sum)
            - 'cor_length': correlation length (-1 for individual or all bins)
            - 'cor_strength': correlation coefficient (1.0 for individual)
    """
    parts = sys_config_string.split(":")

    if parts[0] == "sum":
        # Summed systematic: sum[:cor_length[:cor_strength]]
        if len(parts) > 3:
            msg = (
                f"Invalid sum format: '{sys_config_string}'. "
                f"Sum systematics cannot have group tags. "
                f"Use format: 'sum' or 'sum:cor_length:cor_strength'"
            )
            raise ValueError(msg)

        try:
            cor_length = int(parts[1]) if len(parts) > 1 else -1
            cor_strength = float(parts[2]) if len(parts) > 2 else 1.0
        except (ValueError, IndexError) as e:
            msg = (
                f"Invalid sum format: '{sys_config_string}'. "
                f"Expected 'sum' or 'sum:cor_length:cor_strength' where cor_length is int and cor_strength is float. "
                f"Error: {e}"
            )
            raise ValueError(msg) from e

        config = {
            "type": "sum",
            "name": "sum",
            "group_tag": "",  # Empty - no cross-observable correlation
            "cor_length": cor_length,
            "cor_strength": cor_strength,
        }
        logger.debug(f"Parsed sum: {sys_config_string} -> length={cor_length}, strength={cor_strength}")

    else:
        # Individual systematic: name:group_tag OR just name (auto-generates unique tag)
        if len(parts) == 1:
            # Just 'name' provided - create unique group tag (name itself)
            # This makes each systematic uncorrelated across observables by default
            # User can then use correlation_groups to specify correlation parameters
            config = {
                "type": "individual",
                "name": parts[0],
                "group_tag": parts[0],  # Use name as group tag (unique per systematic)
                "cor_length": -1,  # Not used for individual
                "cor_strength": 1.0,  # Not used for individual
            }
            logger.debug(f"Parsed individual (auto-tag): {sys_config_string} -> name='{parts[0]}', group='{parts[0]}'")

        elif len(parts) == 2:
            # Full 'name:group_tag' format
            config = {
                "type": "individual",
                "name": parts[0],
                "group_tag": parts[1],
                "cor_length": -1,  # Not used for individual
                "cor_strength": 1.0,  # Not used for individual
            }
            logger.debug(f"Parsed individual: {sys_config_string} -> name='{parts[0]}', group='{parts[1]}'")

        else:
            # More than 2 parts - error
            raise ValueError(
                f"Invalid individual systematic format: '{sys_config_string}'. "
                f"Individual systematics must use format 'name' or 'name:group_tag' (e.g., 'jec' or 'jec:alice'). "
                f"Correlation length/strength parameters are not allowed - individual systematics are always fully correlated within observable."
            )
    # Validate correlation parameters (only meaningful for sum)
    if config["type"] == "sum":
        if config["cor_strength"] < 0.0 or config["cor_strength"] > 1.0:
            logger.warning(f"Correlation strength {config['cor_strength']} outside [0,1], clipping to [0,1]")
            config["cor_strength"] = np.clip(config["cor_strength"], 0.0, 1.0)

        if config["cor_length"] < -1 or config["cor_length"] == 0:
            logger.warning(f"Invalid correlation length {config['cor_length']}, setting to -1 (all bins)")
            config["cor_length"] = -1

    return config


@attrs.define
class SystematicCorrelationManager:
    """
    Manages systematic uncertainty correlations based on user configuration.
    Makes no assumptions about the meaning of correlation tags.
    """

    # Map correlation tags to lists of (observable, feature_range, systematic)
    correlation_groups: dict[str, list[tuple[str, int, int, str]]] = attrs.field(default=defaultdict(list))
    # Structure: correlation_tag -> [(observable_label, start_idx, end_idx, systematic_full_name), ...]

    # Map systematic full names to their info
    systematic_info: dict[str, SystematicInfo] = attrs.field(factory=dict)
    # Structure: systematic_full_name -> SystematicInfo

    # Map observables to their expected systematics
    observable_systematics: dict[str, list[str]] = attrs.field(factory=dict)
    # Structure: observable_label -> [systematic_full_names]

    # Store all unique systematic full names for consistent ordering
    all_systematic_names: list[str] = attrs.field(factory=list)

    # Store observable ranges for covariance calculation
    _observable_ranges: list[tuple[int, int, str]] = attrs.field(factory=list)

    _pending_correlation_params: dict[str, str] = attrs.field(factory=dict, init=False)

    default_cor_length: int = attrs.field(default=-1)  # -1 means full correlation (all bins)
    default_cor_strength: float = attrs.field(default=1.0)  # 1.0 means fully correlated

    def parse_configuration(self, parsed_observables: list[tuple[str, list[str], list[str], str | None]]):
        """
        Parse systematic configuration with two separate systems:

        System 1 - Individual systematics (NEW, recommended):
            Format: 'name:group_tag' (e.g., 'jec:alice', 'taa:5020')
            - Always fully correlated within observable
            - Cross-observable correlation via group tags
            - Clean physics interpretation

        System 2 - Summed systematics:
            Format: 'sum:cor_length:cor_strength' or 'sum'
            - Intra-observable correlation via cor_length/cor_strength
            - NO cross-observable correlation
            - Each observable is independent

        NOTE: Cannot mix individual and sum within same observable.
        NOTE: cor_length=-1 will be resolved to actual bin counts in resolve_bin_counts()

        Args:
            parsed_observables: List of (observable_name, sys_data_list, sys_theory_list)
        """
        logger.info("Parsing systematic correlation configuration...")

        all_systematic_full_names = set()

        for obs_name, sys_data_list, sys_theory_list, external_stat_cov in parsed_observables:
            self.observable_systematics[obs_name] = []

            # Check for mixing (not allowed) - collect types first
            sys_types = {parse_systematic_config(s)["type"] for s in sys_data_list}

            if len(sys_types) > 1:
                msg = (
                    f"Observable '{obs_name}' mixes different systematic types: {sys_types}. "
                    f"You must use EITHER individual systematics ('name:tag') "
                    f"OR summed systematics ('sum:...'), not both."
                )
                raise ValueError(msg)

            # Now process systematics
            for sys_config_string in sys_data_list:
                config = parse_systematic_config(sys_config_string)

                sys_base_name = config["name"]
                correlation_tag = config["group_tag"]
                cor_length = config["cor_length"]
                cor_strength = config["cor_strength"]
                is_summed = config["type"] == "sum"

                # Construct full name
                if is_summed:
                    # Sum: Make unique per observable (no cross-observable correlation)
                    full_name = f"sum_{obs_name}"
                else:
                    # Individual: Use base name + group tag
                    # NEW: If group_tag equals base_name (auto-generated), make it unique per observable
                    if correlation_tag == sys_base_name:
                        # Auto-generated tag - make it unique per observable
                        correlation_tag = f"{sys_base_name}_{obs_name}"
                        full_name = f"{sys_base_name}:{correlation_tag}"
                        logger.debug(
                            f"  Auto-generated unique tag for {sys_base_name} on {obs_name}: {correlation_tag}"
                        )
                    else:
                        # Explicit tag from user - use as-is
                        full_name = f"{sys_base_name}:{correlation_tag}"

                # Store systematic info
                sys_info = SystematicInfo(
                    base_name=sys_base_name,
                    correlation_tag=correlation_tag,
                    full_name=full_name,
                    is_summed=is_summed,
                    cor_length=cor_length,
                    cor_strength=cor_strength,
                )
                self.systematic_info[full_name] = sys_info
                self.observable_systematics[obs_name].append(full_name)
                all_systematic_full_names.add(full_name)

                if is_summed:
                    logger.debug(f"  {obs_name}: sum (cor_length={cor_length}, cor_strength={cor_strength})")
                else:
                    logger.debug(f"  {obs_name}: {full_name}")

        # Create consistent ordering
        self.all_systematic_names = sorted(all_systematic_full_names)
        logger.info(f"Found {len(self.all_systematic_names)} unique systematics")

        # Summary
        n_summed = sum(1 for info in self.systematic_info.values() if info.is_summed)
        n_individual = len(self.systematic_info) - n_summed
        logger.info(f"  Individual systematics: {n_individual}")
        logger.info(f"  Summed systematics: {n_summed}")

        # Check for unresolved cor_length
        n_unresolved = sum(1 for info in self.systematic_info.values() if info.is_summed and info.cor_length == -1)
        if n_unresolved > 0:
            logger.info(
                f"  Summed systematics with unresolved cor_length: {n_unresolved} (will resolve after data load)"
            )

    def set_correlation_parameters(self, correlation_groups_params: dict[str, str]):
        """
        Store correlation parameters to be applied after correlation groups are built.

        Special handling for 'default' key which sets fallback parameters for unspecified groups.

        Args:
            correlation_groups_params: Dict like {'default': '20:0.8', 'alice': '10:0.9', ...}
        """
        logger.info("Storing correlation parameters for later application...")

        # Check for 'default' key and extract it
        if "default" in correlation_groups_params:
            default_str = correlation_groups_params["default"]
            logger.info(f"Found 'default' correlation parameters: {default_str}")

            try:
                # Parse "cor_length:cor_strength" format
                parts = default_str.split(":")
                if len(parts) != 2:
                    raise ValueError(f"Expected 'length:strength', got '{default_str}'")

                self.default_cor_length = int(parts[0])
                self.default_cor_strength = float(parts[1])

                # Validate
                if self.default_cor_length < -1 or self.default_cor_length == 0:
                    logger.warning(f"Invalid default cor_length={self.default_cor_length}, using -1")
                    self.default_cor_length = -1
                if self.default_cor_strength < 0.0 or self.default_cor_strength > 1.0:
                    logger.warning(f"default cor_strength={self.default_cor_strength} outside [0,1], clipping")
                    self.default_cor_strength = np.clip(self.default_cor_strength, 0.0, 1.0)

                logger.info(
                    f"Set default correlation parameters: length={self.default_cor_length}, strength={self.default_cor_strength}"
                )

            except (ValueError, IndexError) as e:
                logger.error(f"Failed to parse 'default' correlation parameters '{default_str}': {e}")
                logger.error("Using default values: cor_length=-1, cor_strength=1.0")
                self.default_cor_length = -1
                self.default_cor_strength = 1.0

        # Store all parameters (including 'default' for now, will be filtered later)
        self._pending_correlation_params = correlation_groups_params
        logger.info(f"Stored parameters for {len(correlation_groups_params)} group tags")

    def _apply_correlation_parameters(self, correlation_groups_params: dict[str, str]):
        """
        Set correlation parameters for individual systematic groups from config.

        For groups explicitly listed in correlation_groups_params: use specified parameters
        For groups NOT listed: use default parameters (self.default_cor_length, self.default_cor_strength)

        Args:
            correlation_groups_params: Dict like {'default': '20:0.8', 'alice': '10:0.9', 'cms': '5:0.95'}
        """
        logger.info("Setting correlation parameters from correlation_groups config...")

        # Track which tags are configured vs used (excluding 'default')
        configured_tags = set(correlation_groups_params.keys()) - {"default"}
        used_tags = set(self.correlation_groups.keys())

        # Warn about unused configurations
        unused_tags = configured_tags - used_tags
        if unused_tags:
            logger.warning(f"Correlation groups configured but not used: {sorted(unused_tags)}")

        # Find groups that need default parameters
        unconfigured_tags = used_tags - configured_tags
        if unconfigured_tags:
            logger.info(f"Groups using default parameters: {sorted(unconfigured_tags)}")

        # Process all used groups
        for group_tag in used_tags:
            # Determine which parameters to use
            if group_tag in correlation_groups_params:
                # Use explicitly configured parameters
                param_string = correlation_groups_params[group_tag]

                try:
                    parts = param_string.split(":")
                    if len(parts) != 2:
                        raise ValueError(f"Expected 'length:strength', got '{param_string}'")

                    cor_length = int(parts[0])
                    cor_strength = float(parts[1])

                    # Validate
                    if cor_length < -1 or cor_length == 0:
                        logger.warning(f"Invalid cor_length={cor_length}, using -1")
                        cor_length = -1
                    if cor_strength < 0.0 or cor_strength > 1.0:
                        logger.warning(f"cor_strength={cor_strength} outside [0,1], clipping")
                        cor_strength = np.clip(cor_strength, 0.0, 1.0)

                    source = "explicit"

                except (ValueError, IndexError) as e:
                    logger.error(f"Failed to parse '{group_tag}': {param_string} - {e}")
                    logger.error(f"Using default parameters for group '{group_tag}'")
                    cor_length = self.default_cor_length
                    cor_strength = self.default_cor_strength
                    source = "default (error fallback)"
            else:
                # Use default parameters
                cor_length = self.default_cor_length
                cor_strength = self.default_cor_strength
                source = "default"

            # Find all systematics in this group
            group_systematics = set()
            for _obs_name, _start, _end, sys_full_name in self.correlation_groups[group_tag]:
                group_systematics.add(sys_full_name)

            # Update each systematic
            n_updated = 0
            for sys_full_name in group_systematics:
                if sys_full_name in self.systematic_info:
                    sys_info = self.systematic_info[sys_full_name]
                    if not sys_info.is_summed:
                        sys_info.cor_length = cor_length
                        sys_info.cor_strength = cor_strength
                        n_updated += 1

            logger.info(
                f"  Group '{group_tag}': Updated {n_updated} systematic(s) with length={cor_length}, strength={cor_strength} ({source})"
            )

        logger.info("Correlation parameter configuration complete")

    def register_observable_ranges(self, observable_ranges: list[tuple[int, int, str]]) -> None:
        """
        Register which features belong to which observables and build correlation groups.

        :param observable_ranges: List of (start_idx, end_idx, observable_label)
        """
        logger.info("Building correlation groups from observable ranges...")

        # Store observable ranges for later use in covariance calculation
        self._observable_ranges = observable_ranges

        # Clear existing correlation groups
        self.correlation_groups.clear()

        # Build correlation groups by going through each observable
        for start_idx, end_idx, obs_label in observable_ranges:
            if obs_label in self.observable_systematics:
                for sys_full_name in self.observable_systematics[obs_label]:
                    sys_info = self.systematic_info[sys_full_name]

                    if not sys_info.is_uncorrelated:
                        # Group by correlation tag (whatever the user specified)
                        correlation_tag = sys_info.correlation_tag
                        self.correlation_groups[correlation_tag].append((obs_label, start_idx, end_idx, sys_full_name))

        # Log correlation groups for debugging
        logger.info("Correlation groups built:")
        for group_tag, group_members in self.correlation_groups.items():
            logger.info(f"  Group '{group_tag}': {len(group_members)} entries")
            for obs_label, start, end, sys_name in group_members:
                logger.debug(f"    {sys_name} on {obs_label} (features {start}:{end})")

        # Apply pending correlation parameters now that groups are built
        if self._pending_correlation_params:
            self._apply_correlation_parameters(self._pending_correlation_params)

    def resolve_bin_counts(self, observable_ranges: list[tuple[int, int, str]]) -> None:
        """
        Resolve cor_length=-1 to actual bin counts for SUMMED systematics only.

        Individual systematics are always fully correlated and don't use cor_length.
        This method only updates summed systematics that have cor_length=-1.

        Args:
            observable_ranges: List of (start_idx, end_idx, observable_label)
        """
        logger.info("Resolving correlation lengths for summed systematics...")

        # Build map of observable -> bin count
        obs_bin_counts = {}
        for start_idx, end_idx, obs_label in observable_ranges:
            n_bins = end_idx - start_idx
            obs_bin_counts[obs_label] = n_bins
            logger.debug(f"  Observable '{obs_label}': {n_bins} bins")

        n_resolved = 0
        n_already_set = 0

        # Update only summed systematics with cor_length=-1
        for full_name, sys_info in self.systematic_info.items():
            if not sys_info.is_summed:
                continue

            if sys_info.cor_length == -1:
                # Extract observable name from full_name (format: 'sum_observable_name')
                if full_name.startswith("sum_"):
                    obs_name = full_name[4:]  # Remove 'sum_' prefix

                    if obs_name in obs_bin_counts:
                        actual_bins = obs_bin_counts[obs_name]
                        logger.debug(f"  Resolved '{full_name}': cor_length -1 -> {actual_bins} bins")
                        sys_info.cor_length = actual_bins
                        n_resolved += 1
                    else:
                        logger.warning(f"  Could not find bin count for '{obs_name}', leaving cor_length=-1")
            else:
                n_already_set += 1

        logger.info("Bin count resolution complete:")
        logger.info(f"  Resolved: {n_resolved}")
        logger.info(f"  Already had explicit values: {n_already_set}")

    def build_intra_observable_correlation_matrix(self, systematic_full_name: str, n_bins: int) -> np.ndarray:
        """
        Build intra-observable correlation matrix for a systematic.

        TWO CASES:
        1. Individual systematics: Returns identity matrix (placeholder)
        - Always fully correlated within observable
        - Actual correlation handled by outer product in covariance calculation

        2. Summed systematics: Returns correlation matrix using EXPONENTIAL DECAY
        - C[i,j] = cor_strength * exp(-|i-j| / cor_length) for i ≠ j
        - Smooth decay with characteristic length cor_length

        Args:
            systematic_full_name: Full name of systematic (e.g., 'sum_observable_name')
            n_bins: Number of bins in the observable

        Returns:
            Correlation matrix C of shape (n_bins, n_bins)

        Example for sum with cor_length=2, cor_strength=0.8, n_bins=5:
            Exponential decay: C[i,j] = 0.8 * exp(-|i-j|/2)
            [[1.00, 0.49, 0.29, 0.18, 0.11],
            [0.49, 1.00, 0.49, 0.29, 0.18],
            [0.29, 0.49, 1.00, 0.49, 0.29],
            [0.18, 0.29, 0.49, 1.00, 0.49],
            [0.11, 0.18, 0.29, 0.49, 1.00]]
        """
        sys_info = self.systematic_info.get(systematic_full_name)

        if sys_info is None:
            logger.warning(f"Systematic '{systematic_full_name}' not found, returning identity")
            return np.eye(n_bins)

        if not sys_info.is_summed:
            # Individual systematics: fully correlated (identity is placeholder)
            # Actual correlation handled by outer product in covariance calculation
            return np.eye(n_bins)

        # Summed systematic: use exponential decay correlation
        cor_length = sys_info.cor_length
        cor_strength = sys_info.cor_strength

        logger.debug(f"Building exponential correlation matrix for '{systematic_full_name}':")
        logger.debug(f"  n_bins={n_bins}, cor_length={cor_length}, cor_strength={cor_strength}")

        # Check if cor_length still needs resolution
        if cor_length == -1:
            logger.warning(f"cor_length=-1 for '{systematic_full_name}' not yet resolved!")
            logger.warning("Using full correlation (cor_length=n_bins) as fallback")
            cor_length = n_bins

        # Build correlation matrix with exponential decay
        C = np.zeros((n_bins, n_bins))

        for i in range(n_bins):
            for j in range(n_bins):
                if i == j:
                    # Diagonal is always 1.0
                    C[i, j] = 1.0
                else:
                    # Exponential decay: cor_strength * exp(-|i-j| / cor_length)
                    distance = abs(i - j)
                    C[i, j] = cor_strength * np.exp(-distance / cor_length)

        logger.debug(f"  Matrix shape: {C.shape}")
        logger.debug(f"  Min off-diagonal correlation: {np.min(C[~np.eye(n_bins, dtype=bool)]):.6f}")
        logger.debug(f"  Max off-diagonal correlation: {np.max(C[~np.eye(n_bins, dtype=bool)]):.6f}")

        return C

    def get_systematic_names_for_observable(self, observable_label: str) -> list[str]:
        """Get list of systematic full names for a given observable"""
        return self.observable_systematics.get(observable_label, [])

    def get_all_systematic_names(self) -> list[str]:
        """Get consistent ordering of all systematic names"""
        return self.all_systematic_names.copy()

    def create_systematic_covariance_matrix(  # noqa: C901
        self,
        systematic_uncertainties: np.ndarray,
        systematic_names: list[str],
        n_features: int,
    ) -> np.ndarray:
        """
        Create systematic covariance matrix with two independent systems:

        System 1 - Individual systematics:
            - Fully correlated within observable (all bins)
            - Cross-observable correlation controlled by group tags
            - Same tag → correlated across observables
            - Different tag → uncorrelated across observables

        System 2 - Summed systematics:
            - Intra-observable correlation via cor_length and cor_strength
            - NO cross-observable correlation (each observable independent)

        Args:
            systematic_uncertainties: Matrix of shape (n_features, n_systematics)
                                    Each column is a systematic source
            systematic_names: List of systematic names (must match columns)
            n_features: Total number of features (bins) across all observables

        Returns:
            Covariance matrix of shape (n_features, n_features)
        """
        logger.info("Creating systematic covariance matrix...")

        logger.debug(f"  Input shape: {systematic_uncertainties.shape}")
        logger.debug(f"  n_features: {n_features}, n_systematics: {len(systematic_names)}")

        # Initialize total covariance matrix
        total_cov = np.zeros((n_features, n_features))

        # PATH 1: Process individual systematics (grouped by correlation tag)
        for group_tag, group_members in self.correlation_groups.items():
            if not group_tag:  # Skip empty tags (these are for summed systematics)
                continue

            logger.debug(f"Processing correlation group '{group_tag}' with {len(group_members)} members")

            # Build covariance for all members in this group
            # group_members is: [(obs_label, start_idx, end_idx, systematic_full_name), ...]
            # Get unique systematics in this group
            unique_systematics = list({sys_name for _, _, _, sys_name in group_members})

            # Process each unique systematic
            for sys_full_name in unique_systematics:
                if sys_full_name not in systematic_names:
                    logger.warning(f"Systematic '{sys_full_name}' not found in systematic_names")
                    continue

                sys_idx = systematic_names.index(sys_full_name)
                sys_info = self.systematic_info.get(sys_full_name)
                if not sys_info:
                    continue

                # Get all bins where this systematic appears (group-local indexing)
                group_global_indices = []
                for _obs_label, start, end, sys_name in group_members:
                    if sys_name == sys_full_name:
                        group_global_indices.extend(range(start, end))

                # Build group-local mapping (ignores gaps)
                global_to_group_local = {}
                for group_local_idx, global_idx in enumerate(group_global_indices):
                    global_to_group_local[global_idx] = group_local_idx

                # Get correlation parameters
                cor_length = sys_info.cor_length
                cor_strength = sys_info.cor_strength
                uncertainties = systematic_uncertainties[:, sys_idx]

                logger.info(f"DEBUG: {sys_full_name} - cor_length={cor_length}, cor_strength={cor_strength}")
                logger.info(f"DEBUG: {sys_full_name} - n_bins in group={len(group_global_indices)}")
                logger.info(f"DEBUG: Will use {'FULL correlation' if cor_length == -1 else 'EXPONENTIAL decay'}")

                # Apply correlation
                if cor_length == -1:
                    # Full correlation (default)
                    for global_i in group_global_indices:
                        for global_j in group_global_indices:
                            total_cov[global_i, global_j] += uncertainties[global_i] * uncertainties[global_j]
                else:
                    # Exponential decay using group-local distance
                    for global_i in group_global_indices:
                        for global_j in group_global_indices:
                            if global_i == global_j:
                                correlation = 1.0
                            else:
                                # Distance in group space (ignoring gaps)
                                group_local_i = global_to_group_local[global_i]
                                group_local_j = global_to_group_local[global_j]
                                distance = abs(group_local_i - group_local_j)
                                correlation = cor_strength * np.exp(-distance / cor_length)
                            if global_i != global_j and global_i < 5 and global_j >= 7:  # Cross-observable example
                                logger.info(
                                    f"DEBUG CROSS: Adding {sys_full_name} correlation between bins {global_i} and {global_j}"
                                )

                            total_cov[global_i, global_j] += (
                                correlation * uncertainties[global_i] * uncertainties[global_j]
                            )

        # PATH 2: Process summed systematics (independent per observable)
        for sys_full_name, sys_info in self.systematic_info.items():
            if not sys_info.is_summed:
                continue

            if sys_full_name not in systematic_names:
                logger.warning(f"Summed systematic '{sys_full_name}' not found in systematic_names")
                continue

            sys_idx = systematic_names.index(sys_full_name)

            # Find which observable this summed systematic belongs to
            obs_found = False
            for obs_label, sys_list in self.observable_systematics.items():
                if sys_full_name not in sys_list:
                    continue

                # Find the feature range for this observable
                for start, end, obs_name in self._observable_ranges:
                    if obs_name == obs_label:
                        n_bins = end - start
                        sys_uncertainties = systematic_uncertainties[start:end, sys_idx]

                        # Build intra-observable correlation matrix
                        C = self.build_intra_observable_correlation_matrix(sys_full_name, n_bins)

                        # Add to covariance (only within observable, no cross-observable terms)
                        cov_block = np.outer(sys_uncertainties, sys_uncertainties) * C
                        total_cov[start:end, start:end] += cov_block

                        logger.debug(
                            f"  Added summed systematic: {sys_full_name} for {obs_label} "
                            f"(cor_length={sys_info.cor_length}, cor_strength={sys_info.cor_strength})"
                        )
                        obs_found = True
                        break

                if obs_found:
                    break

            if not obs_found:
                logger.warning(f"Could not find observable range for summed systematic '{sys_full_name}'")

        # Handle uncorrelated systematics (diagonal only)
        for sys_full_name, sys_info in self.systematic_info.items():
            if not sys_info.is_uncorrelated:
                continue

            if sys_full_name not in systematic_names:
                continue

            sys_idx = systematic_names.index(sys_full_name)
            sys_uncertainties = systematic_uncertainties[:, sys_idx]

            # Add as diagonal contribution only
            total_cov += np.diag(sys_uncertainties**2)
            logger.debug(f"  Added uncorrelated systematic: {sys_full_name} (diagonal only)")

        logger.info(f"Systematic covariance matrix created: shape {total_cov.shape}")
        logger.debug(f"  Diagonal mean: {np.mean(np.diag(total_cov)):.6e}")
        logger.debug(f"  Off-diagonal mean: {np.mean(total_cov - np.diag(np.diag(total_cov))):.6e}")
        logger.debug(f"  Total variance: {np.trace(total_cov):.6e}")

        return total_cov

    def _create_correlation_block(
        self, uncertainties: np.ndarray, correlated_indices: list[int], n_features: int
    ) -> np.ndarray:
        """
        Create correlation block for a specific set of features.
        Assumes full correlation: C_ij = σ_i * σ_j for correlated features.

        :param uncertainties: Full uncertainty array (n_features,)
        :param correlated_indices: List of feature indices that should be correlated
        :param n_features: Total number of features
        :return: Covariance matrix with correlation block
        """
        cov_matrix = np.zeros((n_features, n_features))

        # Create fully correlated block: C_ij = σ_i * σ_j
        for i_idx in correlated_indices:
            for j_idx in correlated_indices:
                cov_matrix[i_idx, j_idx] = uncertainties[i_idx] * uncertainties[j_idx]

        return cov_matrix

    def get_correlation_summary(self) -> dict:
        """
        Get summary information about the correlation structure for debugging/validation
        """
        summary = {
            "n_systematics": len(self.all_systematic_names),
            "n_observables": len(self.observable_systematics),
            "n_correlation_groups": len(self.correlation_groups),
            "correlation_groups": {},
            "uncorrelated_systematics": [],
        }

        # Group information
        for group_tag, group_members in self.correlation_groups.items():
            summary["correlation_groups"][group_tag] = {
                "n_entries": len(group_members),
                "systematics": list({sys_name for _, _, _, sys_name in group_members}),
                "observables": list({obs_name for obs_name, _, _, _ in group_members}),
            }

        # Uncorrelated systematics
        for sys_full_name, sys_info in self.systematic_info.items():
            if sys_info.is_uncorrelated:
                summary["uncorrelated_systematics"].append(sys_full_name)

        return summary

    def validate_configuration(self) -> list[str]:
        """
        Validate the correlation configuration and return list of warnings/errors
        """
        warnings = []

        # Check for systematics that appear in config but no correlation groups
        for sys_full_name, sys_info in self.systematic_info.items():
            if not sys_info.is_uncorrelated:
                found_in_group = False
                for group_members in self.correlation_groups.values():
                    if any(sys_name == sys_full_name for _, _, _, sys_name in group_members):
                        found_in_group = True
                        break

                if not found_in_group:
                    warnings.append(f"Systematic {sys_full_name} has correlation tag but no correlation group")

        # Check for empty correlation groups
        for group_tag, group_members in self.correlation_groups.items():
            if len(group_members) <= 1:
                warnings.append(f"Correlation group '{group_tag}' has only {len(group_members)} member(s)")

        return warnings

    def to_dict(self) -> dict:
        """
        Convert SystematicCorrelationManager to a serializable dictionary for HDF5 storage.

        :return: Dictionary representation of the correlation manager
        """
        return {
            "correlation_groups": dict(self.correlation_groups),  # Convert defaultdict to dict
            "systematic_info": {
                full_name: {
                    "base_name": info.base_name,
                    "correlation_tag": info.correlation_tag,
                    "full_name": info.full_name,
                    "is_summed": info.is_summed,
                    "is_uncorrelated": info.is_uncorrelated,
                    "cor_length": info.cor_length,
                    "cor_strength": info.cor_strength,
                }
                for full_name, info in self.systematic_info.items()
            },
            "observable_systematics": dict(self.observable_systematics),
            "all_systematic_names": self.all_systematic_names,
            "_pending_correlation_params": self._pending_correlation_params,
            "default_cor_length": self.default_cor_length,  # ADD THIS LINE
            "default_cor_strength": self.default_cor_strength,  # ADD THIS LINE
            "class_name": "SystematicCorrelationManager",  # For validation during loading
        }

    @classmethod
    def from_dict(cls, data: dict) -> SystematicCorrelationManager:
        """
        Reconstruct SystematicCorrelationManager from serialized dictionary.

        :param data: Dictionary representation from to_dict()
        :return: Reconstructed SystematicCorrelationManager instance
        """
        # Validate that this is the right type of data
        class_name = data.get("class_name")
        if isinstance(class_name, np.ndarray):
            class_name = str(class_name.item())  # Convert numpy scalar to string
        if class_name != "SystematicCorrelationManager":
            msg = f"Invalid data format for SystematicCorrelationManager: {class_name}"
            raise ValueError(msg)

        # Create new instance
        manager = cls()

        manager.default_cor_length = int(data.get("default_cor_length", -1))
        manager.default_cor_strength = float(data.get("default_cor_strength", 1.0))

        # Restore correlation_groups (convert back to defaultdict)
        # Handle potential numpy arrays from HDF5
        manager.correlation_groups = defaultdict(list)
        for tag, group_list in data["correlation_groups"].items():
            # Ensure tag is a string
            tag_str = str(tag.item()) if isinstance(tag, np.ndarray) else str(tag)

            # Convert group_list items if they are numpy arrays
            processed_group_list = []
            for item in group_list:
                if isinstance(item, (list, tuple)) and len(item) == 4:
                    # Convert each element to proper type
                    obs_label = str(item[0].item()) if isinstance(item[0], np.ndarray) else str(item[0])
                    start_idx = int(item[1].item()) if isinstance(item[1], np.ndarray) else int(item[1])
                    end_idx = int(item[2].item()) if isinstance(item[2], np.ndarray) else int(item[2])
                    sys_name = str(item[3].item()) if isinstance(item[3], np.ndarray) else str(item[3])
                    processed_group_list.append((obs_label, start_idx, end_idx, sys_name))
                else:
                    processed_group_list.append(item)

            manager.correlation_groups[tag_str] = processed_group_list

        # Restore systematic_info with type conversion
        manager.systematic_info = {}
        for full_name, info_dict in data["systematic_info"].items():
            # Ensure all strings are proper strings, not numpy arrays
            full_name_str = str(full_name.item()) if isinstance(full_name, np.ndarray) else str(full_name)
            base_name = (
                str(info_dict["base_name"].item())
                if isinstance(info_dict["base_name"], np.ndarray)
                else str(info_dict["base_name"])
            )
            correlation_tag = (
                str(info_dict["correlation_tag"].item())
                if isinstance(info_dict["correlation_tag"], np.ndarray)
                else str(info_dict["correlation_tag"])
            )
            full_name_from_dict = (
                str(info_dict["full_name"].item())
                if isinstance(info_dict["full_name"], np.ndarray)
                else str(info_dict["full_name"])
            )
            _is_uncorrelated = (
                bool(info_dict["is_uncorrelated"].item())
                if isinstance(info_dict["is_uncorrelated"], np.ndarray)
                else bool(info_dict["is_uncorrelated"])
            )

            is_summed = info_dict.get("is_summed", False)
            cor_length = info_dict.get("cor_length", -1)
            cor_strength = info_dict.get("cor_strength", 1.0)

            # Handle numpy arrays from HDF5
            if isinstance(is_summed, np.ndarray):
                is_summed = bool(is_summed.item())
            if isinstance(cor_length, np.ndarray):
                cor_length = int(cor_length.item())
            if isinstance(cor_strength, np.ndarray):
                cor_strength = float(cor_strength.item())

            manager.systematic_info[full_name_str] = SystematicInfo(
                base_name=base_name,
                correlation_tag=correlation_tag,
                full_name=full_name_from_dict,
                is_summed=is_summed,
                cor_length=cor_length,
                cor_strength=cor_strength,
            )

        # Restore other attributes with type conversion
        manager.observable_systematics = {}
        for obs_label, sys_list in data["observable_systematics"].items():
            obs_label_str = str(obs_label.item()) if isinstance(obs_label, np.ndarray) else str(obs_label)
            sys_list_str = [str(item.item()) if isinstance(item, np.ndarray) else str(item) for item in sys_list]
            manager.observable_systematics[obs_label_str] = sys_list_str

        # Convert all_systematic_names to proper strings
        manager.all_systematic_names = [
            str(item.item()) if isinstance(item, np.ndarray) else str(item) for item in data["all_systematic_names"]
        ]

        # Restore pending correlation params (convert numpy arrays to strings)
        pending_params = data.get("_pending_correlation_params", {})
        manager._pending_correlation_params = {}
        for tag, param_string in pending_params.items():
            # Convert numpy arrays to strings
            tag_str = str(tag.item()) if isinstance(tag, np.ndarray) else str(tag)
            param_str = str(param_string.item()) if isinstance(param_string, np.ndarray) else str(param_string)
            manager._pending_correlation_params[tag_str] = param_str

        return manager
