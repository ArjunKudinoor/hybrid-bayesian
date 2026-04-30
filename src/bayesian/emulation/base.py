"""Base functionality needed for implementing an emulator.

This is **NOT** for the user, but rather for developers specifying how to
interact with an individual emulator. The user interface is implemented
in `interface`.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import attrs

from bayesian import analysis, data_IO

logger = logging.getLogger(__name__)


@attrs.define
class BaseEmulatorSettings:
    """Base (i.e. shared) settings for an emulator.

    Store this class in your specialized emulator settings class.
    Composition is preferred to inheritance.

    Attributes:
        force_retrain: If true, force the emulator to retrain.
        _settings: Dict from the YAML config corresponding to the emulator settings.
            It's included here since we need access for derived properties, but it's marked
            as private since we want to encourage access through the specialized emulator settings.
        observable_filter: Class to handle filtering down to only observables that are relevant for
            the emulator. The class itself is cached to minimize computation.
    """

    force_retrain: bool = attrs.field()
    _settings: dict[str, Any] = attrs.field()
    # TODO(RJE): Does this really belong here? Not sure...
    _observable_filter: data_IO.ObservableFilter | None = attrs.field(init=False)

    @classmethod
    def from_emulator_settings(cls, emulator_settings: dict[str, Any]) -> BaseEmulatorSettings:
        """Initialize the base emulator settings from a emulator settings."""
        return cls(
            force_retrain=emulator_settings["force_retrain"],
            settings=emulator_settings,
        )

    @property
    def observable_filter(self) -> data_IO.ObservableFilter | None:
        if self._observable_filter is not None:
            return self._observable_filter
        # Observable filter
        self._observable_filter = None
        observable_list = self._settings.get("observable_list", [])
        observable_exclude_list = self._settings.get("observable_exclude_list", [])
        if observable_list or observable_exclude_list:
            self._observable_filter = data_IO.ObservableFilter(
                include_list=observable_list,
                exclude_list=observable_exclude_list,
            )
        return self.observable_filter


@runtime_checkable
class EmulatorSettings(Protocol):
    """Emulator settings protocol

    Attributes:
        emulator_name: Name of the emulator. Must match the name under which the
            emulator module is registered.
        base_settings: Base emulator settings, which are shared across emulators.
        settings: Dictionary containing the full emulator configuration.
        additional_name: More specific name for the emulator. Default: "" (e.g. empty,
            so we'll omit it)
    """

    emulator_name: ClassVar[str]
    base_settings: BaseEmulatorSettings
    settings: dict[str, Any]
    # More specific name
    additional_name: str = ""


@attrs.define
class IO:
    """Methods related to emulator IO.

    All methods are static, but it's useful to group them together.
    """

    @staticmethod
    def output_filename(emulator_settings: EmulatorSettings, analysis_settings: analysis.AnalysisSettings) -> Path:
        """Determine output filename based on emulator and analysis settings.

        Args:
            emulator_settings: Emulator settings.
            analysis_settings: Overall analysis settings.
        Returns:
            Output filename.
        """
        filename = "emulator.pkl"
        if emulator_settings.additional_name:
            filename = f"emulator_{emulator_settings.additional_name}.pkl"
        return analysis_settings.output_dir / filename

    @staticmethod
    def read_emulator(
        emulator_settings: EmulatorSettings, analysis_settings: analysis.AnalysisSettings
    ) -> dict[str, Any]:
        """Read emulator output from file.

        Args:
            emulator_settings: Emulator settings.
            analysis_settings: Analysis settings.
        Returns:
            Emulator output.
        """
        filename = IO.output_filename(emulator_settings=emulator_settings, analysis_settings=analysis_settings)

        with filename.open("rb") as f:
            results: dict[str, Any] = pickle.load(f)
        return results

    @staticmethod
    def write_emulator(
        emulator_output: dict[str, Any],
        emulator_settings: EmulatorSettings,
        analysis_settings: analysis.AnalysisSettings,
    ) -> None:
        """Write emulator to file.

        Args:
            emulator_output: Output from an emulator to store.
            emulator_settings: Emulator settings.
            analysis_settings: Analysis settings.
        Returns:
            None.
        """
        filename = IO.output_filename(emulator_settings=emulator_settings, analysis_settings=analysis_settings)

        with filename.open("wb") as f:
            pickle.dump(emulator_output, f)
