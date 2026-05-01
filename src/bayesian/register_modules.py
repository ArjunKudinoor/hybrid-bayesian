"""Register modules in a directory

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ValidationFunction(Protocol):
    def __call__(self, name: str, module: Any) -> None: ...


def validation_noop(name: str, module: Any) -> None: ...


def discover_and_register_modules(
    calling_module_name: Any,
    required_attributes: list[str],
    validation_function: ValidationFunction | None = None,
    fail_on_failed_validation: bool = True,
) -> dict[str, ModuleType]:
    """Automatically discovery and registration of modules in the directory of the calling module.

    Modules in the directory indicate they should be registered by defining a '_register_name' attribute.
    Such modules are then validated and passed to the calling modules.

    Args:
        calling_module_name: `__name__` attribute of the module where this is being called.
        required_attributes: List of attributes which are required to exist in the modules
            that will be registered.
        validation_function: Function that allows for generic validation of a module to be
            registered. Any issues are indicated by raising exceptions.
        fail_on_failed_validation: If True, this discovery and registration will fail if
            the validation of the any of the modules fails. Default: True.

    Returns:
        A dictionary where the keys are names the modules are supposed to be registered under,
            and the values are the modules themselves.
    """
    # Setup
    registered_modules: dict[str, Any] = {}
    if validation_function is None:
        validation_function = validation_noop
    # Each module that we're interested in must contain a _register_name.
    # It will optionally contain further attributes, which we can validate here.
    register_attribute = "_register_name"
    required_attributes = [*required_attributes, register_attribute]

    # To simplify the interface for the calling module, we will retrieve the module (as determined by the module name)
    # and then we can grab the necessary attributes ourselves.
    calling_module = sys.modules[calling_module_name]
    # In particular, we need to know:
    # 1. Where to look
    calling_module_file_path = Path(getattr(calling_module, "__file__", ""))
    package_dir = calling_module_file_path.parent
    # 2. The package that we're working in, so we can successfully load the modules.
    calling_module_package = getattr(calling_module, "__package__", __package__)

    # Scan for all modules in the current directory
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        # Import the module
        module = importlib.import_module(f".{module_info.name}", calling_module_package)

        # The module must have the appropriate attributes to be considered.
        # First, it signals that it might be of interest by having the register_attribute
        if not hasattr(module, register_attribute):
            logger.info(f"Skipping module {module_info.name}")
            continue

        has_required_attributes = [hasattr(module, _attr) for _attr in required_attributes]
        if not all(has_required_attributes):
            msg = f"Requested module {module_info.name}, but missing attributes: {[v for v, has in zip(required_attributes, has_required_attributes, strict=True) if not has]}"
            raise ValueError(msg)
        if all(has_required_attributes):
            # We know that it has this attribute at this point, so we're safe to request it.
            name = module._register_name
            # Validate the module. It is expected to raise an exception if it cannot be validated.
            try:
                validation_function(name=name, module=module)
            except Exception as e:
                if fail_on_failed_validation:
                    msg = f"Failed validation of module {module_info.name} under name '{name}'..."
                    raise e from ValueError(msg)
                logger.exception(e)

            logger.info(f"Registering module {name}")
            registered_modules[name] = module

    return registered_modules
