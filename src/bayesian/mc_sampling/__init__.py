"""
Sampling module for Bayesian Inference.

This module provides functionality to compute posterior for a given analysis run

The main functionalities are:
 - run_mcmc() performs MCMC and returns posterior
 - credible_interval() compute credible interval for a given posterior

A configuration class MCMCConfig provides simple access to emulation settings

Based in part on JETSCAPE/STAT code.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

from bayesian.mc_sampling.base import (  # noqa: F401
    MCConfig,
    credible_interval,
    run_mcmc,
)
