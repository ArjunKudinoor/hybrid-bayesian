"""
Sampling module for Bayesian Inference.

This module provides functionality to compute the posterior for a given analysis run.

The main functionalities are:
 - run_mcmc() performs MCMC and returns posterior
 - credible_interval() computes a credible interval for a given posterior
 - map_parameters() computes MAP parameters from a posterior

A configuration class MCConfig provides simple access to MCMC settings.

Based in part on JETSCAPE/STAT code.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
"""

from __future__ import annotations

from bayesian.mc_sampling.base import (  # noqa: F401
    MCConfig,
    credible_interval,
    map_parameters,
    run_mcmc,
)
