# CalibrationTests.jl

Hypothesis tests of calibration.

[![Build Status](https://github.com/devmotion/CalibrationTests.jl/workflows/CI/badge.svg?branch=main)](https://github.com/devmotion/CalibrationTests.jl/actions?query=workflow%3ACI+branch%3Amain)
[![DOI](https://zenodo.org/badge/215970266.svg)](https://zenodo.org/badge/latestdoi/215970266)
[![Codecov](https://codecov.io/gh/devmotion/CalibrationTests.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/devmotion/CalibrationTests.jl)
[![Coveralls](https://coveralls.io/repos/github/devmotion/CalibrationTests.jl/badge.svg?branch=main)](https://coveralls.io/github/devmotion/CalibrationTests.jl?branch=main)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

**There are also [Python](https://github.com/devmotion/pycalibration) and [R](https://github.com/devmotion/rcalibration) interfaces for this package**

## Overview

This package implements different hypothesis tests for calibration of
probabilistic models in the Julia language.

## Related packages

The statistical tests in this package are based on the calibration error estimators
in the package [CalibrationErrors.jl](https://github.com/devmotion/CalibrationErrors.jl).

[pycalibration](https://github.com/devmotion/pycalibration) is a Python interface for CalibrationErrors.jl and CalibrationTests.jl.

[rcalibration](https://github.com/devmotion/rcalibration) is an R interface for CalibrationErrors.jl and CalibrationTests.jl.

## Citing

If you use CalibrationsTests.jl as part of your research, teaching, or other activities, please consider citing the following publications:

Widmann, D., Lindsten, F., & Zachariah, D. (2019). [Calibration tests in multi-class
classification: A unifying framework](https://proceedings.neurips.cc/paper/2019/hash/1c336b8080f82bcc2cd2499b4c57261d-Abstract.html). In
*Advances in Neural Information Processing Systems 32 (NeurIPS 2019)* (pp. 12257–12267).

Widmann, D., Lindsten, F., & Zachariah, D. (2021).
[Calibration tests beyond classification](https://openreview.net/forum?id=-bxf89v3Nx).
To be presented at *ICLR 2021*.

## Acknowledgements

This work was financially supported by the Swedish Research Council via the projects *Learning of Large-Scale Probabilistic Dynamical Models* (contract number: 2016-04278), *Counterfactual Prediction Methods for Heterogeneous Populations* (contract number: 2018-05040), and *Handling Uncertainty in Machine Learning Systems* (contract number: 2020-04122), by the Swedish Foundation for Strategic Research via the project *Probabilistic Modeling and Inference for Machine Learning* (contract number: ICA16-0015), by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation, and by ELLIIT.
