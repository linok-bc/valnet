"""Runway yaw and cross-track from a forward camera, end to end.

    frames -> DeepLabV3+ runway masks (segment) -> Samuel Moyal's mask-fit estimator with a
    crab-angle EKF (estimate) -> per-frame yaw and cross-track (pipeline) -> replay (viz)

Command line: `python -m e2e_runway_dof_estimation.demo --flight weather_000185`;
web app: `python -m e2e_runway_dof_estimation.app`.
"""

__version__ = "0.1.0"
