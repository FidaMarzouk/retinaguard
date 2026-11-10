"""Dagster code location."""

from dagster import Definitions

from retinaguard.pipeline.assets import DataPaths, asset_checks, assets
from retinaguard.pipeline.monitoring import MonitoringConfig, daily_drift, drift_job, drift_reports

defs = Definitions(
    assets=[*assets, drift_reports],
    asset_checks=asset_checks,
    jobs=[drift_job],
    schedules=[daily_drift],
    resources={"data": DataPaths(), "monitoring": MonitoringConfig()},
)
