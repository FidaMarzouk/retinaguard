"""Monitoring assets: a drift report per site and day of stored predictions, refreshed daily.

Configured with ``DATABASE_URL`` (the prediction store), ``DRIFT_REFERENCE`` (the reference
written by ``python -m retinaguard.monitoring.batches reference``) and ``DRIFT_HTML_DIR`` (where
the Evidently report of each batch is written).
"""

import json
import os
from pathlib import Path

import dagster as dg

from retinaguard.monitoring.batches import report
from retinaguard.serving.store import PredictionStore


class MonitoringConfig(dg.ConfigurableResource):  # type: ignore[type-arg]
    database_url: str = os.environ.get("DATABASE_URL", "sqlite:///outputs/retinaguard.db")
    reference: str = os.environ.get("DRIFT_REFERENCE", "outputs/models/reference.json")
    html_dir: str = os.environ.get("DRIFT_HTML_DIR", "outputs/drift_reports")


@dg.asset(group_name="monitoring")
def drift_reports(
    context: dg.AssetExecutionContext, monitoring: MonitoringConfig
) -> dg.MaterializeResult:  # type: ignore[type-arg]
    """Drift report of every (site, day) batch in the prediction store, against the reference."""
    reference = json.loads(Path(monitoring.reference).read_text(encoding="utf-8"))
    reports = report(PredictionStore(monitoring.database_url), reference, Path(monitoring.html_dir))
    alerted = reports if reports.empty else reports[reports["alert"].notna()]
    for _, row in alerted.iterrows():
        context.log.warning(f"drift alert {row['day']} {row['site']}: {row['alert']}")
    return dg.MaterializeResult(metadata={"batches": len(reports), "alerts": len(alerted)})


drift_job = dg.define_asset_job("drift_report", selection=[drift_reports])
daily_drift = dg.ScheduleDefinition(job=drift_job, cron_schedule="0 6 * * *")
