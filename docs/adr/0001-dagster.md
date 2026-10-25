# 0001: Dagster for data orchestration

- Status: accepted
- Date: 2026-10-23

## Context
The data pipeline turns six public datasets into a single manifest, deduplication report, processed images and frozen splits. Each output is a dataset artefact, and the hard invariants in PLAN.md §7.2 must be checked every time those artefacts change. The pipeline runs locally on a laptop CPU.

## Decision
Use Dagster with software-defined assets, partitioned by `source`, with the §7.2 invariants as asset checks (mirrored by unit tests).

## Consequences
- Asset lineage and check results are visible in the Dagster UI and can feed the dashboard's data-health page.
- Airflow and Prefect were rejected: they are task-centric, and Airflow needs more infrastructure than a laptop project warrants.
- Dagster is a large dependency, so it lives in the `pipeline` extra and stays out of the API and app images.
