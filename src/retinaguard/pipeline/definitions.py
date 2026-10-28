"""Dagster code location."""

from dagster import Definitions

from retinaguard.pipeline.assets import DataPaths, asset_checks, assets

defs = Definitions(assets=assets, asset_checks=asset_checks, resources={"data": DataPaths()})
