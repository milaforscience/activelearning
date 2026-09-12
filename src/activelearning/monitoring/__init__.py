"""Monitoring pipeline for active-learning round outputs.

Monitoring combines core metrics, timing information, and optional diagnostics
for each completed round. It sends that shared output to independent sinks:
``Logger`` for live telemetry and ``RunWriter`` for durable structured records.
Public APIs live in their focused submodules to keep imports explicit and avoid
import cycles.
"""
