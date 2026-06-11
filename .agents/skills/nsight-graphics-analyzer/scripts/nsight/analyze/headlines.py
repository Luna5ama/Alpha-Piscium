"""Headline metric selection rules.

The 'headline' set is what the agent sees first in summary.json. We pick by
substring match against NVIDIA PerfWorks-style dotted metric names, first
match wins. The substrings are deliberately stable across architectures so
the schema doesn't churn when moving Ada → Blackwell.
"""
from __future__ import annotations

import re
from typing import Optional

# (key, substring matched against the full metric name)
HEADLINE_METRIC_PATTERNS: list[tuple[str, str]] = [
    ("sm_throughput",    "sm__throughput"),
    ("sm_inst_executed", "sm__inst_executed_realtime"),
    ("warps_inactive",   "warps_inactive_sm_active"),
    ("l1_hit_rate",      "l1tex__t_sector_hit_rate"),
    ("l1_throughput",    "l1tex__throughput"),
]

# Used by analysis.throughput to rank the dominant subsystem.
ANALYSIS_THROUGHPUT_PATTERNS: list[tuple[str, str]] = [
    ("sm",       "sm__throughput"),
    ("l1tex",    "l1tex__throughput"),
    ("l2",       "lts__throughput"),
    ("dram",     "dramc__throughput"),
    ("pcie",     "pcie__throughput"),
]


def first_metric_matching(metric_names: list[str], substring: str) -> Optional[str]:
    pattern = re.compile(re.escape(substring), re.IGNORECASE)
    for name in metric_names:
        if pattern.search(name):
            return name
    return None


def headline_picks(metric_names: list[str]) -> dict[str, str]:
    """Resolve every HEADLINE_METRIC_PATTERNS entry to the first matching name."""
    picks: dict[str, str] = {}
    for key, substring in HEADLINE_METRIC_PATTERNS:
        match = first_metric_matching(metric_names, substring)
        if match:
            picks[key] = match
    return picks
