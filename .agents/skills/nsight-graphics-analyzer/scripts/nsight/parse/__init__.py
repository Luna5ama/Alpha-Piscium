"""TSV bundle parsers for Nsight `--auto-export` artifacts.

The bundle lives at `<session>/BASE/` and contains five files:

    REPRO_INFO.xls       hardware/driver/parameter KV pairs
    FRAME.xls            single row of per-frame GPU durations (ms)
    GPUTRACE_FRAME.xls   one row per metric, per-frame averages
    D3DPERF_EVENTS.xls   marker tree (8-space indent) + per-instance durations
    GPUTRACE_REGIMES.xls per-marker x per-metric x per-frame dense matrix (~300+ MB)

All `.xls` files are CRLF + tab-separated text, NOT Excel binary. REGIMES is
streamed with column projection — never load it whole.
"""
