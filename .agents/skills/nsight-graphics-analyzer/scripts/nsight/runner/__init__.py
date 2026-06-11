"""Subprocess wrappers that build ngfx command lines and run them safely.

`invoke.py` owns the runtime concerns (timeout, process-tree teardown,
shutdown-crash salvage). The other modules build argv lists declaratively;
they perform no I/O.
"""
