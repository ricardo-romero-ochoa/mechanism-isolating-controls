# Release audit

Public release prepared 2026-10-29.

Validation performed before release:

- `python verify_results.py`: **48/48 checks passed**.
- `make selftest`: all regulatory and protein utility self-tests passed.
- `make figures`: all four figures regenerated successfully from the committed compact reference outputs.
- Python source compilation and shell-script syntax checks passed.
- The portable JSON occupancy artifact was materialized back to the expected `(8, 8, 4)` NumPy matrix and the downstream regulatory self-test passed.
- Release-hygiene scan found no Cell Systems/Perspective/rewrite language, local absolute filesystem paths, credentials, or downloaded raw-data caches.

The full regulatory and protein reruns require network downloads and substantial compute; they were not rerun as part of the lightweight public-release audit. Their scripts, fixed settings, quality gates, and machine-readable reference outputs are included.
