# Input-validation error codes

All input-stack prechecks in `disp_nisar.main` and `disp_nisar.pge_runconfig`
raise `disp_nisar._log.InputValidationError` — a `ValueError` subclass carrying
a numeric `.error_code` so the PGE can map failures to a specific cause. These
checks run *before* the expensive `dolphin` workflow starts, so a bad input
list fails in seconds instead of after minutes-to-hours of processing.

Unlike disp-s1, a DISP-NISAR run is a single frame (not many bursts), so every
check below is evaluated on the one GSLC stack rather than per burst.

## Turning the prechecks off

Every numbered check below is gated by one runconfig field, on by default:

```yaml
disp_nisar_workflow:
  run_input_prechecks: true   # default; set false to skip 1000-2001
```

Leave it on. It exists as an operational escape hatch -- forcing a run whose
inputs a precheck rejects -- and set to `false` it skips all five checks, so a
malformed input GSLC list fails deep inside `dolphin` (see 1001) or not at
all. A runconfig that predates the field loads with the prechecks on.

Two details worth knowing:

- 2001 is the only check that runs in `RunConfig.to_workflow` rather than
  `disp_nisar.main.run`. It is gated by withholding `cslc_file_list` from
  `_create_forward_mode_network`, so the interferogram network built is
  identical either way -- only the depth assertion is skipped.
- The duplicate-date check (`_assert_no_duplicate_dates`, an uncoded
  `ValueError`) and the compressed-chain check
  (`_assert_compressed_slcs_consecutive`, also an uncoded `ValueError`) are
  *not* gated. They flag ambiguities no later step can resolve, so they always
  run.

## Summary table

| Code | Function | Applies to | Condition |
|---|---|---|---|
| 1000 | `_assert_no_large_temporal_gaps` (`main.py`) | both modes | consecutive real GSLC dates are more than `max_gap_days` (default 2 years) apart |
| 1001 | `_assert_no_compressed_slc_conflicts` (`main.py`) | both modes, whenever a CCSLC is present | a real GSLC's date exactly equals the most recent CCSLC's reference date |
| 1002 | `_assert_no_compressed_slc_conflicts` (`main.py`) | both modes, whenever a CCSLC is present — boundary is mode-dependent | a CCSLC's reference date is later than allowed (see below) |
| 2000 | `_assert_forward_mode_compressed` (`main.py`) | forward mode only | no CCSLC present at all |
| 2001 | `_create_forward_mode_network` (`pge_runconfig.py`) | forward mode only | fewer than `nearest_n + 1` **real** GSLCs (compressed SLCs don't count toward this depth) |

Numbering convention: **1xxx** = checks that apply regardless of
`product_type`; **2xxx** = forward-mode-only checks.

## 1000 — temporal gap

```
Temporal gap(s) exceeding the {N}-year limit for input GSLCs:
{gap}-day ({gap:.2f}-year) gap between {earlier} and {later}
```

(The limit is rendered in days rather than years when `max_gap_days` is set
below one year, so a short threshold doesn't report itself as a "0-year
limit".)

A multi-year gap between consecutive real GSLC dates destroys interferometric
coherence and almost always indicates a malformed input list (missing files,
wrong frame). Compressed SLCs are excluded — only real-to-real gaps are
measured. (This was a warning in earlier versions of disp-nisar.)

**Fix**: supply the missing intermediate GSLC(s), or confirm the gap is
intentional and raise `max_gap_days` when calling the check.

## 1001 — real GSLC overlaps its CCSLC's reference date

```
Input GSLC list has a real GSLC on {date}, the same date as the most recent
CCSLC reference date. The CCSLC already represents that epoch, so no real
GSLC for the same date should also be included.
```

The compressed SLC's reference date *is* that epoch already — a real GSLC for
the same date is ambiguous with it. Uncaught, this crashes inside `dolphin`'s
ministack construction with a `pydantic.ValidationError` raised by
`dolphin.stack.BaseStack._check_no_date_overlap`. That happens in both modes,
after mask extraction, PS pixels and EMI setup, so the setup time is wasted
before the failure.

The check is conditional on a CCSLC being in the input list: with none
present, it returns immediately. A path that merely contains "compressed"
(e.g. a real GSLC under a directory of that name) yields fewer than three
dates and is not treated as a CCSLC.

## 1002 — CCSLC reference date later than allowed

Forward mode:

```
CCSLC reference date is later than the latest real GSLC (the latest
acquisition must always be real): CCSLC reference date {ref}, real GSLC date
{boundary}
```

Historical mode:

```
CCSLC reference date is later than the earliest real GSLC (historical mode
outputs a product for every real date, so CCSLCs must cover only prior
history): CCSLC reference date {ref}, real GSLC date {boundary}
```

A compressed SLC must summarize *prior* history — it should never reach into
(or past) the real dates it is paired with in the same run. The boundary
depends on `product_type`:

- **`DISP_NISAR_FORWARD`**: the CCSLC's reference date must predate the
  **latest** real GSLC date. Forward mode only outputs one product (for the
  latest date), so earlier real dates in the batch are just context.
- **Any other `product_type`** (historical): the CCSLC's reference date must
  predate the **earliest** real GSLC date. Historical mode outputs one product
  per new real date, so a CCSLC reference falling between two new dates would
  count the earlier one twice — once as real data, once as part of the
  CCSLC's claimed history.

## 2000 — forward mode requires a CCSLC

```
Forward mode requires at least one compressed GSLC (CCSLC) in the input GSLC
list, but none was found.
```

Forward mode is incremental — it must be handed at least one CCSLC to process
against. This check does **not** apply to historical mode, where zero CCSLCs
is the normal case.

**Fix**: supply the CCSLC(s) summarizing the prior ministack.

## 2001 — insufficient stack depth for the forward-mode network

```
Forward mode nearest-{N} network requires at least {N+1} real GSLCs (CCSLCs
don't count toward this depth), but found only {count}.
```

Forward mode's manual-index network (nearest-3 or nearest-4, from
`forward_mode_network_size`) reaches back `nearest_n + 1` positions from the
latest date — but only into `dolphin`'s **real-date-only** phase-linked list.
Compressed SLCs are globbed separately and never enter it, so the required
depth is counted in real GSLCs alone, however many CCSLCs are also present.
Counting CCSLCs toward the depth would produce an `IndexError` in
`dolphin.interferogram._make_ifg_pairs` instead of failing at input
validation.

**Fix**: include more real trailing dates (`nearest_n + 1`), or lower
`forward_mode_network_size` (valid range 3–4, schema-constrained).

## Related: forward-mode reference (`forward_redo_reference`)

Not an input check, but it changes the forward-mode network, so it is noted
here. `forward_redo_reference` (algorithm parameter, default `true`) controls
how a forward product is referenced:

- `true`: the product is re-referenced to the second-to-last date, making it a
  single-step increment (the historical disp-nisar behavior).
- `false`: the product stays referenced to the compressed-SLC epoch, as
  historical mode does. The network gains one interferogram from that epoch to
  the earliest in-window date (dolphin's
  `InterferogramNetwork.compressed_reference_anchor`), so the epoch is a node
  of the unwrapped network, and the re-reference step is skipped. This needs a
  dolphin release that has that option (v0.43.0 is pinned).

`compressed_reference_anchor` cannot be set directly in the algorithm
parameters; it is derived from `forward_redo_reference`.
