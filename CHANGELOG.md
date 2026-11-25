# Unreleased

**Added**

- Initial workflows ported over time dolphin

**Changed**

- Opt-in `ionosphere_options.method: main_diff` estimates the ionosphere from
  frequency A and A*conj(B), for historical GSLC stacks. The default `main_side`
  keeps the split-spectrum workflow.

**Fixed**

- Split-spectrum corrections were scaled by lambda/4pi after already being written
  in meters, leaving the ionosphere layer about 52 times too small.
- The validity mask is now built for runs without an ancillary water mask.
