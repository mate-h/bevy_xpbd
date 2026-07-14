# Exporting an Instruments trace with `xctrace`

This note lives in the **bevy_softbody** repo as a reference for Apple **Instruments** captures (e.g. **Metal System Trace**). It does **not** ship trace data: you export XML **from your `.trace` bundle** into that bundle’s own `exported/` folder (or any path you choose).

An Instruments **`.trace`** file is usually a **package directory** (Show Package Contents in Finder). Pass that directory to `--input`—the extension may be `.trace` or omitted (e.g. `~/Desktop/trace`).

**Typical outputs** (after the commands below):

- `exported/trace_toc.xml` — table of contents / run metadata  
- `exported/metal_gpu_intervals.xml` — GPU interval rows (often tens of MB)  
- `exported/metal_shader_profiler_intervals.xml` — shader profiler table (may be schema-only on some templates)

Replace `PATH_TO_TRACE` with your bundle path. Example: `/Users/mateh/Desktop/trace`.

## Table of contents (all tables / schemas)

```bash
mkdir -p PATH_TO_TRACE/exported
xctrace export --input "PATH_TO_TRACE" \
  --toc \
  --output "PATH_TO_TRACE/exported/trace_toc.xml"
```

## Metal GPU intervals (pass / channel durations)

Used for GPU timeline slices (Vertex, Fragment, Compute, etc.) and labels such as render pass names.

```bash
xctrace export --input "PATH_TO_TRACE" \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
  --output "PATH_TO_TRACE/exported/metal_gpu_intervals.xml"
```

## Shader timeline intervals (optional; may contain schema only—no rows—on some captures)

```bash
xctrace export --input "PATH_TO_TRACE" \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-shader-profiler-intervals"]' \
  --output "PATH_TO_TRACE/exported/metal_shader_profiler_intervals.xml"
```

## Relative path variant

From **inside** the trace bundle directory:

```bash
mkdir -p exported
xctrace export --input . --toc --output exported/trace_toc.xml
xctrace export --input . \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
  --output exported/metal_gpu_intervals.xml
xctrace export --input . \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-shader-profiler-intervals"]' \
  --output exported/metal_shader_profiler_intervals.xml
```

## Notes

- Open **`exported/trace_toc.xml`** and check `<run number="…">` if a capture has multiple runs; adjust the `run[@number="1"]` segment in the `--xpath` if needed.  
- **`metal_gpu_intervals.xml`** can be very large; summing interval durations across the whole file **double-counts** overlapping GPU work—use shares within one trace or Instruments’ own views for wall-time interpretation.  
- Rows often use **`ref="…"`** on `<gpu-channel-name>` (and other fields); scripts should resolve those IDs to classify **Compute** vs other channels.
