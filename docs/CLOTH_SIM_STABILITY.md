# Cloth simulation stability (what fixed the “explosion / ball”)

This note records why the GPU cloth looked unstable or collapsed, and what changed to fix it.

## 1. Solver: clear λ each inner iteration + Gauss–Seidel with batches

XPBD uses a Lagrange multiplier λ per constraint (see Macklin et al., Eq. 17–18). The GPU solver uses **colored Gauss–Seidel**: constraints are sorted into **batches** (greedy edge coloring in `mesh_prep.rs`, **high-degree vertices first** to reduce the number of colors and thus `gs_edges` dispatches per inner iteration) so edges in one batch do not share particles; each batch runs in parallel via `gs_edges`, and batches run in order on a single **`jac_state`** buffer. Corrections apply immediately at both endpoints (**no Jacobi gather**).

**Historical issue (Jacobi era):** λ was accumulated across inner iterations wrong for parallel Jacobi, which interacted badly with stale positions.

**Current rule:** Zero λ (and Δλ storage) **before each inner iteration**:

- GPU: **`clear_constraint_lambda`** then **`gs_edges`** once per batch; **all inner iterations of a substep** share **one** `cloth_pass_distance_gauss_seidel` compute pass (many ordered dispatches; storage writes are visible to the next dispatch per WebGPU/WGSL rules). GS batches use **`binding(19)`** dynamic-uniform **`set_bind_group` offsets** (256 B lut; **`queue.write_buffer`** between dispatches is still unsafe).
- CPU reference (`xpbd_cpu.rs`): `lambda.fill(0.0)` at the start of each inner iteration, then sequential edge solves in **`constraint_*` row order** (same permutation as GPU).
- Parity harness: same pattern in `gpu_cpu_parity.rs`.

**Supporting tuning:** Stretch compliance stays near `DEFAULT_STRETCH_COMPLIANCE` (~`2e-8`) in `mesh_prep.rs`; `ClothSimUniforms::jacobi_omega` (default **`1.0`**) scales each distance correction (`gs_edges`). Lower it if needed; **`JACOBI_CORRECTION_CAP`** clamps each endpoint delta per constraint.

## 2. Rendering: correct particle index (the visual “ball”)

The simulation buffers could stay sane while the mesh looked like a **tiny ball**: every vertex was sampling the **same** slot in `sim_positions` (often index `0`).

**Fix (vertex shader, `assets/shaders/cloth_vertex.wgsl`):**

- Prefer **`vi = u32(vertex.uv_b.x + 0.5)`** when `VERTEX_UVS_B` is defined. The procedural mesh sets **`ATTRIBUTE_UV_1`** so `uv_b.x` is the particle index (see `ClothMeshData::to_bevy_mesh`).
- Otherwise fall back to **`vertex_index - mesh[instance].first_vertex_index`**, which matches Bevy’s indexed `draw_indexed` base vertex for slab‑allocated meshes.

Without a reliable index, the GPU draws garbage positions while compute still updates a full sheet.

## 3. CPU neighbor slices (mesh validation only)

Neighbor lists (**`neighbor_offsets`**, **`neighbor_other`**, **`neighbor_constraint_id`**) remain on `ClothMeshData` after **batch‑sorted** constraints for regression tests and tooling. The GPU no longer uploads them.

**Invariant:** Built with **cursor‑based scatter** in `finalize_cloth_mesh` (`mesh_prep.rs`), aligned with permutation that puts constraints in contiguous color‑batch spans.

**Regression test:** `cloth_neighbor_slices_match_constraints` in `cloth_compute.rs` (`simulation_data_tests`).

## Quick reference (files)

| Area | Location |
|------|----------|
| Inner‑loop λ clear + **`gs_edges`** (**`cloth_pass_distance_gauss_seidel`**) | `src/cloth_compute.rs` — `ClothSimNode` |
| Solve budget + fuse **`predict_copy_sim_to_jac`** + collision stride | `src/cloth_compute.rs` — `ClothSimConfig`, `ClothSimNode`; `assets/shaders/cloth_sim.wgsl` |
| GS kernel | `assets/shaders/cloth_sim.wgsl` — `gs_edges` |
| Inner‑loop λ clear + GS sweep (CPU) | `src/xpbd_cpu.rs` — `xpbd_substep_integrate` |
| Constraint coloring + offsets | `src/mesh_prep.rs` — `partition_constraints_for_gs_batches`, `constraint_batch_offsets` |
| Parity harness | `src/gpu_cpu_parity.rs` |
| Vertex particle index | `assets/shaders/cloth_vertex.wgsl` |
| Neighbor scatter + compliance defaults | `src/mesh_prep.rs` |
| Quad cross-diagonal (shear / zipper fix) | `src/mesh_prep.rs` — `DEFAULT_CROSS_DIAG_COMPLIANCE` |
| Self-collision strength (`coll_scale`; `0` = off) | `src/cloth_compute.rs` — `DEFAULT_COLL_SCALE`, `ClothSimUniforms` |

## Free-edge buckling (bottom hem “furling”)

Triangle meshes mainly constrain **adjacent** vertices. The **bottom boundary** has fewer neighbors and can enter short‑wavelength folds (curl / furl) under gravity + soft bending.

Mitigations in `mesh_prep.rs` (compatible with λ reset + GS batches):

1. **Pin the full top boundary** (all vertices at maximum `y`), not only corners — avoids asymmetric sag that loads the free edge harder.
2. **Stiffer hinge bending** — lower `DEFAULT_BEND_COMPLIANCE` resists sharp curling (`3.8` in code).
3. **Skip‑2 braces on full XY grids** (procedural + regular welded OBJ) — every **two** vertices along each row/column (`DEFAULT_SKIP2_COMPLIANCE` ~ triangle-edge scale; much softer values let the hem **shorten in X**). See `axis_aligned_grid_skip_two_distance_constraints`.
4. **Cross-diagonal constraints on every quad** — the diagonal **not** used as a triangle edge (`DEFAULT_CROSS_DIAG_COMPLIANCE`): removes shear “zipper” / alternating **Z** on the bottom edge (see `sim_positions.csv` diagnosis below).

**Gravity:** Integrated in **`predict_copy_sim_to_jac`**: `v += gravity * dt` (`cloth_sim.wgsl`). Default **`ClothSimUniforms.gravity`** ≈ **−9.81 on Y** only (scene‑scale metres).

**Shear “zipper” (CSV clue):** Quad → two triangles leaves **one diagonal unstretched** as an edge. Exported **`sim_positions`** after a frame often shows the bottom chain with **alternating ±Z** while **X** steps smoothly — missing **in-plane shear** stiffness, not primarily gravity. Fix: cross-diagonal constraints in `mesh_prep.rs` with **`DEFAULT_CROSS_DIAG_COMPLIANCE`** typically **stiffer** than triangle stretch (raise **`INNER_ITERS`** if one-frame CSV still shows residual ±Z).

## If it misbehaves again

- Increase **`SUBSTEPS`** / **`INNER_ITERS`** or lower **`jacobi_omega`** before chasing buffer bugs.
- Confirm the vertex path sees **`VERTEX_UVS_B`** when using UV1 particle ids (material + mesh layout).
- **Self-collision:** default strength is **`DEFAULT_COLL_SCALE` (0.38)** in `src/cloth_compute.rs`; set `ClothSimUniforms.coll_scale` to **`0`** to disable while debugging.
- Re‑run **`cargo test`** — GPU/CPU parity and `cloth_neighbor_slices_match_constraints` guard the worst regressions.

## Example mesh sources

The `cloth_xpbd` example loads welded `assets/cloth.obj` via `parse_welded_obj` in `mesh_prep.rs`; if the file is missing, it falls back to `grid_cloth_hanging`.
