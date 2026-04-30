# Cloth simulation stability (what fixed the “explosion / ball”)

This note records why the GPU cloth looked unstable or collapsed, and what changed to fix it.

## 1. Solver: clear λ every Jacobi inner iteration (main physics fix)

XPBD uses a Lagrange multiplier λ per constraint (see Macklin et al., Eq. 17–18). Our solver uses a **parallel Jacobi** pattern: one pass over edges (`jacobi_edges`) to compute Δλ, then a particle gather (`jacobi_gather`) to apply position corrections.

**Problem:** λ was accumulated across **multiple inner iterations** within the same substep (λ cleared once per substep, then updated each edge pass). That kind of λ warm‑start is closer to **sequential** Gauss–Seidel behavior. With **Jacobi**, neighboring constraints see stale positions while λ keeps growing iteration‑to‑iteration, which often **oscillates or blows up** on dense cloth—even with self‑collision off.

**Fix:** Zero λ (and Δλ) **before each inner iteration**:

- GPU: dispatch `clear_constraint_lambda` inside the `for k in 0..INNER_ITERS` loop in `cloth_compute.rs` (render graph), not only once per substep.
- CPU reference: `lambda.fill(0.0)` and `delta_lambda.fill(0.0)` at the start of each inner iteration in `xpbd_cpu.rs`.
- Parity harness: the same inner‑loop clear in `gpu_cpu_parity.rs` so tests stay aligned with the render graph.

Each inner iteration then behaves like a fresh XPBD correction step from the current Jacobi snapshot, which is much better behaved for this parallel scheme.

**Supporting tuning:** Stretch compliance was eased slightly (`DEFAULT_STRETCH_COMPLIANCE` in `mesh_prep.rs`, currently `2e-8`) so Jacobi is not fighting unrealistically stiff edges with a modest iteration count.

## 2. Rendering: correct particle index (the visual “ball”)

The simulation buffers could stay sane while the mesh looked like a **tiny ball**: every vertex was sampling the **same** slot in `sim_positions` (often index `0`).

**Fix (vertex shader, `assets/shaders/cloth_vertex.wgsl`):**

- Prefer **`vi = u32(vertex.uv_b.x + 0.5)`** when `VERTEX_UVS_B` is defined. The procedural mesh sets **`ATTRIBUTE_UV_1`** so `uv_b.x` is the particle index (see `ClothMeshData::to_bevy_mesh`).
- Otherwise fall back to **`vertex_index - mesh[instance].first_vertex_index`**, which matches Bevy’s indexed `draw_indexed` base vertex for slab‑allocated meshes.

Without a reliable index, the GPU draws garbage positions while compute still updates a full sheet.

## 3. Neighbor layout for Jacobi gather (data structure invariant)

Jacobi gather uses **`neighbor_offsets`** so each particle owns a **contiguous** slice of `neighbor_other` / packed GPU rows. Those entries must list **only** constraints incident on that particle, with the matching **`neighbor_constraint_id`**.

**Invariant:** Built with **cursor‑based scatter** in `finalize_cloth_mesh` (`mesh_prep.rs`), not by appending all edges in global constraint order.

**Regression test:** `cloth_neighbor_slices_match_constraints` in `cloth_compute.rs` (`simulation_data_tests`).

## Quick reference (files)

| Area | Location |
|------|----------|
| Inner‑loop λ clear (GPU) | `src/cloth_compute.rs` — render graph Jacobi loop |
| Inner‑loop λ clear (CPU) | `src/xpbd_cpu.rs` — `xpbd_substep_integrate` |
| Parity harness | `src/gpu_cpu_parity.rs` — `run_one_gpu_substep` |
| Vertex particle index | `assets/shaders/cloth_vertex.wgsl` |
| Neighbor scatter + compliance defaults | `src/mesh_prep.rs` |
| Quad cross-diagonal (shear / zipper fix) | `src/mesh_prep.rs` — `DEFAULT_CROSS_DIAG_COMPLIANCE` |
| Self-collision strength (`coll_scale`; `0` = off) | `src/cloth_compute.rs` — `DEFAULT_COLL_SCALE`, `ClothSimUniforms` |

## Free-edge buckling (bottom hem “furling”)

Triangle meshes mainly constrain **adjacent** vertices. The **bottom boundary** has fewer neighbors and can enter short‑wavelength folds (curl / furl) under gravity + soft bending.

Mitigations in `mesh_prep.rs` (compatible with the Jacobi λ reset described above):

1. **Pin the full top boundary** (all vertices at maximum `y`), not only corners — avoids asymmetric sag that loads the free edge harder.
2. **Stiffer hinge bending** — lower `DEFAULT_BEND_COMPLIANCE` resists sharp curling (`3.8` in code).
3. **Skip‑2 braces on full XY grids** (procedural + regular welded OBJ) — every **two** vertices along each row/column (`DEFAULT_SKIP2_COMPLIANCE` ~ triangle-edge scale; much softer values let the hem **shorten in X** under Jacobi). See `axis_aligned_grid_skip_two_distance_constraints`.
4. **Cross-diagonal constraints on every quad** — the diagonal **not** used as a triangle edge (`DEFAULT_CROSS_DIAG_COMPLIANCE`): removes shear “zipper” / alternating **Z** on the bottom edge (see `sim_positions.csv` diagnosis below).

**Gravity:** Integrated in **`predict`**: `v += gravity * dt` (`cloth_sim.wgsl`). Default **`ClothSimUniforms.gravity`** ≈ **−9.81 on Y** only (scene‑scale metres).

**Shear “zipper” (CSV clue):** Quad → two triangles leaves **one diagonal unstretched** as an edge. Exported **`sim_positions`** after a frame often shows the bottom chain with **alternating ±Z** while **X** steps smoothly — missing **in-plane shear** stiffness, not primarily gravity. Fix: cross-diagonal constraints in `mesh_prep.rs` with **`DEFAULT_CROSS_DIAG_COMPLIANCE`** typically **stiffer** than triangle stretch so Jacobi freezes shear sooner (raise **`INNER_ITERS`** if one-frame CSV still shows residual ±Z).

## If it misbehaves again

- Increase **`SUBSTEPS`** / **`INNER_ITERS`** or lower **`jacobi_omega`** before chasing buffer bugs.
- Confirm the vertex path sees **`VERTEX_UVS_B`** when using UV1 particle ids (material + mesh layout).
- **Self-collision:** default strength is **`DEFAULT_COLL_SCALE` (0.38)** in `src/cloth_compute.rs`; set `ClothSimUniforms.coll_scale` to **`0`** to disable while debugging.
- Re‑run **`cargo test`** — GPU/CPU parity and `cloth_neighbor_slices_match_constraints` guard the worst regressions.

## Example mesh sources

The `cloth_xpbd` example loads welded `assets/cloth.obj` via `parse_welded_obj` in `mesh_prep.rs`; if the file is missing, it falls back to `grid_cloth_hanging`.
