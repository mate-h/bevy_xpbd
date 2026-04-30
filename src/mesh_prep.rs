//! Welded OBJ parsing and procedural grid cloth (topology, edges, bending pairs, invMass, pins).
//!
//! Neighbor arrays for Jacobi gather must stay cursor-scattered by particle range — see **`docs/CLOTH_SIM_STABILITY.md`**.

use bevy::math::{Vec2, Vec3};
use std::collections::HashMap;

/// Bending constraints (hinge “opposite vertices”): lower α = **stiffer** bend, less edge furling / curl on free boundaries.
/// Too soft → bottom hem bows inward (curved rest shape) as hinge chains relax under gravity.
pub const DEFAULT_BEND_COMPLIANCE: f32 = 3.8;
/// Structural (triangle edge) constraints: tiny α improves Jacobi XPBD conditioning vs `0.0`,
/// which with few iterations leaves large length errors and visible blow-ups.
/// Stiff enough for a hanging sheet; tiny α with Jacobi still fights convergence — don’t go far below ~1e‑8 without more inner iterations.
pub const DEFAULT_STRETCH_COMPLIANCE: f32 = 2e-8;
/// Skip‑2 braces on regular grids (`grid_cloth_hanging` + detected OBJ grids): every-other links along rows/cols.
/// Match triangle-edge stiffness order (`≈` [`DEFAULT_STRETCH_COMPLIANCE`]): much softer skip‑2 lets the bottom chain **compress
/// in X** under Jacobi (span looks “shortened”) while curling out of plane.
pub const DEFAULT_SKIP2_COMPLIANCE: f32 = 2.5e-8;
/// Cross diagonal **not** represented as a triangle edge after quad tessellation (locks in-plane shear).
/// OBJ fan `(v0,v1,v2)+(v0,v2,v3)` leaves **v1–v3** unstretched; procedural grid `(i0,i1,i2)+(i1,i3,i2)` leaves **i0–i3**.
/// **Stiffer** than [`DEFAULT_STRETCH_COMPLIANCE`] so shear converges faster under Jacobi (smaller residual ±Z early on).
pub const DEFAULT_CROSS_DIAG_COMPLIANCE: f32 = 6e-9;

#[derive(Clone, Debug)]
pub struct ClothMeshData {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub indices: Vec<u32>,
    /// For each triangle edge (3 per tri), global neighbor edge index or u32::MAX
    pub tri_edge_neighbors: Vec<u32>,
    pub inv_mass: Vec<f32>,
    /// Particle-centric constraint lists (flattened)
    pub neighbor_other: Vec<u32>,
    pub neighbor_rest_len: Vec<f32>,
    pub neighbor_compliance: Vec<f32>,
    /// Parallel to neighbor_other: unique constraint index in `constraint_*` arrays (XPBD λ storage).
    pub neighbor_constraint_id: Vec<u32>,
    pub neighbor_offsets: Vec<u32>,
    /// One row per distance/bending constraint (same order as `num_distance_constraints`).
    pub constraint_i: Vec<u32>,
    pub constraint_j: Vec<u32>,
    pub constraint_rest_len: Vec<f32>,
    pub constraint_compliance: Vec<f32>,
    pub num_particles: u32,
    pub num_distance_constraints: u32,
    pub rest_positions: Vec<Vec3>,
}

fn vec3_from_f32x3(a: [f32; 3]) -> Vec3 {
    Vec3::new(a[0], a[1], a[2])
}

fn sub3(a: Vec3, b: Vec3) -> Vec3 {
    a - b
}

fn cross3(a: Vec3, b: Vec3) -> Vec3 {
    a.cross(b)
}

fn len3(v: Vec3) -> f32 {
    v.length()
}

fn dist(a: Vec3, b: Vec3) -> f32 {
    (a - b).length()
}

pub fn parse_welded_obj(obj_text: &str) -> ClothMeshData {
    let mut verts_pos: Vec<[f32; 3]> = Vec::new();
    let mut verts_uv: Vec<[f32; 2]> = Vec::new();
    let mut verts_nrm: Vec<[f32; 3]> = Vec::new();
    let mut faces: Vec<Vec<String>> = Vec::new();

    for line in obj_text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(head) = parts.next() else { continue };
        match head {
            "v" => {
                let x: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let y: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let z: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                verts_pos.push([x, y, z]);
            }
            "vt" => {
                let u: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let v: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                verts_uv.push([u, v]);
            }
            "vn" => {
                let x: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let y: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let z: f32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                verts_nrm.push([x, y, z]);
            }
            "f" => {
                faces.push(parts.map(|s| s.to_string()).collect());
            }
            _ => {}
        }
    }

    let mut cache: HashMap<String, u32> = HashMap::new();
    let mut positions: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut uvs: Vec<Vec2> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    // Opposite corners v1–v3 per OBJ quad (fan diagonal is v0–v2).
    let mut obj_quad_cross: Vec<(u32, u32)> = Vec::new();

    // Welded corner lookup (mutable closure over mesh buffers).
    let mut corner_index = |face_string: &str| -> u32 {
        if let Some(&idx) = cache.get(face_string) {
            return idx;
        }
        let idx = positions.len() as u32;
        cache.insert(face_string.to_string(), idx);
        let parts: Vec<&str> = face_string.split('/').collect();
        let vi: usize = parts
            .get(0)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1)
            .saturating_sub(1);
        let uvi = parts
            .get(1)
            .and_then(|s| {
                if s.is_empty() {
                    None
                } else {
                    s.parse::<usize>().ok()
                }
            })
            .map(|u| u.saturating_sub(1));
        let ni = parts
            .get(2)
            .and_then(|s| s.parse::<usize>().ok())
            .map(|u| u.saturating_sub(1));

        positions.push(vec3_from_f32x3(
            verts_pos
                .get(vi)
                .copied()
                .unwrap_or([0.0, 0.0, 0.0]),
        ));
        if let Some(ui) = uvi {
            let uv = verts_uv.get(ui).copied().unwrap_or([0.0, 0.0]);
            uvs.push(Vec2::new(uv[0], uv[1]));
        } else {
            uvs.push(Vec2::ZERO);
        }
        if let Some(nix) = ni {
            normals.push(vec3_from_f32x3(
                verts_nrm
                    .get(nix)
                    .copied()
                    .unwrap_or([0.0, 1.0, 0.0]),
            ));
        } else {
            normals.push(Vec3::Y);
        }
        idx
    };

    for face in faces {
        if face.len() < 3 {
            continue;
        }
        if face.len() == 4 {
            // Match Blender triangle fan: (v0,v1,v2) + (v0,v2,v3) → mesh diagonal v0–v2; lock opposite corners v1–v3 for shear.
            let v0 = corner_index(&face[0]);
            let v1 = corner_index(&face[1]);
            let v2 = corner_index(&face[2]);
            let v3 = corner_index(&face[3]);
            indices.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
            let ia = v1.min(v3);
            let ib = v1.max(v3);
            obj_quad_cross.push((ia, ib));
            continue;
        }
        // Triangle fan for n-gons with n > 4 (and triangles via single wedge).
        let c0 = corner_index(&face[0]);
        for i in 1..face.len() - 1 {
            indices.push(c0);
            indices.push(corner_index(&face[i]));
            indices.push(corner_index(&face[i + 1]));
        }
    }

    let obj_cross: Vec<(u32, u32, f32, f32)> = obj_quad_cross
        .into_iter()
        .map(|(ia, ib)| {
            let rest = dist(positions[ia as usize], positions[ib as usize]);
            (ia, ib, rest, DEFAULT_CROSS_DIAG_COMPLIANCE)
        })
        .collect();

    // Regular Blender-exported grids only get triangle edges + bending + cross-diagonals unless we add
    // the same skip‑2 braces used by [`grid_cloth_hanging`]; without them the free bottom hem folds with
    // short-wavelength compression ("bottom edge instability").
    let mut extras =
        axis_aligned_grid_skip_two_distance_constraints(&positions).unwrap_or_default();
    extras.extend(obj_cross);

    finalize_cloth_mesh(positions, normals, uvs, indices, &extras)
}

/// Rectangle cloth in the **XY** plane (+Y up, front facing +Z), subdivided into quads.
/// The entire **top row** (maximum `y`) is pinned (`inv_mass = 0`). Structural triangle edges plus skip‑2 row/column braces reduce bottom‑edge buckling.
///
/// Prefer moderate resolution (e.g. 24×18 quads, `cell_size` 0.045) so XPBD substeps stay stable.
pub fn grid_cloth_hanging(quad_cols: u32, quad_rows: u32, cell_size: f32) -> ClothMeshData {
    assert!(quad_cols >= 1 && quad_rows >= 1);
    assert!(cell_size > 0.0 && cell_size.is_finite());

    let vtx_cols = quad_cols + 1;
    let vtx_rows = quad_rows + 1;
    let mut positions = Vec::with_capacity((vtx_cols * vtx_rows) as usize);
    let mut normals = Vec::with_capacity((vtx_cols * vtx_rows) as usize);
    let mut uvs = Vec::with_capacity((vtx_cols * vtx_rows) as usize);

    let width = quad_cols as f32 * cell_size;
    let x0 = -0.5 * width;

    for iy in 0..vtx_rows {
        for ix in 0..vtx_cols {
            let x = x0 + ix as f32 * cell_size;
            let y = iy as f32 * cell_size;
            positions.push(Vec3::new(x, y, 0.0));
            normals.push(Vec3::Z);
            uvs.push(Vec2::new(
                ix as f32 / quad_cols as f32,
                iy as f32 / quad_rows as f32,
            ));
        }
    }

    let mut indices = Vec::with_capacity((quad_cols * quad_rows * 6) as usize);
    let cols = vtx_cols as usize;
    for iy in 0..quad_rows as usize {
        for ix in 0..quad_cols as usize {
            let i0 = iy * cols + ix;
            let i1 = i0 + 1;
            let i2 = i0 + cols;
            let i3 = i2 + 1;
            indices.extend_from_slice(&[
                i0 as u32, i1 as u32, i2 as u32,
                i1 as u32, i3 as u32, i2 as u32,
            ]);
        }
    }

    let mut extras = grid_skip_two_distance_constraints(vtx_cols, vtx_rows, &positions);
    grid_quad_cross_diagonal_constraints(&mut extras, quad_cols, quad_rows, &positions);
    finalize_cloth_mesh(positions, normals, uvs, indices, &extras)
}

/// Distance **BL–TR** (`i0–i3`) for the `(i0,i1,i2)+(i1,i3,i2)` split (mesh diagonal is **i1–i2**).
fn grid_quad_cross_diagonal_constraints(
    out: &mut Vec<(u32, u32, f32, f32)>,
    quad_cols: u32,
    quad_rows: u32,
    positions: &[Vec3],
) {
    let cols = (quad_cols + 1) as usize;
    for iy in 0..quad_rows as usize {
        for ix in 0..quad_cols as usize {
            let i0 = iy * cols + ix;
            let i3 = (iy + 1) * cols + ix + 1;
            let ia = i0.min(i3) as u32;
            let ib = i0.max(i3) as u32;
            let rest = dist(positions[i0], positions[i3]);
            out.push((ia, ib, rest, DEFAULT_CROSS_DIAG_COMPLIANCE));
        }
    }
}

/// Sorted unique coordinates within `eps` (merge neighbors after sort).
fn dedup_sorted_coords(values: &mut Vec<f32>, eps: f32) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut write = 0usize;
    for read in 0..values.len() {
        if write == 0 || (values[read] - values[write - 1]).abs() > eps {
            values[write] = values[read];
            write += 1;
        }
    }
    values.truncate(write);
}

/// When welded vertices lie on a full Cartesian grid in **XY** (same topology as a subdivided quad sheet),
/// returns skip‑2 brace constraints identical to [`grid_skip_two_distance_constraints`]. Returns `None`
/// if positions are irregular or ambiguous so arbitrary meshes are untouched.
fn axis_aligned_grid_skip_two_distance_constraints(
    positions: &[Vec3],
) -> Option<Vec<(u32, u32, f32, f32)>> {
    const EPS: f32 = 2e-4;
    if positions.is_empty() {
        return None;
    }

    let mut xs: Vec<f32> = positions.iter().map(|p| p.x).collect();
    let mut ys: Vec<f32> = positions.iter().map(|p| p.y).collect();
    dedup_sorted_coords(&mut xs, EPS);
    dedup_sorted_coords(&mut ys, EPS);

    let cols = xs.len();
    let rows = ys.len();
    if cols < 3 || rows < 3 || cols.checked_mul(rows)? != positions.len() {
        return None;
    }

    let mut grid_pid: Vec<Option<u32>> = vec![None; cols * rows];
    for (pid, p) in positions.iter().enumerate() {
        let ix = xs.iter().position(|gx| (gx - p.x).abs() <= EPS)?;
        let iy = ys.iter().position(|gy| (gy - p.y).abs() <= EPS)?;
        let slot = iy * cols + ix;
        if grid_pid[slot].is_some() {
            return None;
        }
        grid_pid[slot] = Some(pid as u32);
    }
    if grid_pid.iter().any(|s| s.is_none()) {
        return None;
    }

    let pid_at = |ix: usize, iy: usize| grid_pid[iy * cols + ix].unwrap();
    let mut out = Vec::new();
    let comp = DEFAULT_SKIP2_COMPLIANCE;

    for iy in 0..rows {
        for ix in 0..cols.saturating_sub(2) {
            let i = pid_at(ix, iy);
            let j = pid_at(ix + 2, iy);
            let rest = dist(positions[i as usize], positions[j as usize]);
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            out.push((a, b, rest, comp));
        }
    }
    for iy in 0..rows.saturating_sub(2) {
        for ix in 0..cols {
            let i = pid_at(ix, iy);
            let j = pid_at(ix, iy + 2);
            let rest = dist(positions[i as usize], positions[j as usize]);
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            out.push((a, b, rest, comp));
        }
    }

    Some(out)
}

/// Every-other-vertex distance links along grid rows and columns (`2 * cell_size` rest length).
fn grid_skip_two_distance_constraints(
    vtx_cols: u32,
    vtx_rows: u32,
    positions: &[Vec3],
) -> Vec<(u32, u32, f32, f32)> {
    let cols = vtx_cols as usize;
    let rows = vtx_rows as usize;
    let mut out = Vec::new();
    let comp = DEFAULT_SKIP2_COMPLIANCE;
    for iy in 0..rows {
        for ix in 0..cols.saturating_sub(2) {
            let i = (iy * cols + ix) as u32;
            let j = i + 2;
            let rest = dist(positions[i as usize], positions[j as usize]);
            out.push((i, j, rest, comp));
        }
    }
    for iy in 0..rows.saturating_sub(2) {
        for ix in 0..cols {
            let i = (iy * cols + ix) as u32;
            let j = (((iy + 2) * cols) + ix) as u32;
            let rest = dist(positions[i as usize], positions[j as usize]);
            out.push((i, j, rest, comp));
        }
    }
    out
}

fn finalize_cloth_mesh(
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<Vec2>,
    indices: Vec<u32>,
    extra_constraints: &[(u32, u32, f32, f32)],
) -> ClothMeshData {
    let n = positions.len();
    let num_tris = indices.len() / 3;
    let mut inv_mass = vec![0.0f32; n];
    for t in 0..num_tris {
        let id0 = indices[3 * t] as usize;
        let id1 = indices[3 * t + 1] as usize;
        let id2 = indices[3 * t + 2] as usize;
        let e0 = sub3(positions[id1], positions[id0]);
        let e1 = sub3(positions[id2], positions[id0]);
        let c = cross3(e0, e1);
        let a = 0.5 * len3(c);
        let p_inv = if a > 0.0 { 1.0 / a / 3.0 } else { 0.0 };
        inv_mass[id0] += p_inv;
        inv_mass[id1] += p_inv;
        inv_mass[id2] += p_inv;
    }

    let tri_edge_neighbors = find_tri_neighbors(&indices);

    // Pin all vertices on the top boundary (maximum y).
    let mut max_y = f32::NEG_INFINITY;
    for p in &positions {
        max_y = max_y.max(p.y);
    }
    let eps = 1e-5_f32;
    for i in 0..n {
        if positions[i].y >= max_y - eps {
            inv_mass[i] = 0.0;
        }
    }

    let rest_positions = positions.clone();

    let edge_pairs = unique_edges_and_rests(&indices, &positions, &tri_edge_neighbors);
    let bend_pairs = bending_pairs(&indices, &tri_edge_neighbors, &positions);

    let mut constraints: Vec<(u32, u32, f32, f32)> = Vec::new();
    for (i, j, rest) in edge_pairs {
        constraints.push((i, j, rest, DEFAULT_STRETCH_COMPLIANCE));
    }
    for (i, j, rest) in bend_pairs {
        constraints.push((i, j, rest, DEFAULT_BEND_COMPLIANCE));
    }
    constraints.extend_from_slice(extra_constraints);
    let num_distance_constraints = constraints.len() as u32;

    let mut neighbor_offsets: Vec<u32> = vec![0; n + 1];

    let mut constraint_i: Vec<u32> = Vec::with_capacity(constraints.len());
    let mut constraint_j: Vec<u32> = Vec::with_capacity(constraints.len());
    let mut constraint_rest_compact: Vec<f32> = Vec::with_capacity(constraints.len());
    let mut constraint_comp_compact: Vec<f32> = Vec::with_capacity(constraints.len());

    let mut counts = vec![0u32; n];
    for &(i, j, _, _) in &constraints {
        counts[i as usize] += 1;
        counts[j as usize] += 1;
    }
    for i in 0..n {
        neighbor_offsets[i + 1] = neighbor_offsets[i] + counts[i];
    }

    let nn = neighbor_offsets[n] as usize;
    let mut neighbor_other: Vec<u32> = vec![0; nn];
    let mut neighbor_rest_len: Vec<f32> = vec![0.0; nn];
    let mut neighbor_compliance: Vec<f32> = vec![0.0; nn];
    let mut neighbor_constraint_id: Vec<u32> = vec![0; nn];

    // Scatter into per-particle contiguous ranges — `push()` in constraint order would leave
    // `neighbor_offsets` pointing at unrelated edges and blow up Jacobi gather (+ GPU `neighbor_packed`).
    let mut cursors = neighbor_offsets[..n].to_vec();
    for (eid, &(i, j, rest, comp)) in constraints.iter().enumerate() {
        let eid = eid as u32;
        let ii = i as usize;
        let jj = j as usize;
        constraint_i.push(i);
        constraint_j.push(j);
        constraint_rest_compact.push(rest);
        constraint_comp_compact.push(comp);

        let pi = cursors[ii] as usize;
        neighbor_other[pi] = j;
        neighbor_rest_len[pi] = rest;
        neighbor_compliance[pi] = comp;
        neighbor_constraint_id[pi] = eid;
        cursors[ii] += 1;

        let pj = cursors[jj] as usize;
        neighbor_other[pj] = i;
        neighbor_rest_len[pj] = rest;
        neighbor_compliance[pj] = comp;
        neighbor_constraint_id[pj] = eid;
        cursors[jj] += 1;
    }
    debug_assert_eq!(cursors.as_slice(), &neighbor_offsets[1..]);

    ClothMeshData {
        positions,
        normals,
        uvs,
        indices,
        tri_edge_neighbors,
        inv_mass,
        neighbor_other,
        neighbor_rest_len,
        neighbor_compliance,
        neighbor_constraint_id,
        neighbor_offsets,
        constraint_i,
        constraint_j,
        constraint_rest_len: constraint_rest_compact,
        constraint_compliance: constraint_comp_compact,
        num_particles: n as u32,
        num_distance_constraints,
        rest_positions,
    }
}

fn find_tri_neighbors(indices: &[u32]) -> Vec<u32> {
    let num_tris = indices.len() / 3;
    #[derive(Clone)]
    struct Edge {
        id0: u32,
        id1: u32,
        edge_nr: u32,
    }
    let mut edges: Vec<Edge> = Vec::with_capacity(num_tris * 3);
    for i in 0..num_tris {
        for j in 0..3u32 {
            let id0 = indices[3 * i + j as usize];
            let id1 = indices[3 * i + ((j + 1) % 3) as usize];
            edges.push(Edge {
                id0: id0.min(id1),
                id1: id0.max(id1),
                edge_nr: (3 * i as u32 + j),
 });
        }
    }
    edges.sort_by(|a, b| a.id0.cmp(&b.id0).then(a.id1.cmp(&b.id1)));

    let mut neighbors = vec![u32::MAX; 3 * num_tris];
    let mut i = 0usize;
    while i + 1 < edges.len() {
        let e0 = &edges[i];
        let e1 = &edges[i + 1];
        if e0.id0 == e1.id0 && e0.id1 == e1.id1 {
            neighbors[e0.edge_nr as usize] = e1.edge_nr;
            neighbors[e1.edge_nr as usize] = e0.edge_nr;
        }
        i += 1;
    }
    neighbors
}

fn unique_edges_and_rests(
    indices: &[u32],
    positions: &[Vec3],
    neighbors: &[u32],
) -> Vec<(u32, u32, f32)> {
    let num_tris = indices.len() / 3;
    let mut seen = HashMap::new();
    let mut out: Vec<(u32, u32, f32)> = Vec::new();
    for i in 0..num_tris {
        for j in 0..3u32 {
            let id0 = indices[3 * i + j as usize];
            let id1 = indices[3 * i + ((j + 1) % 3) as usize];
            let n = neighbors[3 * i + j as usize];
            if n != u32::MAX && id0 >= id1 {
                continue;
            }
            let a = id0.min(id1);
            let b = id0.max(id1);
            if seen.insert((a, b), ()).is_none() {
                let rest = dist(positions[a as usize], positions[b as usize]);
                out.push((a, b, rest));
            }
        }
    }
    out
}

fn bending_pairs(
    indices: &[u32],
    neighbors: &[u32],
    positions: &[Vec3],
) -> Vec<(u32, u32, f32)> {
    let num_tris = indices.len() / 3;
    let mut seen = HashMap::new();
    let mut out = Vec::new();
    for i in 0..num_tris {
        for j in 0..3u32 {
            let n = neighbors[3 * i + j as usize];
            if n == u32::MAX {
                continue;
            }
            let id2 = indices[3 * i + ((j + 2) % 3) as usize];
            let ni = n / 3;
            let nj = n % 3;
            let id3 = indices[3 * ni as usize + ((nj + 2) % 3) as usize];
            let a = id2.min(id3);
            let b = id2.max(id3);
            if seen.insert((a, b), ()).is_none() {
                let rest = dist(positions[a as usize], positions[b as usize]);
                out.push((a, b, rest));
            }
        }
    }
    out
}

impl ClothMeshData {
    pub fn to_bevy_mesh(&self) -> bevy::mesh::Mesh {
        use bevy::asset::RenderAssetUsages;
        use bevy::mesh::{Mesh, PrimitiveTopology};
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
        let pos: Vec<[f32; 3]> = self.positions.iter().map(|v| v.to_array()).collect();
        let nrm: Vec<[f32; 3]> = self.normals.iter().map(|v| v.to_array()).collect();
        let uv: Vec<[f32; 2]> = self.uvs.iter().map(|v| v.to_array()).collect();
        // UV1.x = particle id for `cloth_vertex.wgsl` when `VERTEX_UVS_B` is enabled (matches indexed draw).
        let uv1: Vec<[f32; 2]> = (0..self.positions.len())
            .map(|i| [i as f32, 0.0])
            .collect();
        mesh.insert_attribute(
            bevy::mesh::Mesh::ATTRIBUTE_POSITION,
            pos,
        );
        mesh.insert_attribute(bevy::mesh::Mesh::ATTRIBUTE_NORMAL, nrm);
        mesh.insert_attribute(bevy::mesh::Mesh::ATTRIBUTE_UV_0, uv);
        // Unused by `cloth_vertex.wgsl` (particle index comes from the index buffer + batch base vertex).
        // Kept as documentation / optional future uses (lightmaps); Texcoords remain meaningful for materials.
        mesh.insert_attribute(bevy::mesh::Mesh::ATTRIBUTE_UV_1, uv1);
        mesh.insert_indices(bevy::mesh::Indices::U32(self.indices.clone()));
        mesh
    }
}
