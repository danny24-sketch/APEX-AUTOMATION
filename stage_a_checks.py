"""
Stage A — fast scan verification checks for APEX P&O scans.

Design intent:
- Runs directly on the raw/lightly-cleaned mesh, BEFORE the slow
  Open3D Poisson reconstruction step (Stage B). No reconstruction here —
  every check below is a direct geometric/topological read, which is
  what keeps this fast enough to run while the patient is still present.
- hard_fail checks always block. soft_flag checks return a warning that
  a clinician must confirm before the scan proceeds (see design discussion:
  swelling/atypical anatomy should not be auto-rejected).

Units: all mm-based thresholds assume the mesh vertices are already in
millimeters. If your CR-Studio export is in meters/cm, convert before
calling these functions (or extend check_scale_deviation to detect and
normalize it — see note in that function).
"""

import json
import time
import numpy as np
import open3d as o3d


def load_profile(device_type: str, profile_path: str = "device_profiles.json") -> dict:
    with open(profile_path, "r") as f:
        profiles = json.load(f)
    if device_type not in profiles:
        raise ValueError(f"Unknown device_type '{device_type}'. Known types: {list(profiles.keys())}")
    return profiles[device_type]


def check_double_surface(mesh: o3d.geometry.TriangleMesh, threshold_ratio: float,
                          overlap_volume_threshold: float = 0.3,
                          max_absolute_non_manifold_edges: int = 500) -> dict:
    """
    Detects double-surface / non-manifold geometry using THREE signals:

    1. Non-manifold edges shared between >2 triangles (welded double walls) —
       HIGH RATIO of non-manifold edges to total edges is the signal.
    1b. The SAME, but as an ABSOLUTE COUNT. Confirmed necessary by testing on
       a real 400K-triangle CAD export: 5,137 non-manifold edges (a real,
       substantial defect scattered across ~300 small regions) only
       amounted to 0.84% of total edges — under a 2% ratio threshold, so it
       passed when it shouldn't have. A real single-surface scan should have
       close to zero non-manifold edges regardless of mesh size; large
       meshes must not be able to hide a large absolute defect count behind
       a big denominator.
    2. Separate, spatially-overlapping connected shells (unwelded double
       walls — e.g. scanner captured inner+outer surface as two distinct
       pieces that were never merged). Caught by clustering connected
       components and checking if any two components' bounding volumes
       substantially overlap.
    """
    t0 = time.perf_counter()

    # Signal 1 + 1b: welded non-manifold edges — ratio AND absolute count
    non_manifold_edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
    n_non_manifold = len(non_manifold_edges)
    n_triangles = len(mesh.triangles)
    approx_total_edges = max(1, int(n_triangles * 1.5))
    ratio = n_non_manifold / approx_total_edges
    signal1_triggered = ratio > threshold_ratio
    signal1b_triggered = n_non_manifold > max_absolute_non_manifold_edges

    # Signal 2: overlapping separate shells
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    n_clusters = len(cluster_n_triangles)

    signal2_triggered = False
    max_overlap_fraction = 0.0
    if n_clusters > 1:
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        bboxes = []
        for cid in range(n_clusters):
            tri_idx = np.where(triangle_clusters == cid)[0]
            if len(tri_idx) == 0:
                continue
            verts_idx = np.unique(triangles[tri_idx])
            pts = vertices[verts_idx]
            bboxes.append((pts.min(axis=0), pts.max(axis=0)))

        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                min_i, max_i = bboxes[i]
                min_j, max_j = bboxes[j]
                overlap_min = np.maximum(min_i, min_j)
                overlap_max = np.minimum(max_i, max_j)
                overlap_dims = np.clip(overlap_max - overlap_min, 0, None)
                overlap_vol = np.prod(overlap_dims)
                vol_i = np.prod(np.clip(max_i - min_i, 1e-6, None))
                vol_j = np.prod(np.clip(max_j - min_j, 1e-6, None))
                smaller_vol = min(vol_i, vol_j)
                frac = overlap_vol / smaller_vol if smaller_vol > 0 else 0
                max_overlap_fraction = max(max_overlap_fraction, frac)

        signal2_triggered = max_overlap_fraction >= overlap_volume_threshold

    passed = not (signal1_triggered or signal1b_triggered or signal2_triggered)

    result = {
        "check": "double_surface",
        "category": "hard_fail",
        "passed": passed,
        "non_manifold_edge_count": n_non_manifold,
        "non_manifold_ratio": round(ratio, 5),
        "exceeded_ratio_threshold": signal1_triggered,
        "exceeded_absolute_threshold": signal1b_triggered,
        "connected_component_count": n_clusters,
        "max_shell_overlap_fraction": round(float(max_overlap_fraction), 3),
        "threshold_ratio": threshold_ratio,
        "max_absolute_non_manifold_edges": max_absolute_non_manifold_edges,
        "overlap_volume_threshold": overlap_volume_threshold,
        "time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    return result


def check_digit_separation(mesh: o3d.geometry.TriangleMesh, expected_digit_count: int,
                            slice_range: tuple = (0.5, 0.98), n_slices: int = 40,
                            cluster_eps_mm: float = 4.0) -> dict:
    """
    Detects webbing/bridging between fingers or toes — a common scan
    artifact where the scanner/reconstruction can't resolve the narrow
    gap between closely-spaced digits and fills it in with unwanted
    material. Confirmed necessary and validated against a real defect:
    a hand scan with the middle and ring fingers visibly fused only
    ever separated into 4 regions across 40 slices, never reaching the
    expected 5.

    Method: slice the mesh at many points along its primary axis, in
    the region where digits are expected (default: the far 50%-98% of
    the object's length, i.e. away from the wrist/heel end). At each
    slice, cluster the cross-section points and count how many
    separate regions exist. The MAXIMUM count reached across all
    slices is what matters — if the digits are properly separated
    ANYWHERE along their length, that slice will show the full count.

    SOFT FLAG, not hard fail: real digit fusion (syndactyly) is a
    genuine congenital condition in APEX's patient population, not
    just a scan error — this needs clinician confirmation, the same
    principle already used for swelling/atypical anatomy.

    Only meaningful for device types with digits (insole -> toes, arm
    -> fingers). Skip entirely (return None) for device types where
    expected_digit_count is not set.
    """
    if not expected_digit_count:
        return None

    t0 = time.perf_counter()
    vertices = np.asarray(mesh.vertices)
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    axis = int(np.argmax(extent))
    other_axes = [i for i in range(3) if i != axis]

    axis_min = vertices[:, axis].min()
    axis_max = vertices[:, axis].max()
    slab_thickness = (axis_max - axis_min) * 0.01

    max_regions = 0
    best_frac = None
    for frac in np.linspace(slice_range[0], slice_range[1], n_slices):
        pos = axis_min + frac * (axis_max - axis_min)
        mask = np.abs(vertices[:, axis] - pos) < slab_thickness
        slab_pts = vertices[mask][:, other_axes]
        if len(slab_pts) < 10:
            continue
        pcd = o3d.geometry.PointCloud()
        pts3 = np.hstack([slab_pts, np.zeros((len(slab_pts), 1))])
        pcd.points = o3d.utility.Vector3dVector(pts3)
        labels = np.array(pcd.cluster_dbscan(eps=cluster_eps_mm, min_points=3, print_progress=False))
        n_clusters = len(set(labels[labels >= 0]))
        if n_clusters > max_regions:
            max_regions = n_clusters
            best_frac = round(float(frac), 2)

    passed = max_regions >= expected_digit_count

    result = {
        "check": "digit_separation",
        "category": "soft_flag",
        "passed": passed,
        "expected_digit_count": expected_digit_count,
        "max_separate_regions_found": max_regions,
        "best_slice_fraction": best_frac,
        "n_slices_checked": n_slices,
        "time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    return result


def check_hole_sizes(mesh: o3d.geometry.TriangleMesh, max_hole_mm: float, expected_open_ends: int = 0) -> dict:
    """
    Finds boundary loops (hole edges) and estimates each hole's size
    as the max pairwise distance between points on that loop (a fast
    proxy for hole diameter — good enough for a pass/fail gate).

    expected_open_ends: many orthotic scans are naturally open-ended
    (an AFO/KAFO/arm scan is a segment cut from a limb — both ends are
    real, intentional openings, not defects). The N largest boundary
    loops are treated as these natural ends and excluded from the
    defect count. Set to 0 for device types expected to be a fully
    closed surface (e.g. an insole capturing the whole foot volume).

    soft_flag: does not block by itself, just reports.
    """
    t0 = time.perf_counter()

    triangles = np.asarray(mesh.triangles)
    edge_count = {}
    for tri in triangles:
        for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            key = tuple(sorted(e))
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary_edges = [e for e, c in edge_count.items() if c == 1]

    vertices = np.asarray(mesh.vertices)

    from collections import defaultdict
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    visited = set()
    loops = []
    for start in adj:
        if start in visited:
            continue
        stack = [start]
        loop = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            loop.append(node)
            for nbr in adj[node]:
                if nbr not in visited:
                    stack.append(nbr)
        if loop:
            loops.append(loop)

    hole_entries = []
    for loop in loops:
        pts = vertices[loop]
        if len(pts) < 2:
            continue
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        diameter_estimate = dists.max() * 2
        hole_entries.append({"centroid": centroid, "diameter_mm": diameter_estimate})

    # Identify natural open ends by POSITION (closest to the extremes of
    # the mesh's longest axis) AND by SIZE PLAUSIBILITY — a genuine open
    # end should be roughly as wide as the object's own cross-section,
    # not a small puncture that merely happens to sit near an extreme.
    # Without the size check, an object with very few total boundary
    # loops (e.g. only the one real defect hole present) could have
    # that defect wrongly excluded as a "natural end" by position alone.
    natural_end_indices = set()
    if expected_open_ends > 0 and hole_entries:
        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        axis = int(np.argmax(extent))
        cross_section_dims = [extent[i] for i in range(3) if i != axis]
        cross_section_width = min(cross_section_dims) if cross_section_dims else 0
        min_natural_end_diameter = cross_section_width * 0.5

        axis_positions = np.array([h["centroid"][axis] for h in hole_entries])
        diameters = np.array([h["diameter_mm"] for h in hole_entries])
        plausible_mask = diameters >= min_natural_end_diameter

        candidates_sorted_low = np.argsort(axis_positions)
        candidates_sorted_high = candidates_sorted_low[::-1]
        n_each = max(1, expected_open_ends // 2) if expected_open_ends > 1 else expected_open_ends

        for idx in candidates_sorted_low:
            if len(natural_end_indices) >= n_each:
                break
            if plausible_mask[idx]:
                natural_end_indices.add(int(idx))
        for idx in candidates_sorted_high:
            if len(natural_end_indices) >= expected_open_ends:
                break
            if plausible_mask[idx]:
                natural_end_indices.add(int(idx))

    hole_sizes_mm = [h["diameter_mm"] for h in hole_entries]
    natural_ends = [hole_sizes_mm[i] for i in natural_end_indices]
    defect_holes = [h for i, h in enumerate(hole_sizes_mm) if i not in natural_end_indices]

    largest_defect_hole = max(defect_holes) if defect_holes else 0.0

    result = {
        "check": "hole_size",
        "category": "soft_flag",
        "passed": largest_defect_hole <= max_hole_mm,
        "hole_count_total": len(hole_sizes_mm),
        "excluded_as_natural_ends_mm": [round(float(h), 2) for h in natural_ends],
        "defect_hole_count": len(defect_holes),
        "largest_defect_hole_mm": round(float(largest_defect_hole), 2),
        "all_defect_hole_sizes_mm": [round(float(h), 2) for h in defect_holes][:10],
        "max_hole_mm_threshold": max_hole_mm,
        "time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    return result


def check_scale_deviation(mesh: o3d.geometry.TriangleMesh, expected_length_range_mm, max_scale_factor: float) -> dict:
    """
    hard_fail: catches gross unit mixups (e.g. meters vs mm -> 1000x off,
    or cm vs mm -> 10x off). Uses the longest bounding-box dimension as
    a proxy for "length" and compares against expected range.

    Uses a RATIO/multiplier rather than a percent-of-midpoint, since a
    percent-of-midpoint formula badly undersells large unit-scale errors
    (e.g. a 10x-too-small scan can look like only a "46% deviation" under
    that formula, when it's actually catastrophically wrong). A ratio
    directly reflects "how many times too big/small" the scan is, which
    is what an accidental unit mixup actually produces.

    max_scale_factor of 2.0 means: hard-fail if the longest dimension is
    more than 2x above the expected max, or more than 2x below the
    expected min. Calibrate this per device type once real reference
    scans are available.
    """
    t0 = time.perf_counter()

    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()  # [x, y, z] in whatever units the mesh is in
    longest_dim = float(np.max(extent))

    if expected_length_range_mm is None:
        passed = True
        scale_factor = 1.0
    else:
        lo, hi = expected_length_range_mm
        if longest_dim < lo:
            scale_factor = lo / max(longest_dim, 1e-6)
        elif longest_dim > hi:
            scale_factor = longest_dim / hi
        else:
            scale_factor = 1.0
        passed = scale_factor <= max_scale_factor

    result = {
        "check": "scale_deviation",
        "category": "hard_fail",
        "passed": passed,
        "longest_dimension_mm": round(longest_dim, 2),
        "expected_range_mm": expected_length_range_mm,
        "scale_factor_off": round(scale_factor, 2),
        "max_scale_factor_threshold": max_scale_factor,
        "time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    return result


def check_size_range(mesh: o3d.geometry.TriangleMesh, expected_length_range_mm, expected_girth_range_mm) -> dict:
    """
    soft_flag: proportion sanity check. Length = longest bbox dimension.
    Girth = approximated via the perimeter of the cross-section at the
    bounding box's mid-point along the longest axis (cheap proxy, not
    a true anatomical girth measurement).
    """
    t0 = time.perf_counter()

    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    longest_axis = int(np.argmax(extent))
    length_mm = float(extent[longest_axis])

    other_axes = [i for i in range(3) if i != longest_axis]
    girth_proxy_mm = float(2 * (extent[other_axes[0]] + extent[other_axes[1]]))  # perimeter of bbox cross-section

    length_flag = None
    if expected_length_range_mm is not None:
        lo, hi = expected_length_range_mm
        length_flag = bool(not (lo <= length_mm <= hi))

    girth_flag = None
    if expected_girth_range_mm is not None:
        lo, hi = expected_girth_range_mm
        girth_flag = bool(not (lo <= girth_proxy_mm <= hi))

    passed = bool(not (length_flag or girth_flag))

    result = {
        "check": "size_range",
        "category": "soft_flag",
        "passed": passed,
        "length_mm": round(length_mm, 2),
        "length_out_of_range": length_flag,
        "girth_proxy_mm": round(girth_proxy_mm, 2),
        "girth_out_of_range": girth_flag,
        "note": "Flags here may reflect real anatomy (e.g. swelling) — confirm with clinician, do not auto-reject.",
        "time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    return result


def run_stage_a(mesh: o3d.geometry.TriangleMesh, device_type: str, profile_path: str = "device_profiles.json") -> dict:
    """
    Runs all Stage A checks and aggregates into a single verdict.

    verdict:
      "fail"         -> any hard_fail check failed, block immediately
      "needs_review" -> all hard_fail checks passed, but one+ soft_flag failed
      "pass"         -> everything passed clean
    """
    t0 = time.perf_counter()
    profile = load_profile(device_type, profile_path)

    hf = profile["hard_fail"]
    sf = profile["soft_flag"]

    checks = []
    checks.append(check_double_surface(mesh, hf["double_surface_ratio_threshold"]))
    checks.append(check_scale_deviation(mesh, sf.get("expected_length_range_mm"), hf["max_scale_factor"]))
    checks.append(check_hole_sizes(mesh, sf["max_hole_mm"], sf.get("expected_open_ends", 0)))
    checks.append(check_size_range(mesh, sf.get("expected_length_range_mm"), sf.get("expected_girth_range_mm")))

    digit_check = check_digit_separation(mesh, sf.get("expected_digit_count"))
    if digit_check is not None:
        checks.append(digit_check)

    hard_fails = [c for c in checks if c["category"] == "hard_fail" and not c["passed"]]
    soft_flags = [c for c in checks if c["category"] == "soft_flag" and not c["passed"]]

    if hard_fails:
        verdict = "fail"
    elif soft_flags:
        verdict = "needs_review"
    else:
        verdict = "pass"

    return {
        "device_type": device_type,
        "verdict": verdict,
        "hard_fail_reasons": [c["check"] for c in hard_fails],
        "soft_flag_reasons": [c["check"] for c in soft_flags],
        "checks": checks,
        "total_time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


if __name__ == "__main__":
    import sys
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "test_mesh.ply"
    device_type = sys.argv[2] if len(sys.argv) > 2 else "AFO"

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    result = run_stage_a(mesh, device_type)
    print(json.dumps(result, indent=2))
