"""
Mesh-native preprocessing for Stage A.

Key design correction from the point-cloud version: reconstructing a
mesh from a raw point cloud (ball-pivoting) proved too slow (multiple
seconds, scaling worse with better quality) AND too fragile (fragmented
into dozens of disconnected pieces even on a clean synthetic cylinder,
which would falsely trip the double-surface hard-fail check).

Fix: Stage A should run against the mesh CR-Studio ALREADY built live
during scanning (that's what the on-screen preview is) — not a mesh we
reconstruct ourselves. Reconstruction (Open3D Poisson) stays in Stage B
only, where there's no time pressure.

This module does background/clutter removal by filtering the EXISTING
mesh's triangles directly — no resampling, no reconstruction — which is
both faster and preserves real topology instead of introducing new
artifacts.
"""

import time
import numpy as np
import open3d as o3d


def remove_small_triangle_islands(mesh: o3d.geometry.TriangleMesh, min_triangle_fraction: float = 0.02):
    """
    Drops small disconnected triangle clusters — the mesh-native
    equivalent of statistical outlier removal. A handful of stray
    triangles (sensor noise, a scanner artifact) forms a tiny isolated
    island; real anatomy doesn't.
    """
    t0 = time.perf_counter()
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    total_triangles = len(mesh.triangles)
    min_triangles = max(1, int(total_triangles * min_triangle_fraction))

    keep_clusters = np.where(cluster_n_triangles >= min_triangles)[0]
    keep_mask = np.isin(triangle_clusters, keep_clusters)

    cleaned = o3d.geometry.TriangleMesh()
    cleaned.vertices = mesh.vertices
    cleaned.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[keep_mask])
    cleaned.remove_unreferenced_vertices()
    cleaned.remove_duplicated_vertices()

    return cleaned, {
        "step": "remove_small_triangle_islands",
        "cluster_count_before": len(cluster_n_triangles),
        "clusters_dropped": int(len(cluster_n_triangles) - len(keep_clusters)),
        "triangles_before": total_triangles,
        "triangles_after": len(cleaned.triangles),
        "time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def remove_flat_background_components(mesh: o3d.geometry.TriangleMesh,
                                       flatness_threshold_mm: float = 5.0,
                                       min_component_fraction: float = 0.10):
    """
    Removes background surfaces (scanning table, floor) SAFELY by
    checking flatness PER CONNECTED COMPONENT, not across the whole
    mesh at once.

    Why not a single global plane fit: an insole scan's sole is itself
    a large, genuinely flat real surface — a naive "find the biggest
    flat region anywhere in the mesh and delete it" approach would
    strip that real geometry, not just a background table. Confirmed
    by testing: a capped test cylinder lost an entire end-cap to a
    global plane fit, corrupting real geometry.

    This version only removes an ENTIRE connected component if THAT
    component, considered on its own, is nearly flat (its thinnest
    bounding-box dimension is below flatness_threshold_mm) AND is a
    substantial fraction of the whole mesh. A real scanned object
    (including an insole, whose sole is flat but whose top surface
    isn't) is never flat as a WHOLE connected piece, since it has real
    3D structure elsewhere in the same connected surface. A table is
    genuinely flat as a whole component, because it's disconnected
    from the object and contributes only a thin slab.
    """
    t0 = time.perf_counter()
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    total_triangles = len(triangles)

    removed_components = []
    keep_mask = np.ones(total_triangles, dtype=bool)

    for cid in range(len(cluster_n_triangles)):
        tri_idx = np.where(triangle_clusters == cid)[0]
        if len(tri_idx) == 0:
            continue
        frac_of_mesh = len(tri_idx) / total_triangles

        verts_idx = np.unique(triangles[tri_idx])
        pts = vertices[verts_idx]
        extent = pts.max(axis=0) - pts.min(axis=0)
        thinnest_dim = float(np.min(extent))

        is_flat = thinnest_dim <= flatness_threshold_mm
        is_substantial = frac_of_mesh >= min_component_fraction

        if is_flat and is_substantial:
            keep_mask[tri_idx] = False
            removed_components.append({
                "cluster_id": cid, "triangle_count": int(len(tri_idx)),
                "fraction_of_mesh": round(frac_of_mesh, 3),
                "thinnest_dimension_mm": round(thinnest_dim, 2),
            })

    cleaned = o3d.geometry.TriangleMesh()
    cleaned.vertices = mesh.vertices
    cleaned.triangles = o3d.utility.Vector3iVector(triangles[keep_mask])
    cleaned.remove_unreferenced_vertices()
    cleaned.remove_duplicated_vertices()

    return cleaned, {
        "step": "remove_flat_background_components",
        "components_checked": int(len(cluster_n_triangles)),
        "components_removed": removed_components,
        "flatness_threshold_mm": flatness_threshold_mm,
        "min_component_fraction": min_component_fraction,
        "triangles_before": total_triangles,
        "triangles_after": len(cleaned.triangles),
        "time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def isolate_largest_component(mesh: o3d.geometry.TriangleMesh, secondary_cluster_flag_fraction: float = 0.15):
    """
    Mesh-native version of the point-cloud largest-cluster isolation.
    Same safety principle as before: a substantial secondary component
    (>=15% of the primary) gets FLAGGED, not silently dropped, since it
    may be real anatomy disconnected by a genuine gap (severe swelling
    fold, deformity) rather than clutter.
    """
    t0 = time.perf_counter()
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if len(cluster_n_triangles) == 0:
        return mesh, {"step": "isolate_largest_component", "skipped": True, "reason": "no triangles",
                       "flagged_secondary_cluster": False,
                       "time_ms": round((time.perf_counter() - t0) * 1000, 1)}

    order = np.argsort(cluster_n_triangles)[::-1]
    largest_cluster_id = order[0]
    largest_size = cluster_n_triangles[largest_cluster_id]

    flagged_secondary = False
    secondary_info = []
    for cid in order[1:]:
        size = cluster_n_triangles[cid]
        fraction = size / largest_size
        if fraction >= secondary_cluster_flag_fraction:
            flagged_secondary = True
            secondary_info.append({"cluster_id": int(cid), "triangle_count": int(size),
                                    "fraction_of_primary": round(float(fraction), 3)})

    keep_mask = triangle_clusters == largest_cluster_id
    cleaned = o3d.geometry.TriangleMesh()
    cleaned.vertices = mesh.vertices
    cleaned.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[keep_mask])
    cleaned.remove_unreferenced_vertices()
    cleaned.remove_duplicated_vertices()

    return cleaned, {
        "step": "isolate_largest_component",
        "category": "soft_flag" if flagged_secondary else "info",
        "cluster_count": int(len(cluster_n_triangles)),
        "largest_cluster_triangles": int(largest_size),
        "flagged_secondary_cluster": flagged_secondary,
        "secondary_clusters": secondary_info,
        "note": "A substantial secondary mesh component may be real anatomy (e.g. swelling/fold gap) rather than clutter — confirm with clinician before discarding." if flagged_secondary else None,
        "triangles_before": len(mesh.triangles),
        "triangles_after": len(cleaned.triangles),
        "time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def weld_vertices(mesh: o3d.geometry.TriangleMesh, tolerance_mm: float = 0.01):
    """
    Merges coincident/near-coincident vertices. MANDATORY first step —
    some formats (notably STL exports from CAD tools like Fusion 360)
    store each triangle with its own independent vertex coordinates,
    even where triangles are physically touching. Without welding,
    every triangle looks like its own disconnected "island" to any
    connectivity-based check (component isolation, double-surface
    detection, hole-boundary detection) — confirmed by testing: a real
    406,348-triangle scan produced 405,494 separate components before
    welding, and 1 after. Without this step, background/noise removal
    would wipe out the entire mesh.

    Scanner-native formats (PLY) are usually already welded, so this
    step is a no-op for those, but it's cheap and safe to always run.
    """
    t0 = time.perf_counter()
    triangles_before = len(mesh.triangles)
    vertices_before = len(mesh.vertices)

    mesh.merge_close_vertices(tolerance_mm)

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()

    return mesh, {
        "step": "weld_vertices",
        "tolerance_mm": tolerance_mm,
        "vertices_before": vertices_before,
        "vertices_after": len(mesh.vertices),
        "triangles": triangles_before,
        "connected_components_after_weld": int(len(cluster_n_triangles)),
        "time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def preprocess_mesh(mesh: o3d.geometry.TriangleMesh,
                     min_triangle_fraction: float = 0.02,
                     flatness_threshold_mm: float = 5.0, min_component_fraction: float = 0.10,
                     secondary_cluster_flag_fraction: float = 0.15,
                     isolate_component: bool = True):
    """
    Mesh-native preprocessing chain — no reconstruction, no resampling.
    Operates directly on the mesh CR-Studio already exported.

    isolate_component=False stops after background/noise removal,
    WITHOUT isolating to a single largest component. See note on
    check ordering in pipeline.py — the double-surface check needs to
    run before isolation, or overlapping shells get hidden.
    """
    t0 = time.perf_counter()
    log = []

    mesh0, s0 = weld_vertices(mesh)
    log.append(s0)

    mesh1, s1 = remove_small_triangle_islands(mesh0, min_triangle_fraction)
    log.append(s1)

    mesh2, s2 = remove_flat_background_components(mesh1, flatness_threshold_mm, min_component_fraction)
    log.append(s2)

    if not isolate_component:
        return mesh2, {
            "steps": log,
            "triangles_original": len(mesh.triangles),
            "triangles_final": len(mesh2.triangles),
            "total_time_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    mesh3, s3 = isolate_largest_component(mesh2, secondary_cluster_flag_fraction)
    log.append(s3)

    return mesh3, {
        "steps": log,
        "triangles_original": len(mesh.triangles),
        "triangles_final": len(mesh3.triangles),
        "total_time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
