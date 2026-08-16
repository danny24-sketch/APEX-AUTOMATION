"""
Stage B full-quality cleanup/reconstruction. Runs AFTER a scan has
passed (or been confirmed past a soft_flag on) Stage A verification.
No real-time constraint here — the patient/clinician does not need to
wait for this step, so it's allowed to take longer in exchange for a
genuinely clean, fabrication-ready mesh.

Pipeline: Open3D Poisson reconstruction (fills gaps, produces a smooth
watertight-where-appropriate surface) -> PyMeshLab repair (hole-filling,
smoothing, decimation) -> final mesh export.

Input: ideally the RAW point cloud export (before any lossy meshing),
for the best reconstruction quality. Falls back to sampling points off
a mesh if only a mesh is available.
"""

import os
import tempfile
import time
import json
import numpy as np
import open3d as o3d
import pymeshlab


def load_as_point_cloud(path: str, sample_if_mesh: int = 200000) -> o3d.geometry.PointCloud:
    """
    Loads a scan as a point cloud, branching by file type rather than
    blindly trying a point-cloud read first.

    .PLY and .PCD can genuinely be either a raw point cloud OR a mesh
    (PLY supports both), so those try the point-cloud reader first and
    fall back to mesh-sampling if that comes back empty.

    .OBJ and .STL are MESH-ONLY formats there's no such thing as a
    ".obj point cloud," so a point-cloud read attempt on these would
    always fail (that's expected, not a bug, but there's no reason to
    try it and generate a confusing warning every time). These go
    straight to the mesh loader.
    """
    ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
    mesh_only_formats = {"obj", "stl", "off", "gltf", "glb"}

    if ext in mesh_only_formats:
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.triangles) == 0:
            raise ValueError(f"Could not load a mesh from {path} (0 triangles found)")
        return mesh.sample_points_uniformly(number_of_points=sample_if_mesh)

    # .ply, .pcd, or anything else — could genuinely be either type
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) > 0:
        return pcd
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles) == 0:
        raise ValueError(f"Could not load points or mesh from {path}")
    return mesh.sample_points_uniformly(number_of_points=sample_if_mesh)


def reconstruct_surface(pcd: o3d.geometry.PointCloud, poisson_depth: int = 9,
                         density_trim_quantile: float = 0.02):
    """
    Open3D Poisson surface reconstruction. This is the slow, high-
    quality step that Stage A deliberately avoids.

    poisson_depth controls detail vs smoothness (higher = more detail,
    slower, more prone to keeping noise). density_trim_quantile removes
    the lowest-density fraction of the output, since Poisson tends to
    extrapolate past the real surface at sparse/edge regions.
    """
    t0 = time.perf_counter()

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=20)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)
    densities = np.asarray(densities)

    if density_trim_quantile > 0:
        threshold = np.quantile(densities, density_trim_quantile)
        keep_vertices = densities >= threshold
        mesh.remove_vertices_by_mask(~keep_vertices)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    stats = {
        "step": "poisson_reconstruction",
        "poisson_depth": poisson_depth,
        "density_trim_quantile": density_trim_quantile,
        "input_points": len(pcd.points),
        "output_vertices": len(mesh.vertices),
        "output_triangles": len(mesh.triangles),
        "time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    return mesh, stats


def repair_and_finish(mesh: o3d.geometry.TriangleMesh, target_triangle_count: int = None,
                       smoothing_iterations: int = 3, max_hole_edges: int = 2000):
    """
    PyMeshLab repair pass: closes holes (including the larger gaps left
    by Poisson's density trimming), repairs non-manifold edges, smooths
    (Taubin — doesn't shrink the mesh the way plain Laplacian smoothing
    does), and optionally decimates to a target triangle count.
    """
    t0 = time.perf_counter()

    # Use unique temporary files to ensure multi-user safety on cloud servers
    tmp_in = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    tmp_in_path = tmp_in.name
    tmp_out_path = tmp_out.name
    tmp_in.close()
    tmp_out.close()

    try:
        o3d.io.write_triangle_mesh(tmp_in_path, mesh)
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(tmp_in_path)

        triangles_before = ms.current_mesh().face_number()
        repair_log = []

        try:
            ms.meshing_repair_non_manifold_edges()
            repair_log.append("non_manifold_edges_repaired")
        except Exception as e:
            repair_log.append(f"non_manifold_repair_skipped: {e}")

        try:
            ms.meshing_close_holes(maxholesize=max_hole_edges)
            repair_log.append("holes_closed")
        except Exception as e:
            repair_log.append(f"hole_closing_skipped: {e}")

        try:
            ms.apply_coord_taubin_smoothing(stepsmoothnum=smoothing_iterations)
            repair_log.append("smoothed")
        except Exception as e:
            repair_log.append(f"smoothing_skipped: {e}")

        if target_triangle_count and triangles_before > target_triangle_count:
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_triangle_count)
            repair_log.append("decimated")

        ms.save_current_mesh(tmp_out_path)
        result_mesh = o3d.io.read_triangle_mesh(tmp_out_path)

    finally:
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)
        if os.path.exists(tmp_out_path):
            os.remove(tmp_out_path)

    stats = {
        "step": "pymeshlab_repair",
        "repair_log": repair_log,
        "triangles_before": triangles_before,
        "triangles_after": len(result_mesh.triangles),
        "smoothing_iterations": smoothing_iterations,
        "target_triangle_count": target_triangle_count,
        "is_watertight_after": result_mesh.is_watertight(),
        "time_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    return result_mesh, stats


def run_stage_b(raw_scan_path: str, output_path: str, poisson_depth: int = 9,
                 target_triangle_count: int = None) -> dict:
    """
    Full Stage B pipeline: raw scan -> reconstructed -> repaired ->
    saved to output_path. Returns a report with timing for each step.
    """
    t_total0 = time.perf_counter()

    pcd = load_as_point_cloud(raw_scan_path)
    reconstructed_mesh, recon_stats = reconstruct_surface(pcd, poisson_depth=poisson_depth)
    final_mesh, repair_stats = repair_and_finish(reconstructed_mesh, target_triangle_count=target_triangle_count)

    o3d.io.write_triangle_mesh(output_path, final_mesh)

    return {
        "input_path": raw_scan_path,
        "output_path": output_path,
        "reconstruction": recon_stats,
        "repair": repair_stats,
        "final_triangle_count": len(final_mesh.triangles),
        "final_is_watertight": final_mesh.is_watertight(),
        "total_time_ms": round((time.perf_counter() - t_total0) * 1000, 1),
    }


if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "test_raw.ply"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "test_raw_cleaned_stageB.ply"

    result = run_stage_b(input_path, output_path)
    print(json.dumps(result, indent=2))
