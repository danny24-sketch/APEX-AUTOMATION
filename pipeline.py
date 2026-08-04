"""
End-to-end Stage A pipeline: exported mesh in -> full verification
verdict out. Preprocessing + Stage A checks combined.

DESIGN NOTES (from testing):

1. Reconstruction was abandoned for this path. Earlier versions took a
   raw POINT CLOUD and reconstructed a mesh ourselves (ball-pivoting)
   before running checks — too slow (multiple seconds) and too fragile
   (fragmented a clean synthetic object into dozens of pieces, causing
   false hard-fails). Stage A now runs directly against the mesh
   CR-Studio already builds live during scanning. Reconstruction stays
   in Stage B only, with no time pressure.

2. Check ordering matters. The double-surface check (overlapping
   shells) must run BEFORE component isolation — isolation would
   otherwise discard one overlapping shell as "disconnected anatomy"
   and hide the defect. So preprocessing runs twice: once without
   isolation (feeds double_surface), once with isolation (feeds every
   other check).
"""

import time
import json
import numpy as np
import open3d as o3d

from preprocess_mesh import preprocess_mesh
from stage_a_checks import (
    load_profile, check_double_surface, check_scale_deviation,
    check_hole_sizes, check_size_range, check_digit_separation,
)


def verify_scan(scan_mesh_path: str, device_type: str, profile_path: str = "device_profiles.json") -> dict:
    t_total0 = time.perf_counter()

    raw_mesh = o3d.io.read_triangle_mesh(scan_mesh_path)
    n_raw_triangles = len(raw_mesh.triangles)

    profile = load_profile(device_type, profile_path)
    hf = profile["hard_fail"]
    sf = profile["soft_flag"]

    # Phase 1: background/noise removal only, NO component isolation yet —
    # this is what the double-surface check needs to see.
    pre_isolation_mesh, preprocess_report_1 = preprocess_mesh(raw_mesh, isolate_component=False)
    double_surface_result = check_double_surface(pre_isolation_mesh, hf["double_surface_ratio_threshold"])

    # Phase 2: full preprocessing including component isolation, for
    # every other check (these care about the main object only, not
    # about detecting overlapping shells).
    cleaned_mesh, preprocess_report_2 = preprocess_mesh(raw_mesh, isolate_component=True)

    scale_result = check_scale_deviation(cleaned_mesh, sf.get("expected_length_range_mm"), hf["max_scale_factor"])
    hole_result = check_hole_sizes(cleaned_mesh, sf["max_hole_mm"], sf.get("expected_open_ends", 0))
    size_result = check_size_range(cleaned_mesh, sf.get("expected_length_range_mm"), sf.get("expected_girth_range_mm"))
    digit_result = check_digit_separation(cleaned_mesh, sf.get("expected_digit_count"))

    checks = [double_surface_result, scale_result, hole_result, size_result]
    if digit_result is not None:
        checks.append(digit_result)
    hard_fails = [c["check"] for c in checks if c["category"] == "hard_fail" and not c["passed"]]
    soft_flags = [c["check"] for c in checks if c["category"] == "soft_flag" and not c["passed"]]

    component_step = next((s for s in preprocess_report_2["steps"] if s["step"] == "isolate_largest_component"), {})
    preprocessing_flagged = component_step.get("flagged_secondary_cluster", False)

    if hard_fails:
        final_verdict = "fail"
    elif soft_flags or preprocessing_flagged:
        final_verdict = "needs_review"
    else:
        final_verdict = "pass"

    reasons = list(hard_fails) + list(soft_flags)
    if preprocessing_flagged:
        reasons.append("possible_disconnected_anatomy")

    total_time_ms = round((time.perf_counter() - t_total0) * 1000, 1)

    return {
        "device_type": device_type,
        "final_verdict": final_verdict,
        "reasons": reasons,
        "raw_triangle_count": n_raw_triangles,
        "preprocessing_phase1_background_removal": preprocess_report_1,
        "preprocessing_phase2_component_isolation": preprocess_report_2,
        "checks": {
            "double_surface": double_surface_result,
            "scale_deviation": scale_result,
            "hole_size": hole_result,
            "size_range": size_result,
            "digit_separation": digit_result,
        },
        "hard_fail_reasons": hard_fails,
        "soft_flag_reasons": soft_flags,
        "total_pipeline_time_ms": total_time_ms,
    }


def _json_default(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_mesh_scene.ply"
    device_type = sys.argv[2] if len(sys.argv) > 2 else "AFO"

    result = verify_scan(path, device_type)
    print(json.dumps(result, indent=2, default=_json_default))
    print()
    print(f"VERDICT: {result['final_verdict']}  |  reasons: {result['reasons']}  |  total time: {result['total_pipeline_time_ms']}ms")
