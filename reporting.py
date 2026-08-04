"""
Detailed reporting — converts the raw Stage A / Stage B result dicts
into specific, itemized statements (exact numbers, exact counts, exact
locations where available) instead of generic pass/fail labels.

Built because APEX needs this for TRAINING users — a trainee needs to
see "double surface detected: 3 overlapping regions, 340 affected
edges" not just "double surface (fail)". Same for Stage B: not "holes
were closed" but "3 holes closed, largest was 12.4mm".
"""


def describe_stage_a(result: dict) -> list[str]:
    """
    Returns one itemized statement PER CHECK, always — regardless of
    pass/fail — so a trainee sees exactly what was checked and exactly
    what was found, not just the checks that happened to fail.
    """
    lines = []
    checks = result.get("checks", {})

    # --- Double surface ---
    ds = checks.get("double_surface", {})
    if ds:
        if ds["passed"]:
            lines.append(
                f"✓ Single-layer mesh confirmed {ds['connected_component_count']} surface region(s) checked, "
                f"no overlapping shells detected."
            )
        else:
            reasons = []
            if ds.get("exceeded_ratio_threshold"):
                reasons.append(f"{ds['non_manifold_edge_count']} non-manifold edges "
                                f"({ds['non_manifold_ratio']*100:.1f}% of edges, limit is {ds['threshold_ratio']*100:.1f}%)")
            if ds.get("exceeded_absolute_threshold"):
                reasons.append(f"{ds['non_manifold_edge_count']} non-manifold edges found "
                                f"(limit: {ds['max_absolute_non_manifold_edges']}) scattered defect geometry, "
                                f"even though it's a small % of this large mesh")
            if ds.get("max_shell_overlap_fraction", 0) > 0.3:
                reasons.append(f"{ds['connected_component_count']} separate surface shells found, "
                                f"overlapping by {ds['max_shell_overlap_fraction']*100:.0f}% the scanner likely "
                                f"captured the same surface twice")
            detail = "; ".join(reasons) if reasons else "criteria exceeded"
            lines.append(f"✕ Double surface detected {detail}.")

    # --- Scale deviation ---
    sd = checks.get("scale_deviation", {})
    if sd:
        rng = sd.get("expected_range_mm")
        if sd["passed"]:
            rng_txt = f" (expected {rng[0]}–{rng[1]}mm)" if rng else ""
            lines.append(f"✓ Scale looks correct longest dimension is {sd['longest_dimension_mm']}mm{rng_txt}.")
        else:
            rng_txt = f"expected {rng[0]}–{rng[1]}mm" if rng else "expected range not set for this device type"
            lines.append(
                f"✕ Scale error measured {sd['longest_dimension_mm']}mm, {rng_txt}. "
                f"That's {sd['scale_factor_off']}x outside the expected range likely a unit mix-up "
                f"(e.g. cm exported as mm)."
            )

    # --- Hole size ---
    hs = checks.get("hole_size", {})
    if hs:
        ends = hs.get("excluded_as_natural_ends_mm", [])
        ends_txt = f" ({len(ends)} identified as the segment's natural open end(s), not defects)" if ends else ""
        if hs["passed"]:
            if hs["defect_hole_count"] == 0:
                lines.append(f"✓ No defect holes found{ends_txt}.")
            else:
                sizes = ", ".join(f"{s}mm" for s in hs["all_defect_hole_sizes_mm"])
                lines.append(
                    f"✓ {hs['defect_hole_count']} hole(s) found, all within the {hs['max_hole_mm_threshold']}mm "
                    f"fillable limit sizes: {sizes}{ends_txt}."
                )
        else:
            sizes = ", ".join(f"{s}mm" for s in hs["all_defect_hole_sizes_mm"])
            lines.append(
                f"⚠ Hole exceeds fillable limit largest defect hole is {hs['largest_defect_hole_mm']}mm "
                f"(limit: {hs['max_hole_mm_threshold']}mm). {hs['defect_hole_count']} defect hole(s) total: "
                f"{sizes}{ends_txt}."
            )

    # --- Size / proportion range ---
    sr = checks.get("size_range", {})
    if sr:
        if sr["passed"]:
            lines.append(f"✓ Proportions within typical range length {sr['length_mm']}mm, girth {sr['girth_proxy_mm']}mm.")
        else:
            flags = []
            if sr.get("length_out_of_range"):
                flags.append(f"length {sr['length_mm']}mm is outside the typical range")
            if sr.get("girth_out_of_range"):
                flags.append(f"girth {sr['girth_proxy_mm']}mm is outside the typical range")
            lines.append(
                f"⚠ Unusual proportions {'; '.join(flags)}. This may reflect real anatomy "
                f"(e.g. swelling, atypical build) rather than a scan error clinician confirmation required."
            )

    # --- Background / disconnected anatomy flag ---
    # Only surface this if double_surface didn't already explain the same
    # disconnection as a genuine defect — otherwise a trainee sees the same
    # overlapping shells described twice, once as a defect and once as
    # "might be real anatomy," which is confusing and contradictory.
    double_surface_already_failed = not checks.get("double_surface", {}).get("passed", True)
    comp_step = None
    for section in ("preprocessing_phase2_component_isolation",):
        steps = result.get(section, {}).get("steps", [])
        comp_step = next((s for s in steps if s["step"] == "isolate_largest_component"), None)
    if comp_step and comp_step.get("flagged_secondary_cluster") and not double_surface_already_failed:
        for sec in comp_step.get("secondary_clusters", []):
            lines.append(
                f"⚠ Separate section detected {sec['triangle_count']} triangles "
                f"({sec['fraction_of_primary']*100:.0f}% the size of the main scan), physically disconnected "
                f"from the main surface. This was NOT deleted may be real anatomy (e.g. a deep swelling fold "
                f"splitting the scan) rather than scanner noise. Clinician confirmation required."
            )

    # --- Digit separation (fingers/toes) ---
    ds2 = checks.get("digit_separation")
    if ds2:
        if ds2["passed"]:
            lines.append(
                f"✓ Digit separation confirmed {ds2['max_separate_regions_found']} separate digits found "
                f"(expected {ds2['expected_digit_count']})."
            )
        else:
            lines.append(
                f"⚠ Digits appear fused/webbed only {ds2['max_separate_regions_found']} of "
                f"{ds2['expected_digit_count']} expected digits were found separated, at any point along the scan. "
                f"This may be a scan artifact (scanner couldn't resolve the gap between close digits) OR real "
                f"anatomy (e.g. congenital webbing/syndactyly) clinician confirmation required."
            )

    return lines


def describe_stage_b(result: dict) -> list[str]:
    """
    Returns an itemized, exact list of what Stage B actually did to the
    mesh — reconstruction stats, then every repair operation that ran,
    with real numbers.
    """
    lines = []
    recon = result.get("reconstruction", {})
    repair = result.get("repair", {})

    if recon:
        lines.append(
            f"Reconstructed surface from {recon['input_points']:,} scan points → "
            f"{recon['output_vertices']:,} vertices, {recon['output_triangles']:,} triangles "
            f"(Poisson reconstruction, depth {recon['poisson_depth']}, {recon['time_ms']/1000:.1f}s)."
        )

    repair_log = repair.get("repair_log", [])
    triangles_before = repair.get("triangles_before")
    triangles_after = repair.get("triangles_after")

    if "non_manifold_edges_repaired" in repair_log:
        lines.append("Repaired non-manifold edges (fixed any double-surface artifacts left after reconstruction).")
    if "holes_closed" in repair_log:
        lines.append("Closed remaining holes in the surface.")
    if "smoothed" in repair_log:
        lines.append(f"Smoothed the surface ({repair.get('smoothing_iterations', '?')} iterations, Taubin smoothing preserves shape, doesn't shrink the mesh).")
    if "decimated" in repair_log:
        lines.append(f"Reduced mesh to {repair.get('target_triangle_count'):,} triangles for a lighter file.")

    for entry in repair_log:
        if entry.startswith("non_manifold_repair_skipped") or entry.startswith("hole_closing_skipped") or entry.startswith("smoothing_skipped"):
            lines.append(f"⚠ Step skipped: {entry}")

    if triangles_before is not None and triangles_after is not None:
        lines.append(f"Triangle count: {triangles_before:,} → {triangles_after:,}.")

    watertight = result.get("final_is_watertight")
    if watertight:
        lines.append(f"✓ Final mesh is fully watertight {result['final_triangle_count']:,} triangles, ready for fabrication.")
    else:
        lines.append(
            f"⚠ Final mesh has some remaining gaps {result['final_triangle_count']:,} triangles, not fully "
            f"watertight. Recommend a brief manual check (e.g. in Meshmixer) before fabrication."
        )

    lines.append(f"Total processing time: {result['total_time_ms']/1000:.1f}s.")
    return lines
