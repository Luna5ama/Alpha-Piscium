#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# noqa: SIZE_OK - the mutation harness belongs beside the shader contracts it probes.

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run scripts/restir_cv_math_check.py --shader-root shaders --case full
# 3. Or make executable and run:
#      chmod +x scripts/restir_cv_math_check.py && ./scripts/restir_cv_math_check.py --shader-root shaders --case full
# ──────────────────

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path

Vec3 = tuple[float, float, float]
POOL_SIZES = (
    ("65536", "8388608"),
    ("131072", "16777216"),
    ("262144", "33554432"),
    ("524288", "67108864"),
)


@dataclass(frozen=True, slots=True)
class Sources:
    radiance_cache: str
    update: str
    sample: str
    props_source: str
    props_generated: str
    options_source: str
    options_generated: str


def load_sources(args: argparse.Namespace) -> Sources:
    root = args.shader_root
    repo = root.parent
    return Sources(
        radiance_cache=(args.override_radiance_cache or root / "techniques/gi/RadianceCache.glsl").read_text(
            encoding="utf-8"
        ),
        update=(args.override_radiance_cache_update or root / "techniques/gi/RadianceCacheUpdate.glsl").read_text(
            encoding="utf-8"
        ),
        sample=(args.override_radiance_cache_sample or root / "techniques/gi/RadianceCacheSample.glsl").read_text(
            encoding="utf-8"
        ),
        props_source=(repo / "scripts/shaders.properties").read_text(encoding="utf-8"),
        props_generated=(root / "shaders.properties").read_text(encoding="utf-8"),
        options_source=(repo / "scripts/options.main.kts").read_text(encoding="utf-8"),
        options_generated=(root / "base/Options.glsl").read_text(encoding="utf-8"),
    )


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def contains_expr(text: str, expr: str) -> bool:
    return compact(expr) in compact(text)


def block_body(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    if match is None:
        return ""
    start = text.find("{", match.end())
    if start < 0:
        return ""
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    return ""


def function_body(text: str, name: str) -> str:
    return block_body(text, rf"\b{re.escape(name)}\s*\(")


def struct_body(text: str, name: str) -> str:
    return block_body(text, rf"\bstruct\s+{re.escape(name)}\b")


def require(failures: list[str], condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def require_exprs(failures: list[str], scope: str, prefix: str, checks: tuple[tuple[str, str], ...]) -> None:
    for expr, message in checks:
        require(failures, contains_expr(scope, expr), f"{prefix}: {message}")


def assert_vec_close(actual: Vec3, expected: Vec3) -> None:
    assert all(abs(a - e) < 1e-6 for a, e in zip(actual, expected)), (actual, expected)


def check_buffer_sizes(failures: list[str], name: str, text: str) -> None:
    require(failures, "4 uvec4 records" in text, f"{name}: binding 11 comment must name 4 uvec4 records")
    for pool_size, expected_size in POOL_SIZES:
        pattern = rf"#(?:if|elif)\s+SETTING_RC_POOL_SIZE\s*==\s*{pool_size}\s*\n\s*bufferObject\.11={expected_size}"
        require(
            failures,
            re.search(pattern, text) is not None,
            f"{name}: SETTING_RC_POOL_SIZE {pool_size} must size binding 11 to {expected_size}",
        )
    require(
        failures,
        re.search(r"#else\s*\n\s*bufferObject\.11=134217728", text) is not None,
        f"{name}: fallback binding 11 size must be 134217728",
    )
    for old_size in ("6291456", "12582912", "25165824", "50331648", "100663296"):
        require(failures, old_size not in text, f"{name}: old 3-record binding size {old_size} remains")


def check_layout(sources: Sources) -> list[str]:
    failures: list[str] = []
    text = sources.radiance_cache
    require(failures, "#define RC_RESERVOIR_RECORDS 4u" in text, "RadianceCache: missing 4-record layout")
    require_exprs(failures, struct_body(text, "RCReservoir"), "RadianceCache layout", (("vec3 estimate;", "missing Fi"),))
    require_exprs(
        failures,
        function_body(text, "rc_reservoirLoad"),
        "RadianceCache load",
        (
            ("uvec4 r3 = rc_reservoirs[recordIndex + 3u];", "missing record 3"),
            ("reservoir.estimate = uintBitsToFloat(r3.xyz);", "missing estimate unpack"),
        ),
    )
    require_exprs(
        failures,
        function_body(text, "rc_reservoirStore"),
        "RadianceCache store",
        (("rc_reservoirs[recordIndex + 3u] = uvec4(floatBitsToUint(reservoir.estimate), 0u);", "missing Fi pack"),),
    )
    estimate_body = function_body(text, "rc_reservoirEstimateRadiance")
    require(failures, contains_expr(estimate_body, "reservoir.estimate"), "RadianceCache: helper must return Fi")
    require(
        failures,
        not contains_expr(estimate_body, "reservoir.radiance * reservoir.avgWY"),
        "RadianceCache: helper still reconstructs sample * avgWY",
    )
    check_buffer_sizes(failures, "scripts/shaders.properties", sources.props_source)
    check_buffer_sizes(failures, "shaders/shaders.properties", sources.props_generated)
    return failures


def check_temporal(sources: Sources) -> list[str]:
    failures: list[str] = []
    text = sources.update
    require(failures, "const float RC_CV_ALPHA = 1.0;" in text, "RadianceCacheUpdate: missing CV alpha")
    require(failures, "const float RC_CV_M_CAP = 128.0;" in text, "RadianceCacheUpdate: missing CV M cap")
    require_exprs(failures, struct_body(text, "RCCVAccumulator"), "accumulator", (("bool invalid;", "missing poison field"),))
    for name, checks in (
        ("rc_cvAccumulatorInit", (("accumulator.invalid = false;", "must clear poison"),)),
        ("rc_cvAccumulatorAdd", (
            ("weight <= 0.0", "must ignore non-positive q"),
            ("isnan(weight) || any(isnan(estimate))", "must detect invalid input"),
            ("accumulator.invalid = true;", "must poison invalid state"),
        )),
        ("rc_cvAccumulatorValid", (("!accumulator.invalid", "must reject poison"),)),
        ("rc_cvInitialEstimate", (("return candidate.valid ? candidate.radiance : vec3(0.0);", "must use direct sample"),)),
    ):
        require_exprs(failures, function_body(text, name), f"RadianceCacheUpdate {name}", checks)
    body = function_body(text, "rc_updateFace")
    require_exprs(failures, body, "RadianceCacheUpdate temporal", (
        ("float qInit = candidate.valid ? 1.0 : 0.0;", "invalid initial candidate must have zero q"),
        ("rc_luminance(reservoir.radiance) > 0.0", "history needs positive luminance"),
        ("!any(isinf(reservoir.radiance))", "history must be finite"),
        ("float qHistory = min(historyM, RC_CV_M_CAP);", "missing capped history q"),
        (
            "wSum = historyBeforeRevalidate.avgWY "
            "* rc_luminance(historyBeforeRevalidate.radiance) * reservoir.m;",
            "wSum must preserve pre-revalidation target mass",
        ),
        ("uint validateId = worldKeyHash + faceId;", "validation phase must use stable reservoir identity"),
        (
            "if ((validateId & 7u) == (uint(frameCounter) & 7u))",
            "history revalidation must run once per eight frames",
        ),
        (
            "vec3 fromHistory = RC_CV_ALPHA * historyBeforeRevalidate.estimate "
            "+ reservoir.avgWY * (reservoir.radiance - RC_CV_ALPHA * historyBeforeRevalidate.radiance);",
            "wrong one-sided temporal CV",
        ),
        ("rc_cvAccumulatorAdd(cvAccumulator, fromHistory, qHistory);", "history Fi must enter accumulator"),
        ("bijective, full-support reprojection", "must document temporal support assumption"),
        ("reservoir.estimate = rc_cvAccumulatorResolve(cvAccumulator);", "final reservoir must store Fi"),
    ))
    body_compact = compact(body)
    revalidate_pos = body_compact.find("historyValid=rc_revalidateHistoryReservoir(")
    w_sum_pos = body_compact.find(
        "wSum=historyBeforeRevalidate.avgWY*rc_luminance(historyBeforeRevalidate.radiance)*reservoir.m;"
    )
    require(failures, 0 <= revalidate_pos < w_sum_pos, "RadianceCacheUpdate temporal: wSum precedes revalidation")
    require(failures, "gl_WorkGroupID.x" not in body,
            "RadianceCacheUpdate temporal: validation phase follows dynamic workgroup")
    require(failures, not contains_expr(body, "all(greaterThan(reservoir.radiance, vec3(0.0)))"),
            "RadianceCacheUpdate temporal: rejects monochromatic history")
    return failures


def check_spatial(sources: Sources) -> list[str]:
    failures: list[str] = []
    text = sources.update
    require(failures, "const float RC_SPATIAL_M_CAP = 8.0;" in text,
            "RadianceCacheUpdate spatial: missing effective source M cap")
    mis_body = function_body(text, "rc_pairwiseSpatialMIS_MAware")
    require_exprs(failures, mis_body, "RadianceCacheUpdate spatial MIS", (
        ("shiftWeight = pTargetArea * safeRcp(pSourceArea);", "missing area-measure shift ratio"),
        ("float sourceMass = sourceM * sourceTargetWeight;", "source mass must use source target weight"),
        ("float targetMass = targetM * targetShiftedWeight * shiftWeight;", "target mass must use shifted target weight"),
        ("if (denom <= 0.0)", "mass denominator fallback must only reject non-positive mass"),
        ("return sourceMass * safeRcp(denom);", "MIS must use source owner"),
    ))
    require(failures, not contains_expr(mis_body, "float denom = targetM * pTargetArea + sourceM * pSourceArea;"),
            "RadianceCacheUpdate spatial MIS: proposal densities used as Eq. 11 masses")
    body = function_body(text, "rc_updateFace")
    spatial_scope = function_body(text, "rc_generateSpatialCandidate") + body
    require_exprs(failures, spatial_scope, "RadianceCacheUpdate spatial", (
        ("float sourceTargetWeight = rc_luminance(sourceReservoir.radiance);", "missing source-domain target weight"),
        ("misWeight = rc_pairwiseSpatialMIS_MAware", "must expose pairwise MIS"),
        ("bool sourceShiftValid = rc_generateSpatialCandidate", "missing j-to-i shift"),
        ("targetShiftValid = rc_generateSpatialCandidate", "missing independent i-to-j shift"),
        ("float storedSourceM = neighborReservoir.m;", "missing stored source M"),
        ("float sourceM = min(storedSourceM, RC_SPATIAL_M_CAP);", "source reuse M must be capped"),
        ("float targetM = max(reservoir.m, 0.0);", "MIS must preserve full target M"),
        ("if (targetM > 0.0 && (selectedFlags & RC_RES_FLAG_SURFACE_HIT) != 0u)", "reverse shift needs selected surface"),
        ("float sourceMISWeight = sourceShiftValid ? sourceMIS : 1.0;", "failed source shift must own sample"),
        ("float targetMISWeight = targetShiftValid ? targetMIS : 1.0;", "failed target shift must own sample"),
        ("vec3 targetTerm = targetMISWeight * reservoir.avgWY "
         "* (reservoir.radiance - RC_CV_ALPHA * shiftedTarget);", "missing current representative term"),
        ("vec3 sourceTerm = sourceMISWeight * neighborReservoir.avgWY "
         "* (shiftedSource - RC_CV_ALPHA * neighborReservoir.radiance);", "source term must apply one UCW"),
        ("vec3 spatialDifference = targetTerm + sourceTerm;", "missing two-sided Eq. 10"),
        ("vec3 fromSpatial = RC_CV_ALPHA * neighborReservoir.estimate + spatialDifference;", "wrong from-j estimator"),
        ("float spatialConfidence = min(max(targetM, 1.0) * safeRcp(sourceM), 1.0);",
         "missing low-M target confidence ramp"),
        ("float spatialStrength = SETTING_RC_SPATIAL_STRENGTH * spatialConfidence;",
         "spatial strength must include target confidence"),
        ("float qSpatial = min(sourceM, RC_CV_M_CAP) * spatialStrength;",
         "spatial q must use ramped strength"),
        ("spatialCandidate.targetWeight * sourceShiftWeight "
         "* neighborReservoir.avgWY * sourceM * sourceMISWeight * spatialStrength", "wrong local /M resampling weight"),
        ("float spatialEffectiveMInc = sourceM * spatialStrength;", "M increment must ignore MIS"),
        ("rc_cvAccumulatorAdd(cvAccumulator, fromSpatial, qSpatial);", "spatial Fi must enter accumulator"),
    ))
    body_compact = compact(body)
    selected_update = "if(selectedCandidate){selectedAge=0u;selectedFlags=candidate.flags;}"
    reverse_guard = "if(targetM>0.0&&(selectedFlags&RC_RES_FLAG_SURFACE_HIT)!=0u)"
    require(failures, body_compact.count(selected_update) == 1
            and body_compact.find(selected_update) < body_compact.find(reverse_guard),
            "RadianceCacheUpdate spatial: selected candidate identity is stale at reverse shift")
    require(failures, "sourceCorrection" not in body, "RadianceCacheUpdate spatial: duplicated or clamped UCW")
    return failures


def check_lookup(sources: Sources) -> list[str]:
    failures: list[str] = []
    require(
        failures,
        re.search(r"reservoir\.radiance\s*\*\s*reservoir\.avgWY", sources.sample) is None,
        "RadianceCacheSample: lookup still uses sample * avgWY",
    )
    require(
        failures,
        sources.sample.count("rc_reservoirEstimateRadiance(reservoir)") >= 2,
        "RadianceCacheSample: both lookup paths must call rc_reservoirEstimateRadiance",
    )
    require(
        failures,
        'slider("SETTING_DEBUG_RC_MODE", 0, 0..10)' in sources.options_source,
        "options.main.kts: RC debug slider must include mode 10",
    )
    require(
        failures,
        "#define SETTING_DEBUG_RC_MODE 0//[0 1 2 3 4 5 6 7 8 9 10]" in sources.options_generated,
        "Options.glsl: generated RC debug range must include mode 10",
    )
    return failures


CHECKS = {"layout": check_layout, "temporal": check_temporal, "spatial": check_spatial, "lookup": check_lookup}


def run_checks(case: str, sources: Sources) -> list[str]:
    checks = CHECKS.values() if case == "full" else (CHECKS[case],)
    return [failure for check in checks for failure in check(sources)]


def mutate_update(sources: Sources, old: str, new: str) -> Sources:
    assert sources.update.count(old) == 1, old
    return replace(sources, update=sources.update.replace(old, new))


def self_test(args: argparse.Namespace) -> None:
    sources = load_sources(args)
    sample = "vec3 f() { if (true) { return a + (b - C * d); } return x; }"
    assert contains_expr(sample, "a + (b - C * d)")
    assert "return x;" in function_body(sample, "f")
    assert not run_checks("spatial", sources)
    assert not run_checks("temporal", sources)
    denom = 2.0 * 0.1 + 8.0 * 0.4
    assert abs(8.0 * 0.4 / denom - 0.941176471) < 1e-9
    assert abs(2.0 * 0.1 / denom - 0.058823529) < 1e-9
    print("PASS numeric: source-owner and canonical-owner MIS")
    non_unit_shift_denom = 8.0 * 0.4 + 2.0 * 0.1 * 3.0
    assert abs(8.0 * 0.4 / non_unit_shift_denom - 0.842105263) < 1e-9
    assert 8.0 * 0.4 / non_unit_shift_denom != 8.0 * 0.4 / denom
    print("PASS numeric: Eq. 11 target mass includes non-unit shift Jacobian")
    small_source_mass = 3e-9
    small_target_mass = 1e-9
    small_denom = small_source_mass + small_target_mass
    small_owner = small_source_mass / small_denom if small_denom > 0.0 else 0.0
    scaled_source_mass = small_source_mass * 0.1
    scaled_target_mass = small_target_mass * 0.1
    scaled_denom = scaled_source_mass + scaled_target_mass
    scaled_owner = scaled_source_mass / scaled_denom if scaled_denom > 0.0 else 0.0
    old_cutoff_owner = small_source_mass / small_denom if small_denom > 1e-6 else 0.0
    assert small_denom < 1e-6 and scaled_denom < 1e-6
    assert abs(small_owner - 0.75) < 1e-12
    assert abs(scaled_owner - small_owner) < 1e-12
    assert old_cutoff_owner == 0.0
    print("PASS numeric: sub-cutoff positive MIS masses preserve scale-invariant ownership")
    source_term = tuple(0.75 * (x - y) for x, y in zip((1.0, 4.0, 8.0), (10.0, 1.0, 2.0)))
    target_term = tuple(0.5 * (x - y) for x, y in zip((8.0, 3.0, 2.0), (2.0, 2.0, 1.5)))
    assert_vec_close(source_term, (-6.75, 2.25, 4.5))
    assert_vec_close(tuple(x + y for x, y in zip(source_term, target_term)), (-3.75, 2.75, 4.75))
    print("PASS numeric: full two-sided Eq. 10")
    assert 2.0 * (5.0 - 1.0) == 8.0 and 2.0 * 2.0 * (5.0 - 1.0) != 8.0
    assert 8.0 * 0.5 == 4.0 and 8.0 * 0.25 * 0.5 != 4.0
    print("PASS numeric: one UCW and MIS-independent M")
    temporal = tuple(a + c - o for a, c, o in zip((8.0, 4.0, 2.0), (2.0, 10.0, 4.0), (6.0, 2.0, 1.0)))
    assert_vec_close(temporal, (4.0, 12.0, 5.0))
    colors = ((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0))
    assert all(all(isfinite(x) for x in value) and sum(x * x for x in value) > 0.0 for value in colors)
    assert sum(x * x for x in (0.0, 0.0, 0.0)) == 0.0
    assert not all(isfinite(x) for x in (float("nan"), 1.0, 1.0))
    print("PASS numeric: temporal replay and monochromatic validity")
    old_weight_sum = 1.0 * 1.0 * 10.0
    confidence_ratio = 1.0 / 16.0
    revalidated_weight_sum = old_weight_sum * confidence_ratio
    pulsed_weight_sum = 1.0 * 4.0 * (10.0 * confidence_ratio)
    assert revalidated_weight_sum == 0.625
    assert pulsed_weight_sum == 2.5
    print("PASS numeric: revalidation preserves old target mass (0.625, not 2.5)")
    reservoir_ids = ((0x12345678, 0), (0x9ABCDEF0, 3), (0x13579BDF, 5))
    assert all(
        sum((((world_key_hash + face_id) & 7) == frame) for frame in range(8)) == 1
        for world_key_hash, face_id in reservoir_ids
    )
    print("PASS numeric: stable reservoir identity validates once per eight frames")
    valid_q_init = 1.0 if True else 0.0
    invalid_q_init = 1.0 if False else 0.0
    history_estimate = 4.0
    assert (valid_q_init * 2.0 + history_estimate) / (valid_q_init + 1.0) == 3.0
    assert (invalid_q_init * 0.0 + history_estimate) / (invalid_q_init + 1.0) == history_estimate
    print("PASS numeric: invalid initial candidate contributes zero CV weight")
    from_history = tuple(h + c - o for h, c, o in zip((2.0, 1.0, 0.5), (1.5, 0.25, 0.75), (1.0, 0.5, 0.25)))
    estimate_sum = tuple(initial + 4.0 * history for initial, history in zip((0.3, 0.6, 0.9), from_history))
    assert_vec_close(tuple(x / 5.0 for x in estimate_sum), (2.06, 0.72, 0.98))
    print("PASS numeric: accumulator PIN (2.06, 0.72, 0.98)")
    local_weight = 2.0 * 1.5 * 0.25 * 8.0 * 0.75 * 0.5
    assert local_weight == 2.25 and local_weight / 8.0 != 2.25 and local_weight * 8.0 != 2.25
    print("PASS numeric: local /M weight has source M exactly once")
    stored_source_m = 1024.0
    source_m = min(stored_source_m, 8.0)
    assert source_m == 8.0
    assert stored_source_m / source_m == 128.0
    print("PASS numeric: spatial reuse caps effective source M (1024 -> 8)")
    target_ms = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
    spatial_confidence = tuple(min(max(target_m, 1.0) / source_m, 1.0) for target_m in target_ms)
    effective_source_ms = tuple(source_m * confidence for confidence in spatial_confidence)
    assert spatial_confidence == (0.125, 0.125, 0.25, 0.5, 1.0, 1.0)
    assert effective_source_ms == (1.0, 1.0, 2.0, 4.0, 8.0, 8.0)
    print("PASS numeric: low-M target ramps spatial q, weight, and M together")
    mutations = (
        ("reversed MIS numerator", check_spatial, "return sourceMass * safeRcp(denom);", "return targetMass * safeRcp(denom);"),
        (
            "proposal-density-only MIS", check_spatial,
            "float sourceMass = sourceM * sourceTargetWeight;\n    float targetMass = targetM * targetShiftedWeight * shiftWeight;",
            "float sourceMass = sourceM * pSourceArea;\n    float targetMass = targetM * pTargetArea;",
        ),
        ("restored absolute MIS denominator cutoff", check_spatial, "if (denom <= 0.0)", "if (denom <= 1e-6)"),
        ("missing reverse surface guard", check_spatial,
            "if (targetM > 0.0 && (selectedFlags & RC_RES_FLAG_SURFACE_HIT) != 0u)",
            "if (targetM > 0.0)"),
        ("stale reverse-shift identity", check_spatial,
            "if (selectedCandidate) {\n        selectedAge = 0u;\n        selectedFlags = candidate.flags;\n    }",
            "if (selectedCandidate) {\n        selectedAge = 0u;\n    }",
        ),
        ("missing Eq.10 target term", check_spatial, "vec3 spatialDifference = targetTerm + sourceTerm;", "vec3 spatialDifference = sourceTerm;"),
        ("missing Eq.10 source term", check_spatial, "vec3 spatialDifference = targetTerm + sourceTerm;", "vec3 spatialDifference = targetTerm;"),
        (
            "duplicated UCW", check_spatial,
            "sourceMISWeight * neighborReservoir.avgWY * (shiftedSource",
            "sourceMISWeight * neighborReservoir.avgWY * neighborReservoir.avgWY * (shiftedSource",
        ),
        (
            "MIS-scaled M", check_spatial,
            "float spatialEffectiveMInc = sourceM * spatialStrength;",
            "float spatialEffectiveMInc = sourceM * sourceMISWeight * spatialStrength;",
        ),
        (
            "uncapped spatial reuse M", check_spatial,
            "float sourceM = min(storedSourceM, RC_SPATIAL_M_CAP);",
            "float sourceM = storedSourceM;",
        ),
        (
            "missing low-M spatial confidence ramp", check_spatial,
            "float spatialConfidence = min(max(targetM, 1.0) * safeRcp(sourceM), 1.0);",
            "float spatialConfidence = 1.0;",
        ),
        (
            "unramped spatial CV weight", check_spatial,
            "float qSpatial = min(sourceM, RC_CV_M_CAP) * spatialStrength;",
            "float qSpatial = min(sourceM, RC_CV_M_CAP) * SETTING_RC_SPATIAL_STRENGTH;",
        ),
        (
            "dynamic workgroup validation phase", check_temporal,
            "uint validateId = worldKeyHash + faceId;",
            "uint validateId = gl_WorkGroupID.x + (gl_WorkGroupID.x >> 3);",
        ),
        (
            "removed 1-in-8 temporal gate", check_temporal,
            "uint validateId = worldKeyHash + faceId;\n"
            "        if ((validateId & 7u) == (uint(frameCounter) & 7u)) {\n"
            "            historyValid = rc_revalidateHistoryReservoir(\n                worldCellCoord,\n"
            "                level,\n                faceId,\n                reservoir\n            );\n"
            "            if (!historyValid) {\n                reservoir = rc_reservoirInit();\n            }\n        }",
            "historyValid = rc_revalidateHistoryReservoir(\n            worldCellCoord,\n            level,\n"
            "            faceId,\n            reservoir\n        );\n"
            "        if (!historyValid) {\n            reservoir = rc_reservoirInit();\n        }",
        ),
        (
            "invalid initial candidate CV weight", check_temporal,
            "float qInit = candidate.valid ? 1.0 : 0.0;",
            "float qInit = 1.0;",
        ),
        (
            "revalidated-target weight pulse", check_temporal,
            "wSum = historyBeforeRevalidate.avgWY\n"
            "            * rc_luminance(historyBeforeRevalidate.radiance) * reservoir.m;",
            "wSum = reservoir.avgWY * rc_luminance(reservoir.radiance) * reservoir.m;",
        ),
        ("all-components-positive validity", check_temporal,
            "rc_luminance(reservoir.radiance) > 0.0",
            "all(greaterThan(reservoir.radiance, vec3(0.0)))"),
        ("omitted source M", check_spatial, "* neighborReservoir.avgWY\n                    * sourceM", "* neighborReservoir.avgWY"),
        (
            "double-counted source M", check_spatial,
            "* neighborReservoir.avgWY\n                    * sourceM",
            "* neighborReservoir.avgWY\n                    * sourceM\n                    * sourceM",
        ),
    )
    for name, check, old, new in mutations:
        assert check(mutate_update(sources, old, new)), name
        assert not check(sources), f"{name} restoration"
        print(f"PASS mutation: {name}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ReSTIR-CV radiance-cache shader contracts.")
    parser.add_argument("--shader-root", type=Path, default=Path("shaders"))
    parser.add_argument("--case", choices=("full", *CHECKS.keys()), default="full")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--override-radiance-cache", type=Path)
    parser.add_argument("--override-radiance-cache-update", type=Path)
    parser.add_argument("--override-radiance-cache-sample", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test(args)
        print("PASS self-test")
        return 0
    try:
        failures = run_checks(args.case, load_sources(args))
    except OSError as exc:
        print(f"FAIL {exc}")
        return 1
    if failures:
        print(f"FAIL ReSTIR-CV contract ({args.case})")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"PASS ReSTIR-CV contract ({args.case})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
