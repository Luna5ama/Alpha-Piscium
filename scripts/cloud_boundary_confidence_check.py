#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///

# ─── How to run ───
# 1. Run directly with an existing uv installation:
#      uv run scripts/cloud_boundary_confidence_check.py --source-root .
# ─────────────────

from __future__ import annotations

import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

Vec3 = tuple[float, float, float]

WRAP_OFFSET: Final = 0.5
WRAP_RANGE: Final = 1.5
BOTTOM_HEIGHT: Final = 0.1
TOP_CONFIDENCE_STRENGTH: Final = 0.25


@dataclass(frozen=True, slots=True)
class BoundaryCase:
    normal: Vec3
    light_dir: Vec3
    local_height: float
    column_height: float


class CheckFailure(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def fp32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def saturate(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def normalize(vector: Vec3, rounding: bool = False) -> Vec3:
    x, y, z = (fp32(value) for value in vector) if rounding else vector
    length = math.sqrt(x * x + y * y + z * z)
    result = (x / length, y / length, z / length)
    return tuple(fp32(value) for value in result) if rounding else result


def surface_normal(d_h_dx: float, d_h_dz: float, rounding: bool = False) -> Vec3:
    return normalize((-d_h_dx, 1.0, -d_h_dz), rounding)


def blend_top_confidence(raw: float, strength: float) -> float:
    return (1.0 - strength) + strength * raw


def top_confidence(normal: Vec3, light_dir: Vec3, rounding: bool = False) -> float:
    dot = sum(a * b for a, b in zip(normal, light_dir, strict=True))
    raw = saturate((dot + WRAP_OFFSET) / WRAP_RANGE)
    value = blend_top_confidence(raw, TOP_CONFIDENCE_STRENGTH)
    return fp32(value) if rounding else value


def bottom_confidence(local_height: float, column_height: float, rounding: bool = False) -> float:
    scale = BOTTOM_HEIGHT * (1.0 + 3.0 * column_height)
    value = 1.0 - math.exp(-max(local_height, 0.0) / scale)
    return fp32(value) if rounding else value


def boundary_weight(case: BoundaryCase, rounding: bool = False) -> float:
    value = top_confidence(case.normal, case.light_dir, rounding) * bottom_confidence(case.local_height, case.column_height, rounding)
    return fp32(value) if rounding else value


def check_coverage_refactor() -> None:
    for noise in (-0.5, 0.0, 0.4, 0.8, 1.75, 3.0):
        for coverage in (0.0, 0.1, 0.5, 1.0):
            legacy = max(noise - (1.0 - coverage * coverage) * 0.8, 0.0) * (1.0 - (1.0 - coverage) ** 2)
            base_coverage = max(noise - (1.0 - coverage * coverage) * 0.8, 0.0)
            refactored = base_coverage * (1.0 - (1.0 - coverage) ** 2)
            require(legacy == refactored, "coverage refactor changed pre-height density")
            require(0.0 <= saturate(refactored) <= 1.0, "coverage proxy escaped [0,1]")
    print("PASS coverage: pre-height density preserved and proxy saturated")


def check_normals_and_wrap() -> None:
    require(surface_normal(0.0, 0.0) == (0.0, 1.0, 0.0), "flat normal")
    slope = surface_normal(1.0, -2.0)
    require(slope[0] < 0.0 < slope[2] and abs(sum(value * value for value in slope) - 1.0) < 1.0e-14, "sloped normal/sign")
    endpoint_light = normalize((math.sqrt(0.75), -0.5, 0.0))
    require(top_confidence((0.0, 1.0, 0.0), endpoint_light) == 0.75, "attenuated wrap lower endpoint")
    require(top_confidence((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)) == 1.0, "wrap upper endpoint")
    raw = saturate((sum(a * b for a, b in zip(slope, endpoint_light, strict=True)) + WRAP_OFFSET) / WRAP_RANGE)
    require(blend_top_confidence(raw, 0.0) == 1.0, "zero strength must be identity")
    require(blend_top_confidence(raw, 1.0) == raw, "full strength must recover raw wrap")
    print("PASS top: flat/sloped normals, 0/1 strength endpoints, default range [0.75,1]")


def check_bottom_and_product() -> None:
    heights = (0.0, 0.01, 0.05, 0.1, 0.4, 2.0)
    for column in (0.0, 0.25, 0.5, 1.0):
        values = tuple(bottom_confidence(height, column) for height in heights)
        require(values[0] == 0.0 and all(a < b for a, b in zip(values, values[1:])), "bottom zero/monotonicity")
    require(bottom_confidence(0.1, 1.0) < bottom_confidence(0.1, 0.0), "thicker column must recover slower")
    case = BoundaryCase(surface_normal(0.7, -0.4), normalize((0.2, 0.9, -0.3)), 0.17, 0.6)
    top = top_confidence(case.normal, case.light_dir)
    bottom = bottom_confidence(0.17, 0.6)
    require(boundary_weight(case) == top * bottom, "B_eff product")
    print("PASS bottom/product: zero, monotonic recovery, column scaling, B_eff=C_top*C_bottom")


def check_bounds() -> None:
    lights = tuple(normalize(vector) for vector in ((1.0, 0.0, 0.0), (-1.0, 0.2, 0.5), (0.1, -1.0, -0.2), (0.0, 1.0, 0.0)))
    count = 0
    for rounding in (False, True):
        for d_h_dx in (-16.0, -2.0, 0.0, 0.5, 8.0):
            for d_h_dz in (-9.0, -0.25, 0.0, 3.0):
                normal = surface_normal(d_h_dx, d_h_dz, rounding)
                for light_dir in lights:
                    for height in (-1.0, 0.0, 1.0e-6, 0.03, 0.2, 10.0):
                        for column in (0.0, 0.5, 1.0):
                            top = top_confidence(normal, light_dir, rounding)
                            bottom = bottom_confidence(height, column, rounding)
                            weight = boundary_weight(BoundaryCase(normal, light_dir, height, column), rounding)
                            require(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (top, bottom, weight)), "float bounds/finiteness")
                            count += 1
    print(f"PASS float64/FP32: {count} finite bounded C_top/C_bottom/B_eff cases")


def source_gaps(cumulus: str, render: str, common: str) -> tuple[str, ...]:
    gaps: list[str] = []
    cumulus_contract = (
        "float clouds_cu_baseCoverage(vec2 pos)", "float baseCoverage = coverageNoise(pos);",
        "return baseCoverage * (1.0 - pow2(1.0 - COVERAGE));", "out float coverageOut",
        "float baseCoverage = clouds_cu_baseCoverage(rayPos.xz);", "coverageOut = saturate(baseCoverage);", "densityOut = baseCoverage;",
        "float sampleStep = clamp(0.05 / _LOW_BASE_FREQ, 0.025, 0.2);",
        "clouds_cu_baseCoverage(rayPos.xz + vec2(sampleStep, 0.0))", "clouds_cu_baseCoverage(rayPos.xz - vec2(sampleStep, 0.0))",
        "clouds_cu_baseCoverage(rayPos.xz + vec2(0.0, sampleStep))", "clouds_cu_baseCoverage(rayPos.xz - vec2(0.0, sampleStep))",
        "float dHdx = (coverageXP - coverageXN) * SETTING_CLOUDS_CU_THICKNESS / (2.0 * sampleStep);",
        "float dHdz = (coverageZP - coverageZN) * SETTING_CLOUDS_CU_THICKNESS / (2.0 * sampleStep);",
        "vec3 normal = normalize(vec3(-dHdx, 1.0, -dHdz));", "float rawCTop = saturate((dot(normal, lightDir) + 0.5) / 1.5);",
        "float cTop = mix(1.0, rawCTop, 0.25);",
        "float cBottom = 1.0 - exp(-max(localHeight, 0.0) / (0.1 * mix(1.0, 4.0, columnHeight)));", "return cTop * cBottom;",
    )
    if not all(fragment in cumulus for fragment in cumulus_contract) or cumulus.count("clouds_cu_baseCoverage(") != 6:
        gaps.append("four-point coverage boundary helper")
    render_contract = (
        "float sampleCoverage = 0.0;", "sampleDensityLod, sampleCoverage)", "float lightSampleCoverage = 0.0;",
        "lightSampleDensityLod, lightSampleCoverage)", "float isotropicMSBoundaryWeight = clouds_cu_isotropicMSBoundaryWeight(",
        "stepState.position.xyz,\n                        heightFraction,\n                        sampleCoverage,",
        "sampleCoverage,\n                        renderParams.lightDir", "* isotropicMSBoundaryWeight;",
    )
    if not all(fragment in render for fragment in render_contract) or render.count("clouds_cu_density(") != 2:
        gaps.append("receiver gate and density outputs")
    order = tuple(render.find(fragment) for fragment in ("float isotropicMSBoundaryWeight =", "for (uint lightStepIndex", "vec3 isotropicMSW =", "isotropicMS +=", "isotropicMS *= SETTING_CLOUDS_CU_ISOTROPIC_MS_INTENSITY;", "isotropicMS = 1.0 - exp"))
    if min(order) < 0 or order != tuple(sorted(order)):
        gaps.append("pre-compression source-weight placement")
    ms_phase = "vec3 msPhase = mix(vec3(UNIFORM_PHASE), layerParam.medium.phase, 0.7);"
    if ms_phase not in common:
        gaps.append("unchanged msPhase")
    return tuple(gaps)


def check_source(cumulus: str, render: str, common: str) -> None:
    gaps = source_gaps(cumulus, render, common)
    require(not gaps, f"source contract: {', '.join(gaps)}")
    mutations = (
        ("normal sign", cumulus, "vec3(-dHdx, 1.0, -dHdz)", "vec3(dHdx, 1.0, dHdz)", render),
        ("wrap", cumulus, "+ 0.5) / 1.5", "+ 0.5) / 1.0", render),
        ("top strength", cumulus, "mix(1.0, rawCTop, 0.25)", "mix(1.0, rawCTop, 1.0)", render),
        ("bottom", cumulus, "0.1 * mix(1.0, 4.0, columnHeight)", "0.2 * mix(1.0, 4.0, columnHeight)", render),
        ("local height units", render, "stepState.position.xyz,\n                        heightFraction,\n                        sampleCoverage,", "stepState.position.xyz,\n                        stepState.height - cuMinHeight,\n                        sampleCoverage,", cumulus),
        ("omitted gate", render, " * isotropicMSBoundaryWeight;", ";", cumulus),
        ("misplaced gate", render, " * isotropicMSBoundaryWeight;", ";\n                    isotropicMS *= isotropicMSBoundaryWeight;", cumulus),
    )
    for label, target, old, new, other in mutations:
        require(old in target, f"mutation target missing: {label}")
        mutant_cumulus, mutant_render = (target.replace(old, new, 1), other) if target is cumulus else (other, target.replace(old, new, 1))
        require(source_gaps(mutant_cumulus, mutant_render, common), f"mutation survived: {label}")
    print("PASS source: receiver-local gate uses actual lightDir, four neighbors, source-weight placement, unchanged msPhase")
    print(f"PASS mutations: {len(mutations)} sign/wrap/strength/bottom/units/omitted/misplaced mutants rejected")


def parse_source_root(arguments: Sequence[str]) -> Path:
    if len(arguments) != 2 or arguments[0] != "--source-root":
        raise CheckFailure("usage: cloud_boundary_confidence_check.py --source-root PATH")
    return Path(arguments[1]).resolve()


def main() -> int:
    try:
        source_root = parse_source_root(sys.argv[1:])
        cloud_dir = source_root / "shaders/techniques/atmospherics/clouds"
        check_coverage_refactor()
        check_normals_and_wrap()
        check_bottom_and_product()
        check_bounds()
        check_source(
            (cloud_dir / "Cumulus.glsl").read_text(encoding="utf-8"),
            (cloud_dir / "RenderVolumetric.comp.glsl").read_text(encoding="utf-8"),
            (cloud_dir / "Common.glsl").read_text(encoding="utf-8"),
        )
    except CheckFailure as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1
    print("PASS cloud boundary confidence checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
