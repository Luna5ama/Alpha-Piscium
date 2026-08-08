#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///

# ─── How to run ───
# 1. Run directly with an existing uv installation:
#      uv run scripts/cloud_isotropic_ms_check.py --source-root .
# ─────────────────

from __future__ import annotations

import hashlib
import math
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

Spectral = tuple[float, float, float]

CASE_COUNT: Final = 4096; SOURCE_COUNT: Final = 8; CHANNEL_COUNT: Final = 3
OMEGA_0: Final = 0.999; ABSORPTION: Final = 0.001; DIFFUSION_K: Final = math.sqrt(3.0 * ABSORPTION)
CLOUDS_CU_ASYM: Final[Spectral] = (0.8615159687912013, 0.8732937077048064, 0.9375708300315341)
FP32_NEGATIVE_TOLERANCE: Final = 2.0e-6
WDT22_BASELINE: Final = "6b1dcf94786340c30793dddbcb4430134f0c81e573f4680beaeb62875112767a"
WDT22_STABILIZER: Final = "fMS = mix(fMS, fMS * 0.95, linearStep(0.9, 1.0, fMS));"


@dataclass(frozen=True, slots=True)
class Source:
    prefix: Spectral; sigma_t: Spectral; ds: float; radius: float


@dataclass(frozen=True, slots=True)
class Case:
    sources: tuple[Source, ...]; anisotropy: Spectral


@dataclass(frozen=True, slots=True)
class Factorization:
    phi: Spectral; sum_a: Spectral; sum_b: Spectral


Tolerance = tuple[float, float]


class CheckFailure(Exception):
    pass


FLOAT64_TOLERANCE: Final[Tolerance] = (3.0e-10, 2.0e-12); FLOAT32_DIRECT_TOLERANCE: Final[Tolerance] = (8.0e-5, 2.0e-6); FLOAT32_FACTORIZED_TOLERANCE: Final[Tolerance] = (2.0e-3, 2.0e-6)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def fraction(seed: int) -> float:
    return float((seed * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF) / 4_294_967_295.0


def build_case(index: int) -> Case:
    if index == 0:
        scale, ds = 0.0, 1.0
    elif index == 1:
        scale, ds = 1.0e-3, 0.01
    elif index == 2:
        scale, ds = 80.0, 4.0
    else:
        scale = 10.0 ** (-3.0 + 6.0 * fraction(index * 13 + 1)); ds = 0.02 + 1.98 * fraction(index * 17 + 2)

    cumulative = [0.0, 0.0, 0.0]; sources: list[Source] = []
    for source_index in range(SOURCE_COUNT):
        sigma_t: Spectral = (
            scale * (0.35 + 1.10 * fraction(index * 101 + source_index * 11 + 0)),
            scale * (0.35 + 1.10 * fraction(index * 101 + source_index * 11 + 1)),
            scale * (0.35 + 1.10 * fraction(index * 101 + source_index * 11 + 2)),
        )
        prefix = (
            cumulative[0] + 0.5 * sigma_t[0] * ds,
            cumulative[1] + 0.5 * sigma_t[1] * ds,
            cumulative[2] + 0.5 * sigma_t[2] * ds,
        )
        sources.append(Source(prefix, sigma_t, ds, (source_index + 0.5) * ds))
        for channel in range(CHANNEL_COUNT):
            cumulative[channel] += sigma_t[channel] * ds
    return Case(tuple(sources), CLOUDS_CU_ASYM)


@dataclass(frozen=True, slots=True)
class Arithmetic:
    fp32: bool; inverse_radius: bool


FLOAT64: Final = Arithmetic(False, True)
FLOAT32: Final = Arithmetic(True, True)


def rounded(value: float, arithmetic: Arithmetic) -> float:
    if arithmetic.fp32:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    return value


def add(left: float, right: float, arithmetic: Arithmetic) -> float:
    return rounded(rounded(left, arithmetic) + rounded(right, arithmetic), arithmetic)


def subtract(left: float, right: float, arithmetic: Arithmetic) -> float:
    return rounded(rounded(left, arithmetic) - rounded(right, arithmetic), arithmetic)


def multiply(left: float, right: float, arithmetic: Arithmetic) -> float:
    return rounded(rounded(left, arithmetic) * rounded(right, arithmetic), arithmetic)


def exponential(value: float, arithmetic: Arithmetic) -> float:
    return rounded(math.exp(rounded(value, arithmetic)), arithmetic)


def source_weight(source: Source, channel: int, arithmetic: Arithmetic) -> float:
    sigma_t = rounded(source.sigma_t[channel], arithmetic)
    weight = multiply(multiply(multiply(rounded(OMEGA_0, arithmetic), sigma_t, arithmetic), source.ds, arithmetic), sigma_t, arithmetic)
    if arithmetic.inverse_radius:
        return rounded(weight / rounded(max(source.radius, source.ds * 0.5), arithmetic), arithmetic)
    return weight


def direct(case: Case, arithmetic: Arithmetic) -> Spectral:
    values: list[float] = []
    for channel in range(CHANNEL_COUNT):
        build = rounded(1.0 - case.anisotropy[channel], arithmetic)
        total = 0.0
        for source in case.sources:
            prefix = rounded(source.prefix[channel], arithmetic)
            factor = exponential(-multiply(ABSORPTION, prefix, arithmetic), arithmetic)
            factor = multiply(factor, subtract(1.0, exponential(-multiply(build, prefix, arithmetic), arithmetic), arithmetic), arithmetic)
            factor = multiply(factor, exponential(-multiply(DIFFUSION_K, prefix, arithmetic), arithmetic), arithmetic)
            total = add(total, multiply(source_weight(source, channel, arithmetic), factor, arithmetic), arithmetic)
        values.append(total)
    return values[0], values[1], values[2]


def factorized(case: Case, arithmetic: Arithmetic) -> Factorization:
    values: list[float] = []; sums_a: list[float] = []; sums_b: list[float] = []
    p = rounded(ABSORPTION + DIFFUSION_K, arithmetic)
    for channel in range(CHANNEL_COUNT):
        q = rounded(ABSORPTION + (1.0 - case.anisotropy[channel]) + DIFFUSION_K, arithmetic)
        sum_a = 0.0
        sum_b = 0.0
        for source in case.sources:
            weight = source_weight(source, channel, arithmetic)
            prefix = rounded(source.prefix[channel], arithmetic)
            sum_a = add(sum_a, multiply(weight, exponential(-multiply(p, prefix, arithmetic), arithmetic), arithmetic), arithmetic)
            sum_b = add(sum_b, multiply(weight, exponential(-multiply(q, prefix, arithmetic), arithmetic), arithmetic), arithmetic)
        values.append(subtract(sum_a, sum_b, arithmetic))
        sums_a.append(sum_a)
        sums_b.append(sum_b)
    return Factorization((values[0], values[1], values[2]), (sums_a[0], sums_a[1], sums_a[2]), (sums_b[0], sums_b[1], sums_b[2]))


def close(reference: Spectral, candidate: Spectral, tolerance: Tolerance) -> bool:
    return all(abs(actual - expected) <= tolerance[1] + tolerance[0] * abs(expected) for expected, actual in zip(reference, candidate, strict=True))


def check_math(cases: Sequence[Case]) -> None:
    for case in cases:
        require(len(case.sources) == SOURCE_COUNT, "math: source count changed")
        require(all(ABSORPTION + (1.0 - g) + DIFFUSION_K > 0.0 for g in case.anisotropy), "math: q must stay positive")
        require(all(case.sources[i].prefix[c] <= case.sources[i + 1].prefix[c] for i in range(SOURCE_COUNT - 1) for c in range(CHANNEL_COUNT)), "math: sources are not ordered")
        direct_result = direct(case, FLOAT64); factorized_result = factorized(case, FLOAT64).phi
        direct_fp32 = direct(case, FLOAT32); factorized_fp32 = factorized(case, FLOAT32).phi
        require(all(math.isfinite(value) for value in direct_result + factorized_result + direct_fp32 + factorized_fp32), "math: non-finite output")
        require(close(direct_result, factorized_result, FLOAT64_TOLERANCE), "math: float64 direct/factorized mismatch")
        require(close(direct_result, direct_fp32, FLOAT32_DIRECT_TOLERANCE), "math: float32 direct error bound exceeded")
        require(close(direct_result, factorized_fp32, FLOAT32_FACTORIZED_TOLERANCE), "math: float32 factorized error bound exceeded")
        require(all(value >= -FP32_NEGATIVE_TOLERANCE * max(1.0, abs(expected)) for expected, value in zip(direct_result, factorized_fp32, strict=True)), "math: float32 result is too negative")
    print(f"PASS math: {CASE_COUNT} cases x {SOURCE_COUNT} ordered sources x {CHANNEL_COUNT} channels; f64 rel<={FLOAT64_TOLERANCE[0]:.1e}")
    print(f"PASS float32: direct rel<={FLOAT32_DIRECT_TOLERANCE[0]:.1e}, factorized rel<={FLOAT32_FACTORIZED_TOLERANCE[0]:.1e}, nonnegative tol={FP32_NEGATIVE_TOLERANCE:.1e} scaled")


def check_edges(cases: Sequence[Case]) -> None:
    empty = factorized(cases[0], FLOAT64)
    require(direct(cases[0], FLOAT64) == (0.0, 0.0, 0.0) and empty.phi == (0.0, 0.0, 0.0) and empty.sum_b == (0.0, 0.0, 0.0), "edges: empty density is not exact zero")
    thin = cases[1]
    thin_limit = tuple(sum(source_weight(source, channel, FLOAT64) * (1.0 - thin.anisotropy[channel]) * source.prefix[channel] for source in thin.sources) for channel in range(CHANNEL_COUNT))
    require(close((thin_limit[0], thin_limit[1], thin_limit[2]), direct(thin, FLOAT64), (8.0e-5, 1.0e-18)), "edges: optically-thin limit mismatch")
    require(all(math.isfinite(value) and value >= 0.0 for value in direct(cases[2], FLOAT64) + factorized(cases[2], FLOAT64).phi), "edges: extreme optical depth is invalid")
    print("PASS edges: empty density, thin limit, and extreme OD")


def read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def wdt22_digest(common: str) -> str:
    start_token = "vec3 fMS = (sampleScattering / sampleExtinction) * (1.0 - exp(-D * sampleExtinction));"; end_token = "sampleIrradiance += sampleMSIrradiance;"
    start = common.index(start_token); end = common.index(end_token, start) + len(end_token)
    return hashlib.sha256(" ".join(common[start:end].split()).encode()).hexdigest()


def check_wdt22_numeric() -> None:
    dense_extinction = multiply(multiply(0.19358580629452912, 96.0, FLOAT32), 4.0, FLOAT32)
    radius = rounded(2.0 ** -1.5, FLOAT32)
    raw = subtract(1.0, exponential(-multiply(radius, dense_extinction, FLOAT32), FLOAT32), FLOAT32)
    unstabilized = math.inf if raw == 1.0 else raw / (1.0 - raw)
    blend = min(max((raw - 0.9) / 0.1, 0.0), 1.0)
    stabilized = rounded(raw * (1.0 - blend) + raw * 0.95 * blend, FLOAT32)
    stabilized_term = rounded(stabilized / subtract(1.0, stabilized, FLOAT32), FLOAT32)
    require(raw == 1.0 and not math.isfinite(unstabilized), "WDT22: dense FP32 case no longer reaches the nonfinite raw ratio")
    require(math.isfinite(stabilized_term), "WDT22: stabilized dense FP32 ratio is nonfinite")
    print(f"PASS WDT22 FP32: raw={raw:.1f}, raw ratio=nonfinite, stabilized ratio={stabilized_term:.6g}")


def source_contract_gaps(common: str, render: str) -> tuple[str, ...]:
    gaps: list[str] = []
    if wdt22_digest(common) != WDT22_BASELINE or WDT22_STABILIZER not in common:
        gaps.append("exact WDT22 stabilizer/base digest")
    if "vec3 msPhase = mix(vec3(UNIFORM_PHASE), layerParam.medium.phase, 0.7);" not in common or "sampleIrradiance += renderParams.lightIrradiance * tLightToSample * sampleIsotropicMSIrradiance * msPhase;" not in common:
        gaps.append("msPhase contract")
    contracts = (
        ("source constants", ("const float isotropicMSA = 0.001;", "const float isotropicMSK = sqrt(0.003);", "vec3 isotropicMSBuildRate = max(1.0 - CLOUDS_CU_ASYM, vec3(0.0));")),
        ("uniform-distance jitter/radius/prefix U", ("float lightRaySamplePrefixLength = lightRayJitterRand * lightRayStepLength;", "float lightRaySampleOffset = pow2(indexF * CLOUDS_CU_LIGHT_RAYMARCH_STEP_RCP) * lightRayLen + lightRaySamplePrefixLength;", "lightRayDir * lightRaySampleOffset", "isotropicMSOpticalDepth + sigmaTr * lightRaySamplePrefixLength")),
        ("full-bin source weight/1-r", ("vec3 isotropicMSDeltaOpticalDepth = sigmaTr * lightRayStepLength;", "(sigmaS * lightRayStepLength) * sigmaTr / max(lightRaySampleOffset, 0.5 * lightRayStepLength)")),
        ("combined absorption/diffusion", ("exp(-(isotropicMSA + isotropicMSK) * isotropicMSU)",)),
        ("post-intensity compression", ("isotropicMS = 1.0 - exp(-max(isotropicMS, vec3(0.0)));",)),
    )
    for label, fragments in contracts:
        if not all(fragment in render for fragment in fragments):
            gaps.append(label)
    order = tuple(render.find(fragment) for fragment in ("vec3 isotropicMSU =", "vec3 isotropicMSW =", "isotropicMS +=", "isotropicMSOpticalDepth +="))
    if min(order) < 0 or order != tuple(sorted(order)):
        gaps.append("source contribution/update order")
    if "isotropicMSBoundaryConfidence" in render:
        gaps.append("dead boundary confidence scaffolding")
    return tuple(gaps)


def check_source_contract(common: str, render: str) -> None:
    gaps = source_contract_gaps(common, render)
    require(not gaps, f"source contract: {', '.join(gaps)}")
    print(f"PASS WDT22 pinned baseline: {wdt22_digest(common)}")


def check_source_mutations(common: str, render: str) -> None:
    common_mutations = (("WDT", WDT22_STABILIZER, WDT22_STABILIZER.replace("0.95", "0.96")), ("msPhase mix", "vec3 msPhase = mix(vec3(UNIFORM_PHASE), layerParam.medium.phase, 0.7);", "vec3 msPhase = vec3(UNIFORM_PHASE);"), ("msPhase injection", "sampleIsotropicMSIrradiance * msPhase;", "sampleIsotropicMSIrradiance;"))
    render_mutations = (("A", "const float isotropicMSA = 0.001;", "const float isotropicMSA = 0.002;"), ("K", "const float isotropicMSK = sqrt(0.003);", "const float isotropicMSK = sqrt(0.004);"), ("jitter PDF", "lightRayJitterRand * lightRayStepLength", "pow2(lightRayJitterRand) * lightRayStepLength"), ("weight", "(sigmaS * lightRayStepLength) * sigmaTr", "(sigmaS * lightRaySamplePrefixLength) * sigmaTr"), ("1-r", "max(lightRaySampleOffset, 0.5 * lightRayStepLength)", "max(lightRayStepLength, 0.5 * lightRayStepLength)"), ("prefix U", "sigmaTr * lightRaySamplePrefixLength", "0.5 * isotropicMSDeltaOpticalDepth"), ("full-bin update", "vec3 isotropicMSDeltaOpticalDepth = sigmaTr * lightRayStepLength;", "vec3 isotropicMSDeltaOpticalDepth = sigmaTr * lightRaySamplePrefixLength;"), ("update order", "isotropicMS += isotropicMSW", "isotropicMSOpticalDepth += isotropicMSDeltaOpticalDepth;\n                                isotropicMS += isotropicMSW"), ("attenuation", "exp(-(isotropicMSA + isotropicMSK) * isotropicMSU)", "exp(-isotropicMSA * isotropicMSU)"), ("compression", "isotropicMS = 1.0 - exp(-max(isotropicMS, vec3(0.0)));", "isotropicMS = max(isotropicMS, vec3(0.0));"))
    for label, old, new in common_mutations:
        require(old in common, f"mutations: missing common target {label}"); require(source_contract_gaps(common.replace(old, new, 1), render), f"mutations: common {label} survived")
    for label, old, new in render_mutations:
        require(old in render, f"mutations: missing render target {label}"); require(source_contract_gaps(common, render.replace(old, new, 1)), f"mutations: render {label} survived")
    print(f"PASS source mutations: {len(common_mutations) + len(render_mutations)} GLSL mutants rejected")


def integration_gaps(source_root: Path) -> tuple[str, ...]:
    policy = read_utf8(source_root / 'AGENTS.md'); options = read_utf8(source_root / "scripts/options.main.kts"); programs = read_utf8(source_root / "scripts/programs.main.kts")
    properties = read_utf8(source_root / "scripts/shaders.properties"); shadesmith = read_utf8(source_root / "shaders/shadesmith.json")
    cloud_dir = source_root / "shaders/techniques/atmospherics/clouds"; common = read_utf8(cloud_dir / "Common.glsl")
    render = read_utf8(cloud_dir / "RenderVolumetric.comp.glsl")
    cumulus = read_utf8(cloud_dir / "Cumulus.glsl")
    sky = read_utf8(source_root / "shaders/techniques/atmospherics/SkyComposite.glsl")
    gaps: list[str] = []
    vibris_policy = "Vibris is profiling-only and is not a screenshot, debugging, or correctness-validation surface."
    if vibris_policy not in policy:
        gaps.append("root AGENTS Vibris policy")
    if "SETTING_CLOUDS_CU_ISOTROPIC_MS_INTENSITY" not in options:
        gaps.append("setting")
    for identifier in ("isotropicMSOpticalDepth", "isotropicMS"):
        if identifier not in render:
            gaps.append(identifier)
    for identifier in ("isotropicMSSumA", "isotropicMSSumB", "isotropicMSPrevU"):
        if identifier in render:
            gaps.append(f"obsolete {identifier}")
    if common.count("sampleIsotropicMSIrradiance") < 2:
        gaps.append("Common.glsl argument/term")
    cirrus_call = re.search(r"clouds_computeLighting\([^;]+vec3\(0\.0\),\s*vec3\(0\.0\),\s*ciAccum\s*\)", sky, re.DOTALL)
    if cirrus_call is None:
        gaps.append("SkyComposite cirrus zero")
    visible_sources = common + render + cumulus
    if "HanPi Volume Cloud" not in visible_sources or "AshenOneArt" not in visible_sources:
        gaps.append("HanPi/AshenOneArt attribution")
    caller_count = sum(read_utf8(path).count("clouds_computeLighting(") for path in (source_root / "shaders").rglob("*.glsl")) - common.count("clouds_computeLighting(")
    require(caller_count == 2, "integration scope: clouds_computeLighting caller count changed")
    require("isotropic" not in programs.lower(), "integration scope: new pass detected")
    require("isotropic" not in (properties + shadesmith).lower(), "integration scope: new texture detected")
    require(re.search(r"\bisotropicMS\w*\s*\[", render) is None, "integration scope: runtime array detected")
    require(render.count("clouds_cu_density(") == 2 and "clouds_cu_computeDensity" not in render, "integration scope: additional density march detected")
    return tuple(gaps)


def parse_source_root(arguments: Sequence[str]) -> Path:
    if len(arguments) != 2 or arguments[0] != "--source-root":
        raise CheckFailure("usage: cloud_isotropic_ms_check.py --source-root PATH")
    return Path(arguments[1]).resolve()


def main() -> int:
    try:
        source_root = parse_source_root(sys.argv[1:])
        cases = tuple(build_case(index) for index in range(CASE_COUNT))
        cloud_dir = source_root / "shaders/techniques/atmospherics/clouds"
        common = read_utf8(cloud_dir / "Common.glsl"); render = read_utf8(cloud_dir / "RenderVolumetric.comp.glsl")
        check_math(cases)
        check_edges(cases)
        check_wdt22_numeric()
        check_source_contract(common, render)
        check_source_mutations(common, render)
        gaps = integration_gaps(source_root)
        if gaps:
            print(f"FAIL integration contract: {', '.join(gaps)}")
            return 1
        print("PASS integration contracts")
        return 0
    except CheckFailure as error:
        print(f"FAIL {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
