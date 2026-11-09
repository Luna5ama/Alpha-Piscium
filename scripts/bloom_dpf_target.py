"""Author a complete RGB target PSF independently of the fitted cascade."""
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import ndtr

from bloom_dpf_model import (
    LEVELS, REFERENCE, Cascade, Filter, axis_filter, downsample, impulse, reference_taps, window,
)


REFERENCE_PRESETS = (
    "standard", "diamond-four", "horizontal-streak", "cross-four", "camera-six", "camera-ten",
    "cinema-star", "prismatic-streak", "digits-67", "trollface",
)
LOBE_PRESET = "custom-lobes"
PRESETS = (*REFERENCE_PRESETS, LOBE_PRESET)
MAX_LOBES = 24
LOBE_TAIL_SIGMAS = 6
LOBE_REFERENCE_AREA = math.tau * 32 * 2  # Gain units for a neutral 96 px by 2 px lobe.
CAMERA_PRESETS = frozenset(("diamond-four", "cross-four", "camera-six", "camera-ten", "cinema-star"))
RAY_PRESETS = {
    "diamond-four": ((35, 145), (144, 144), (32, 32), 1., .025, None),
    "cross-four": ((0, 90), (144, 144), (32, 32), 1., .025, None),
    "camera-six": ((0, 60, 120), (144,)*3, (32,)*3, .55, .035, None),
    "camera-ten": ((18, 54, 90, 126, 162), (110, 96, 126, 96, 110),
                   (27.5, 24, 31.5, 24, 27.5), .55, .035, None),
    "cinema-star": ((0, 45, 90, 135), (144, 76, 50, 76), (32, 76*2/9, 50*2/9, 76*2/9),
                    1., .025, (.48, .20, .12, .20)),
    "prismatic-streak": ((0,), (144,), (32,), 1., .035, None),
}
REFERENCE_PHASES = ((-109, -111), (-78, 71), (-53, -26), (-20, 102),
                    (18, -69), (51, 36), (84, -12), (117, 125))
REFERENCE_RADIUS = 768


@dataclass(frozen=True)
class Lobe:
    angle: float
    length: float = 96.
    width: float = 2.
    gain: float = .25
    profile: float = 2.
    pair: int | None = None

    def __post_init__(self):
        if not math.isfinite(self.angle):
            raise ValueError("Lobe angle must be finite")
        for name, low, high in (("length",6,240), ("width",.75,24), ("gain",0,4), ("profile",2,6)):
            if not low <= getattr(self,name) <= high:
                raise ValueError(f"Lobe {name} must be in [{low}, {high}]")
        if self.pair is not None and (type(self.pair) is not int or not 0 <= self.pair < MAX_LOBES):
            raise ValueError("Invalid lobe pair")
        object.__setattr__(self,"angle",self.angle % 360)


@dataclass(frozen=True)
class Recipe:
    preset: str = "standard"
    size: float = 1.
    core_gain: float = 0.
    core_sigma: float = 3.
    tail_gain: float = 1.
    shape_gain: float = 1.
    camera: float | None = None
    prism: float | None = None
    lobes: tuple[Lobe, ...] | None = None

    def __post_init__(self):
        if self.preset not in PRESETS:
            raise ValueError(f"Unknown Bloom preset {self.preset!r}")
        if not .25 <= self.size <= 4:
            raise ValueError("size must be in [.25, 4]")
        if not .25 <= self.core_sigma <= 64:
            raise ValueError("core_sigma must be in [.25, 64]")
        for name in ("tail_gain", "core_gain", "shape_gain"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("camera", "prism"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 2):
                raise ValueError(f"{name} must be in [0, 2]")

        if self.preset == LOBE_PRESET:
            if self.lobes is None:
                lobes = tuple(Lobe(angle,pair=i % 3) for i,angle in enumerate(range(0,360,60)))
            else:
                if not isinstance(self.lobes,(list,tuple)) or len(self.lobes) > MAX_LOBES:
                    raise ValueError(f"Expected at most {MAX_LOBES} lobes")
                lobes = tuple(lobe if isinstance(lobe,Lobe) else Lobe(**lobe) for lobe in self.lobes)
            pairs = [lobe.pair for lobe in lobes if lobe.pair is not None]
            if any(pairs.count(pair) != 2 for pair in set(pairs)):
                raise ValueError("Each linked pair must contain two lobes")
            object.__setattr__(self,"lobes",lobes)
        elif self.lobes is not None:
            raise ValueError("Only the custom lobe preset accepts lobes")

    def resolved(self):
        values = asdict(self)
        if self.camera is None:
            values["camera"] = float(self.preset in CAMERA_PRESETS)
        if self.prism is None:
            values["prism"] = float(self.preset == "prismatic-streak")
        if self.lobes is None:
            del values["lobes"]
        return values

    def digest(self):
        values = self.resolved()
        if self.lobes is not None:
            values["lobe_profile"] = "neutral-source-origin"
            for lobe in values["lobes"]:
                del lobe["pair"]
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=1)
def reference_shape_tables():
    """Frozen cd33a99c authoring reference, independent of generated runtime GLSL."""
    with np.load(REFERENCE, allow_pickle=False) as archive:
        names = archive["shape_names"].tolist()
        if tuple(names) != REFERENCE_PRESETS[1:]:
            raise ValueError("The published shape table is incomplete")
        return {name: (float(gain), archive[f"shape_{i}"])
                for i, (name, gain) in enumerate(zip(names, archive["shape_gains"], strict=True), 1)}


@lru_cache(maxsize=2)
def reference_tail(side=2048):
    down, up = reference_taps()
    total = np.zeros((2 * REFERENCE_RADIUS + 1,) * 2, dtype=np.float32)
    weights = [up[:, 2:3].astype(np.float32)] * LEVELS
    injection = np.full((LEVELS, 1), 1 / LEVELS, dtype=np.float32)
    for phase in REFERENCE_PHASES:
        pyr = [a[:, None] for a in downsample(impulse(side, [phase]), down)]
        output, _ = Cascade((side, side), pyr, [up[:, :2]] * LEVELS).forward(weights, injection)
        total += output[0, 0][window(side, phase, REFERENCE_RADIUS)]
    total /= len(REFERENCE_PHASES)
    return total


@lru_cache(maxsize=1)
def reference_d1():
    side, radius = 64, 6
    phases = [(x, y) for y in range(2) for x in range(2)]
    pyramid = downsample(impulse(side, phases), reference_taps()[0], levels=1)
    result = Filter(pyramid[0].shape[-2:], (side, side), [(0., 0.)]).apply(pyramid[0], [np.float32(1)])
    return np.mean([image[window(side, phase, radius)] for image, phase in zip(result, phases, strict=True)], axis=0)


def shape_response(taps, radius=160):
    carrier = reference_d1()
    edge = carrier.shape[0] // 2
    image = np.zeros((3, 2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    for x, y, *weight in taps:
        ix, iy = math.floor(x), math.floor(y)
        w = np.asarray(weight, dtype=np.float32)[:, None, None]
        for dx, u in ((0, ix+1-x), (1, x-ix)):
            for dy, v in ((0, iy+1-y), (1, y-iy)):
                if u * v:
                    px, py = radius - 2 * (ix + dx), radius - 2 * (iy + dy)
                    image[:, py-edge:py+edge+1, px-edge:px+edge+1] += (u * v) * w * carrier
    return image


@lru_cache(maxsize=16)
def reference_shape(name):
    gain, taps = reference_shape_tables()[name]
    response = shape_response(taps)
    response *= (gain / response.sum(axis=(1, 2), dtype=np.float64))[:, None, None]
    return response


@lru_cache(maxsize=24)
def resize_axis(source_radius, target_radius, scale, shift=0.):
    rows, columns, values = [], [], []
    for col in range(2 * source_radius + 1):
        center = col - source_radius
        low, high = (center - .5) * scale + shift, (center + .5) * scale + shift
        first = max(-target_radius, math.floor(low + .5))
        last = min(target_radius, math.ceil(high - .5))
        for row in range(first, last + 1):
            overlap = min(high, row + .5) - max(low, row - .5)
            if overlap > 0:
                rows.append(row + target_radius)
                columns.append(col)
                values.append(overlap / scale)
    return csr_matrix((np.asarray(values, dtype=np.float32), (rows, columns)),
                      shape=(2 * target_radius + 1, 2 * source_radius + 1))


def resize_mass(image, radius, scale, *, shift_x=0., horizontal=False):
    x = axis_filter(image, resize_axis(image.shape[-1] // 2, radius, scale, shift_x))
    y = resize_axis(image.shape[-2] // 2, radius, 1. if horizontal else scale)
    return axis_filter(x.swapaxes(-2, -1), y).swapaxes(-2, -1)


def color_raster(image, camera, prism, shifts, *, horizontal=False):
    """Tint a neutral target while retaining each channel's total mass."""
    radius = image.shape[-1] // 2
    fraction = .30 * prism
    output = []
    for stretch, shift in zip((1 + .035*camera, 1., 1 - .035*camera), shifts, strict=True):
        channel = image
        if fraction:
            offset = shift * prism
            a = resize_mass(image, radius, 1., shift_x=offset)
            b = resize_mass(image, radius, 1., shift_x=-offset)
            channel = (1 - fraction) * image + .5 * fraction * (a + b)
        if camera:
            channel = resize_mass(channel, radius, stretch, horizontal=horizontal)
        output.append(channel * np.float32(image.sum(dtype=np.float64) / channel.sum(dtype=np.float64)))
    return np.stack(output)


def halo_target(radius, scale, camera, prism):
    halo = reference_tail()
    if not (camera or prism):
        return resize_mass(halo, radius, scale)
    fraction = .30 * prism
    result = []
    for stretch, shift in zip((1 + .035*camera, 1., 1 - .035*camera), (88,68,48), strict=True):
        channel_scale = scale * stretch
        channel = (1 - fraction) * resize_mass(halo,radius,channel_scale)
        if fraction:
            delta = shift * prism * channel_scale
            channel += .5 * fraction * (resize_mass(halo,radius,channel_scale,shift_x=delta)
                                      + resize_mass(halo,radius,channel_scale,shift_x=-delta))
        result.append(channel)
    return np.stack(result)


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def axis_taps(angle, length, taper_width, stretch, *, scaled_dispersion, spacing):
    angle = np.deg2rad(angle)
    direction = np.array([np.cos(angle), np.sin(angle)])
    r = np.arange(spacing, length, spacing)
    taper = smoothstep((length - r) / taper_width)
    scaled = r * (144 / length)
    ramp = smoothstep(((scaled if scaled_dispersion else r) - 18) / 30)
    d = scaled[:, None] / (1 + ramp[:, None] * (np.asarray(stretch) - 1))
    weights = taper[:, None] * (.4 * np.exp(-.5 * (d / 16)**2) + .6 / (1 + (d / 34)**2))
    weights *= spacing / 2.8
    taps = []
    for sign in (-1, 1):
        for distance, weight in zip(r, weights, strict=True):
            x, y = -sign * distance * direction / 2
            taps.append((x, y, weight))
    return taps


def prism_axis(angle, length, camera, prism, dispersion, scaled_dispersion, *, spacing=2.):
    count = int(length // spacing)
    r = np.arange(-count,count+1,dtype=float)*spacing
    distance = np.abs(r) * (144 / length)
    ramp_input = distance if scaled_dispersion else np.abs(r)
    ramp = smoothstep((ramp_input - 18) / 30)
    stretch = 1 + ramp[:, None] * np.array([dispersion*camera, 0., -dispersion*camera])
    center = np.array([88, 68, 48]) * (1 + .15 * (prism - 1))
    delta = np.abs(distance[:, None] / stretch - center)
    weights = np.exp(-.5 * (delta / 7)**2) * smoothstep((21 - delta) / 3)
    weights /= weights.sum(axis=0)
    angle = np.deg2rad(angle)
    direction = -np.array([np.cos(angle), np.sin(angle)]) / 2
    taps = []
    for distance, w in zip(r, weights, strict=True):
        if np.any(w):
            x, y = distance * direction
            taps.append((x, y, w))
    return taps


def ray_taps(name, camera, prism, scale=1.):
    angles, lengths, tapers, origin, dispersion, axis_shares = RAY_PRESETS[name]
    stretch = (1 + dispersion*camera, 1., 1 - dispersion*camera)
    scaled = name == "cinema-star"
    groups = [axis_taps(a,l,t,stretch,scaled_dispersion=scaled,spacing=2.8/max(1.,scale))
              for a, l, t in zip(angles, lengths, tapers, strict=True)]
    if axis_shares is None:
        # Each preset has one shared origin tap, divided over its axes here.
        center_share = origin / len(groups)
        masses = np.array([np.sum([w for _, _, w in taps], axis=0) + center_share for taps in groups])
        axis_shares = masses / masses.sum(axis=0)
    else:
        center_share = origin
    result = []
    fraction = .30 * prism
    for a, l, group, share in zip(angles, lengths, groups, axis_shares, strict=True):
        needle = [(0., 0., np.full(3, center_share)), *group]
        mass = np.sum([weight for _,_,weight in needle],axis=0)
        needle = [(x,y,weight/mass) for x,y,weight in needle]
        result.extend((x, y, w * (1 - fraction) * share) for x, y, w in needle)
        if fraction:
            lobes = prism_axis(a,l,camera,prism,dispersion,scaled,spacing=2./max(1.,scale))
            result.extend((x, y, w * fraction * share) for x, y, w in lobes)
    return np.asarray([(x,y,*w) for x,y,w in result])


def colored_shape(recipe):
    p = recipe.resolved()
    camera, prism = p["camera"], p["prism"]
    reference = reference_shape(recipe.preset)
    defaults = Recipe(recipe.preset).resolved()
    if (camera, prism) == (defaults["camera"], defaults["prism"]):
        return reference
    gain = reference_shape_tables()[recipe.preset][0]
    if recipe.preset in RAY_PRESETS:
        response = shape_response(ray_taps(recipe.preset,camera,prism))
        return response * (gain / response.sum(axis=(1, 2), dtype=np.float64)).astype(np.float32)[:, None, None]
    if recipe.preset == "horizontal-streak":
        needle = color_raster(reference.mean(axis=0), camera, 0., (0, 0, 0), horizontal=True)
        if not prism:
            return needle
        lobes = shape_response([(x, y, *w) for x, y, w in prism_axis(0, 144, camera, prism, .035, False)])
        fraction = .30 * prism
        return (1 - fraction) * needle + gain * fraction * lobes
    width = 44 if recipe.preset == "digits-67" else 60
    return color_raster(reference.mean(axis=0), camera, prism, (.20*width, .14*width, .08*width))


def preset_seed_taps(recipe):
    """Resample the author's support for the current preset geometry and dispersion."""
    name = recipe.preset
    gain, frozen = reference_shape_tables()[name]
    resolved = recipe.resolved()
    default = Recipe(name).resolved()
    camera,prism = resolved["camera"],resolved["prism"]
    same_color = (camera,prism) == (default["camera"],default["prism"])
    if name in RAY_PRESETS and (not same_color or recipe.size > 1):
        taps = ray_taps(name,camera,prism,recipe.size)
    elif same_color:
        taps = frozen.copy()
    else:
        xy = frozen[:,:2]
        weights = frozen[:,2:].mean(axis=1)
        rgb = np.eye(3)
        stretch = (1+.035*camera,1.,1-.035*camera)
        fraction = .30*prism
        if name == "horizontal-streak":
            parts = [np.column_stack((xy*[scale,1],weights[:,None]*color*(1-fraction)))
                     for scale,color in zip(stretch,rgb,strict=True)]
            if fraction:
                parts.append(np.asarray([(x,y,*(w*fraction)) for x,y,w in
                                         prism_axis(0,144,camera,prism,.035,False,spacing=2./max(1.,recipe.size))]))
        else:
            width = 44 if name == "digits-67" else 60
            parts = []
            for scale,shift,color in zip(stretch,(.20*width,.14*width,.08*width),rgb,strict=True):
                parts.append(np.column_stack((xy*scale,weights[:,None]*color*(1-fraction))))
                if fraction:
                    delta = .5*shift*prism
                    for sign in (-1,1):
                        parts.append(np.column_stack(((xy+[sign*delta,0])*scale,
                                                      weights[:,None]*color*(.5*fraction))))
        taps = np.concatenate(parts)
    taps[:,:2] *= recipe.size
    taps[:,2:] *= gain*recipe.shape_gain/taps[:,2:].sum(axis=0)
    return taps


def lobe_channels(lobe, size, camera, prism):
    length = lobe.length * size
    for stretch, center in zip((1+.035*camera,1.,1-.035*camera),(88,68,48),strict=True):
        yield length*stretch/3, length*center/144 * (1+.15*(prism-1)), length*stretch*7/144


def lobe_extent(channels, width, prism, sigmas):
    longitudinal = max(sigma*sigmas for sigma,_,_ in channels)
    if prism:
        longitudinal = max(longitudinal, max(center+band*sigmas for _,center,band in channels))
    return longitudinal, width*sigmas


def lobe_fit_geometry(recipe):
    """Fine-detail boundary and outer canvas for the active analytic lobes."""
    settings = recipe.resolved()
    detail = 6 * recipe.size + 1
    support = 0
    if recipe.shape_gain:
        for lobe in recipe.lobes:
            if not lobe.gain:
                continue
            channels = tuple(lobe_channels(lobe,recipe.size,settings["camera"],settings["prism"]))
            width = lobe.width*recipe.size
            detail = max(detail,math.hypot(*lobe_extent(channels,width,settings["prism"],3)))
            support = max(support,math.hypot(*lobe_extent(channels,width,settings["prism"],LOBE_TAIL_SIGMAS)))
    return detail, max(math.ceil(support)+24,math.ceil(768*max(1,recipe.size)))


def lobe_longitudinal(along, sigma, center, band, gate_width, fraction):
    ray = (1-fraction) * np.exp(-.5 * (along/sigma)**2) / (.5*sigma)
    if fraction:
        # The prism band has zero response at the emitter and unit longitudinal area.
        narrow = band / math.sqrt(3)
        band_mass = (band * ndtr(center / math.hypot(band,gate_width))
                     - narrow * math.exp(-(center/band)**2/3)
                     * ndtr((center/3) / math.hypot(narrow,gate_width)))
        band_shape = np.exp(-.5 * ((along-center)/band)**2) * -np.expm1(-(along/band)**2)
        ray += fraction * band_shape / band_mass
    return ndtr(along / gate_width) * ray


@lru_cache(maxsize=16)
def lobe_quadrature(order):
    positions, weights = np.polynomial.legendre.leggauss(order)
    return tuple(zip(positions*.5,weights*.5,strict=True))


def lobe_response(lobe, radius, pixel_scale, size, camera, prism):
    channels = tuple(lobe_channels(lobe,size,camera,prism))
    width = lobe.width * size
    longitudinal, transverse = lobe_extent(channels,width,prism,LOBE_TAIL_SIGMAS)
    angle = math.radians(lobe.angle)
    c, s = math.cos(angle), math.sin(angle)
    rx = math.ceil(abs(c)*longitudinal + abs(s)*transverse + .5)
    ry = math.ceil(abs(s)*longitudinal + abs(c)*transverse + .5)
    y,x = np.ogrid[-ry:ry+1,-rx:rx+1]
    along = x*c + y*s
    across = y*c - x*s
    half_diagonal = math.sqrt(.5)
    active = (np.abs(along) <= longitudinal+half_diagonal) & (np.abs(across) <= transverse+half_diagonal)
    u,v = along[active],across[active]

    # Anchor the neutral lobe before dispersion, whose longitudinal profiles have equal area.
    gain = lobe.gain * (lobe.length*size/3) / LOBE_REFERENCE_AREA
    fraction = .30 * prism
    profiles = [(sigma,center,band,min(width,.25*sigma),fraction)
                for sigma,center,band in channels]
    longitudinal_spacing = min(band if fraction else sigma for sigma,_,band in channels)
    spacing = min(width/lobe.profile,longitudinal_spacing)
    order = min(40,max(2,math.ceil(2/spacing)))
    samples = lobe_quadrature(order)
    density = np.zeros((3,len(u)),dtype=np.float64)
    for ox,wx in samples:
        for oy,wy in samples:
            sample_u = u + ox*c + oy*s
            sample_v = v + oy*c - ox*s
            footprint = (wx*wy) * np.exp(-.5 * (np.abs(sample_v)/width)**lobe.profile)
            for values,profile in zip(density,profiles,strict=True):
                values += footprint * lobe_longitudinal(sample_u,*profile)
    images = np.zeros((3,2*ry+1,2*rx+1),dtype=np.float32)
    images[:,active] = density * gain
    return resize_mass(images,radius,1/pixel_scale)


def target_components(recipe, radius, pixel_scale=1.):
    """Return mass per full-res pixel on a centered canvas; outside the canvas is clipped."""
    if radius < 1:
        raise ValueError("Target radius must be positive")
    p = recipe.resolved()
    side = 2 * radius + 1
    scale = recipe.size / pixel_scale
    tail = np.zeros((side, side), dtype=np.float32)
    if recipe.tail_gain:
        camera, prism = (p["camera"],p["prism"]) if recipe.preset == "standard" else (0.,0.)
        tail = halo_target(radius,scale,camera,prism) * recipe.tail_gain
    sigma = recipe.core_sigma * recipe.size
    gaussian_radius = math.ceil(8 * sigma)
    line = np.exp(-.5 * (np.arange(-gaussian_radius,gaussian_radius+1) / sigma)**2)
    line /= line.sum()
    samples = resize_axis(gaussian_radius,radius,1/pixel_scale) @ line.astype(np.float32)
    core = np.outer(samples,samples) * np.float32(recipe.core_gain)
    shape = np.zeros((3, side, side), dtype=np.float32)
    if recipe.shape_gain and recipe.lobes is not None:
        for lobe in recipe.lobes:
            if lobe.gain:
                shape += lobe_response(lobe,radius,pixel_scale,recipe.size,p["camera"],p["prism"])
        shape *= recipe.shape_gain
    elif recipe.preset != "standard" and recipe.shape_gain:
        shape = resize_mass(colored_shape(recipe), radius, scale) * recipe.shape_gain
    area = np.float32(pixel_scale**2)
    return {"tail": tail / area, "core": core / area, "shape": shape / area}


def target_kernel(recipe, radius, pixel_scale=1.):
    parts = target_components(recipe, radius, pixel_scale)
    return parts["tail"] + parts["core"] + parts["shape"]
