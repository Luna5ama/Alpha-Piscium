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


BUILTIN_PRESETS = {
    0:"standard", 2:"horizontal-streak", 4:"camera-six",
    5:"camera-ten", 6:"cinema-star", 8:"digits-67", 9:"trollface", 11:"user-8",
}
BITMAP_PRESETS = ("digits-67", "trollface")
LOBE_PRESET = "custom-lobes"
PRESETS = (*BUILTIN_PRESETS.values(), LOBE_PRESET)
MAX_LOBES = 24
LOBE_TAIL_SIGMAS = 6
LOBE_REFERENCE_AREA = math.tau * 32 * 2  # Gain units for a neutral 96 px by 2 px lobe.
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


def paired_lobes(angles, lengths, gains, widths):
    return tuple(Lobe(angle+opposite,length,width,gain,pair=i)
                 for opposite in (0,180)
                 for i,(angle,length,gain,width) in enumerate(zip(angles,lengths,gains,widths,strict=True)))


GAUSSIAN_LOBES = {
    "horizontal-streak": paired_lobes((0,),(128,),(.75,),(1.,)),
    "camera-six": paired_lobes((0,60,120),(112,)*3,(.40,)*3,(.75,)*3),
    "camera-ten": paired_lobes((18,54,90,126,162),(112,96,128,96,112),(.25,)*5,(.75,)*5),
    "cinema-star": (
        Lobe(0,128,1,0.75,2,pair=0),
        Lobe(40,72,1,0.15,2,pair=1),
        Lobe(140,72,1,0.15,2,pair=2),
        Lobe(180,128,1,0.75,2,pair=0),
        Lobe(220,72,1,0.15,2,pair=1),
        Lobe(320,72,1,0.15,2,pair=2),
        Lobe(90,48,1,0.2,2,pair=3),
        Lobe(270,48,1,0.2,2,pair=3),
        Lobe(20,32,0.75,0.25,2,pair=4),
        Lobe(200,32,0.75,0.25,2,pair=4),
        Lobe(340,32,0.75,0.25,2,pair=5),
        Lobe(160,32,0.75,0.25,2,pair=5),
        Lobe(65,48,0.75,0.15,2,pair=6),
        Lobe(295,48,0.75,0.15,2,pair=7),
        Lobe(245,48,0.75,0.15,2,pair=6),
        Lobe(115,48,0.75,0.15,2,pair=7),
    ),
    "user-8": (
        Lobe(0,144,1,0.8,2,pair=0),
        Lobe(60,84,1,0.15,2,pair=1),
        Lobe(120,84,1,0.15,2,pair=2),
        Lobe(180,144,1,0.8,2,pair=0),
        Lobe(240,84,1,0.15,2,pair=1),
        Lobe(300,84,1,0.15,2,pair=2),
        Lobe(90,48,1,0.2,2,pair=3),
        Lobe(270,48,1,0.2,2,pair=3),
        Lobe(105,32,0.75,0.25,2,pair=4),
        Lobe(285,32,0.75,0.25,2,pair=4),
        Lobe(30,48,0.75,0.25,2,pair=5),
        Lobe(210,48,0.75,0.25,2,pair=5),
        Lobe(75,32,0.75,0.15,2,pair=6),
        Lobe(330,48,0.75,0.15,2,pair=7),
        Lobe(255,32,0.75,0.15,2,pair=6),
        Lobe(150,48,0.75,0.15,2,pair=7),
    ),
}


@dataclass(frozen=True)
class Recipe:
    preset: str = "standard"
    size: float = 1.
    core_gain: float | None = None
    core_sigma: float | None = None
    tail_gain: float | None = None
    shape_gain: float | None = None
    camera: float | None = None
    prism: float | None = None
    lobes: tuple[Lobe, ...] | None = None

    def __post_init__(self):
        if self.preset not in PRESETS:
            raise ValueError(f"Unknown Bloom preset {self.preset!r}")
        if self.preset == "user-8":
            defaults = (1,3,1,.9,2,.2)
        elif self.preset in GAUSSIAN_LOBES:
            defaults = (3.1,2.45,1.1,.93,2.,.22)
        else:
            defaults = (0.,3.,1.,1.,0.,0.)
        for name,default in zip(("core_gain","core_sigma","tail_gain","shape_gain","camera","prism"),
                                defaults,strict=True):
            if getattr(self,name) is None:
                object.__setattr__(self,name,default)
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

        lobes = self.lobes
        if self.preset in GAUSSIAN_LOBES:
            if lobes is None:
                lobes = GAUSSIAN_LOBES[self.preset]
        elif self.preset == LOBE_PRESET:
            if lobes is None:
                lobes = tuple(Lobe(angle,pair=i % 3) for i,angle in enumerate(range(0,360,60)))
        elif lobes is not None:
            raise ValueError("This preset does not accept directional lobes")
        if lobes is not None:
            if not isinstance(lobes,(list,tuple)) or len(lobes) > MAX_LOBES:
                raise ValueError(f"Expected at most {MAX_LOBES} lobes")
            lobes = tuple(lobe if isinstance(lobe,Lobe) else Lobe(**lobe) for lobe in lobes)
            pairs = [lobe.pair for lobe in lobes if lobe.pair is not None]
            if any(pairs.count(pair) != 2 for pair in set(pairs)):
                raise ValueError("Each linked pair must contain two lobes")
            object.__setattr__(self,"lobes",lobes)

    def resolved(self):
        values = asdict(self)
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
        tables = {name:(float(gain),archive[f"shape_{i}"])
                  for i,(name,gain) in enumerate(zip(names,archive["shape_gains"],strict=True),1)
                  if name in BITMAP_PRESETS}
        if set(tables) != set(BITMAP_PRESETS):
            raise ValueError("The published bitmap table is incomplete")
        return tables


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


def resize_mass(image, radius, scale, *, shift_x=0.):
    x = axis_filter(image, resize_axis(image.shape[-1] // 2, radius, scale, shift_x))
    y = resize_axis(image.shape[-2] // 2, radius, scale)
    return axis_filter(x.swapaxes(-2, -1), y).swapaxes(-2, -1)


def color_raster(image, camera, prism, shifts):
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
            channel = resize_mass(channel, radius, stretch)
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


def colored_bitmap_shape(recipe):
    reference = reference_shape(recipe.preset)
    if not (recipe.camera or recipe.prism):
        return reference
    width = 44 if recipe.preset == "digits-67" else 60
    return color_raster(reference.mean(axis=0), recipe.camera, recipe.prism, (.20*width, .14*width, .08*width))


def bitmap_seed_taps(recipe):
    gain,frozen = reference_shape_tables()[recipe.preset]
    camera,prism = recipe.camera,recipe.prism
    if not (camera or prism):
        taps = frozen.copy()
    else:
        xy = frozen[:,:2]
        weights = frozen[:,2:].mean(axis=1)
        rgb = np.eye(3)
        stretch = (1+.035*camera,1.,1-.035*camera)
        fraction = .30*prism
        width = 44 if recipe.preset == "digits-67" else 60
        parts = []
        for scale,shift,color in zip(stretch,(.20*width,.14*width,.08*width),rgb,strict=True):
            parts.append(np.column_stack((xy*scale,weights[:,None]*color*(1-fraction))))
            if fraction:
                delta = .5*shift*prism
                for sign in (-1,1):
                    parts.append(np.column_stack(((xy+[sign*delta,0])*scale,weights[:,None]*color*(.5*fraction))))
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


@dataclass(frozen=True, eq=False)
class RasterTarget:
    """Linear RGB mass with an explicit optical origin and full-resolution pixel spacing."""
    pixels: np.ndarray
    origin: tuple[float, float] | None = None
    pixel_size: float = 1.

    def __post_init__(self):
        pixels = np.array(self.pixels, dtype=np.float32, order="C", copy=True)
        if pixels.ndim != 3 or pixels.shape[0] != 3 or any(n % 2 != 1 for n in pixels.shape[1:]):
            raise ValueError("A raster target requires three channels and odd spatial dimensions")
        if not np.isfinite(pixels).all() or np.any(pixels < 0) or np.any(pixels.sum(axis=(1,2)) <= 0):
            raise ValueError("A raster target requires finite nonnegative RGB with positive channel energy")
        if not math.isfinite(self.pixel_size) or self.pixel_size <= 0:
            raise ValueError("Raster pixel spacing must be positive and finite")
        origin = self.origin or ((pixels.shape[2]-1)/2, (pixels.shape[1]-1)/2)
        if len(origin) != 2 or not all(math.isfinite(x) for x in origin):
            raise ValueError("Raster optical origin must contain two finite coordinates")
        pixels.flags.writeable = False
        object.__setattr__(self, "pixels", pixels)
        object.__setattr__(self, "origin", tuple(map(float,origin)))
        object.__setattr__(self, "pixel_size", float(self.pixel_size))
        header = json.dumps((pixels.shape, self.origin, self.pixel_size)).encode()
        object.__setattr__(self, "_digest", hashlib.sha256(header + pixels.tobytes()).hexdigest())

    size = 1.
    lobes = None

    def digest(self):
        return self._digest

    def resolved(self):
        return dict(raster_sha256=self.digest(), origin=self.origin, pixel_size=self.pixel_size,
                    dimensions=self.pixels.shape)


def target_components(recipe, radius, pixel_scale=1.):
    """Return mass per full-res pixel on a centered canvas; outside the canvas is clipped."""
    if radius < 1:
        raise ValueError("Target radius must be positive")
    if isinstance(recipe, RasterTarget):
        height, width = recipe.pixels.shape[1:]
        scale = recipe.pixel_size / pixel_scale
        x = resize_axis(width//2, radius, scale, (width//2-recipe.origin[0])*scale)
        y = resize_axis(height//2, radius, scale, (height//2-recipe.origin[1])*scale)
        image = axis_filter(axis_filter(recipe.pixels, x).swapaxes(-2,-1), y).swapaxes(-2,-1)
        # Area integration produces mass per represented pixel. Consumers use density.
        image /= np.float32(pixel_scale**2)
        return dict(tail=np.zeros_like(image), core=np.zeros_like(image), shape=image)
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
    elif recipe.preset in BITMAP_PRESETS and recipe.shape_gain:
        shape = resize_mass(colored_bitmap_shape(recipe), radius, scale) * recipe.shape_gain
    area = np.float32(pixel_scale**2)
    return {"tail": tail / area, "core": core / area, "shape": shape / area}


def target_kernel(recipe, radius, pixel_scale=1.):
    parts = target_components(recipe, radius, pixel_scale)
    return parts["tail"] + parts["core"] + parts["shape"]
