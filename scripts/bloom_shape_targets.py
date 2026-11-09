"""Full-resolution visual targets for the built-in Bloom shapes."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def axis_taps(angle, length, taper_width, stretch, scaled_dispersion=False):
    angle = np.deg2rad(angle)
    direction = np.array([np.cos(angle), np.sin(angle)])
    r = np.arange(2.8, length, 2.8)
    taper = smoothstep((length - r) / taper_width)
    scaled = r * (144 / length)
    ramp = smoothstep(((scaled if scaled_dispersion else r) - 18) / 30)
    d = scaled[:, None] / (1 + ramp[:, None] * (np.asarray(stretch) - 1))
    weights = taper[:, None] * (.4 * np.exp(-.5 * (d / 16)**2) + .6 / (1 + (d / 34)**2))
    taps = []
    for sign in (-1, 1):
        for distance, weight in zip(r, weights, strict=True):
            x, y = -sign * distance * direction / 2
            taps.append((x, y, weight))
    return taps


def normalize(taps):
    mass = np.sum([w for _, _, w in taps], axis=0)
    return [(x, y, w / mass) for x, y, w in taps]


def rays(angles, lengths, taper_widths, stretch, center):
    taps = [(0., 0., np.full(3, center))]
    for angle, length, taper_width in zip(angles, lengths, taper_widths, strict=True):
        taps.extend(axis_taps(angle, length, taper_width, stretch))
    return normalize(taps)


def cinema_star():
    stretch = (1.025, 1., .975)
    center = np.zeros(3)
    taps = []
    for angle, length, mass in ((0, 144, .48), (45, 76, .20), (90, 50, .12), (135, 76, .20)):
        axis = normalize([(0., 0., np.ones(3)),
                          *axis_taps(angle, length, length * 2 / 9, stretch, scaled_dispersion=True)])
        center += axis[0][2] * mass
        taps.extend((x, y, w * mass) for x, y, w in axis[1:])
    return [(0., 0., center), *taps]


def prismatic_streak():
    needle = rays((0,), (144,), (32,), (1., 1., 1.), 1.)
    offsets = np.arange(-144, 145, 2)
    distance = np.abs(np.abs(offsets[:, None]) - np.array([88, 68, 48]))
    lobes = np.exp(-.5 * (distance / 7)**2) * smoothstep((21 - distance) / 3)
    lobes *= .30 / lobes.sum(axis=0)
    taps = {(x, y): w * .70 for x, y, w in needle}
    for offset, w in zip(offsets, lobes):
        if np.any(w):
            pos = (-offset / 2, 0.)
            taps[pos] = taps.get(pos, 0.) + w
    return [(x, y, w) for (x, y), w in taps.items()]


def digits_67(x, y):
    curves = [
        [(49, -29), (37, -44), (15, -37), (11, -14)],
        [(11, -14), (7, 2), (8, 19), (19, 26)],
        [(19, 26), (30, 33), (49, 27), (50, 13)],
        [(50, 13), (51, -1), (38, -8), (26, -4)],
        [(26, -4), (19, -2), (13, 2), (10, 9)],
    ]
    t = np.linspace(0, 1, 129)
    bezier = np.array([(1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3]).T
    six = np.concatenate([bezier @ np.array(curve) for curve in curves])
    six = (six - six.min(axis=0)) / np.ptp(six, axis=0) * [40, 72] + [8, -36]
    seven = np.array([[57, -36], [96, -36], [65, 36]])
    dist = np.full(x.shape, np.inf)
    for points in (six, seven):
        for a, b in zip(points[:-1], points[1:]):
            if np.array_equal(a, b):
                continue
            v = b - a
            dx, dy = x-a[0], y-a[1]
            t = np.clip((dx*v[0] + dy*v[1]) / (v @ v), 0, 1)
            np.minimum(dist, (dx-t*v[0])**2 + (dy-t*v[1])**2, out=dist)
    return np.exp(-.5 * dist / 1.35**2)


def overlaps(grid, edges):
    low = np.maximum(grid[:, None] - .5, edges[:-1])
    high = np.minimum(grid[:, None] + .5, edges[1:])
    return np.maximum(0., high - low)


def trollface(x, y):
    # Carlos Ramirez, original drawing. Source and attribution are in the scripts guide.
    path = Path(__file__).with_name("assets") / "trollface-source.png"
    if hashlib.sha256(path.read_bytes()).hexdigest() != "603b99115c372cfe0cf00b9584852dcc4462d51ef08d28b77d83dba1aa6293d9":
        raise ValueError("Unexpected Trollface source image")
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=float) / 255.
    ink = np.clip(1 - rgba[:, :, :3] @ [.2126, .7152, .0722], 0, 1) * rgba[:, :, 3]
    rows, columns = np.nonzero(ink > 1e-12)
    ink = ink[rows.min():rows.max()+1, columns.min():columns.max()+1]
    height, width = ink.shape
    scaled_height = 120 * height / width
    x_area = overlaps(x[0], np.linspace(8, 128, width + 1))
    y_area = overlaps(y[:, 0], np.linspace(-scaled_height/2, scaled_height/2, height + 1))
    return y_area @ ink @ x_area.T
