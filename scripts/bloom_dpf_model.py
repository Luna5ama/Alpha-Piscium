"""CPU model of a dyadic Bloom cascade, with the GLSL bilinear footprints."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


REFERENCE = Path(__file__).with_name("assets") / "bloom-dpf-reference.npz"
LEVELS = 8


@lru_cache(maxsize=1)
def reference_taps():
    with np.load(REFERENCE, allow_pickle=False) as archive:
        return archive["down"], archive["up"]


@lru_cache(maxsize=512)
def axis_matrix(source, destination, offset, down, start, pool):
    axis = start + np.arange(destination)[:, None] * pool + np.arange(pool)
    position = (2 * axis + .5 if down else .5 * axis - .25) + offset
    if not down:
        position = np.clip(position, 0, source - 1)
    low = np.floor(position).astype(int)
    fraction = position - low
    columns = np.stack((low, low + 1), axis=2)
    weights = np.stack((1 - fraction, fraction), axis=2) / pool
    rows = np.broadcast_to(np.arange(destination)[:, None, None], columns.shape)
    keep = (weights != 0) & (columns >= 0) & (columns < source)
    matrix = csr_matrix((weights[keep].astype(np.float32), (rows[keep], columns[keep])),
                        shape=(destination, source))
    return matrix, matrix.T.tocsr()


def axis_filter(image, matrix):
    return (matrix @ image.reshape(-1, image.shape[-1]).T).T.reshape(*image.shape[:-1], matrix.shape[0])


class Filter:
    def __init__(self, source, destination, offsets, *, down=False, start=(0, 0), pool=1):
        self.destination = destination
        self.x = tuple(axis_matrix(source[1], destination[1], float(x), down, start[1], pool) for x, _ in offsets)
        self.y = tuple(axis_matrix(source[0], destination[0], float(y), down, start[0], pool) for _, y in offsets)

    def tap(self, image, i):
        x = axis_filter(image, self.x[i][0])
        return axis_filter(x.swapaxes(-2, -1), self.y[i][0]).swapaxes(-2, -1)

    def transpose_tap(self, derivative, i):
        y = axis_filter(derivative.swapaxes(-2, -1), self.y[i][1]).swapaxes(-2, -1)
        return axis_filter(y, self.x[i][1])

    def apply(self, image, weights):
        batch = np.broadcast_shapes(image.shape[:-2], np.shape(weights)[1:])
        result = np.zeros((*batch, *self.destination), dtype=np.float32)
        for i, weight in enumerate(weights):
            result += self.tap(image, i) * weight[..., None, None]
        return result

    def backward(self, image, derivative, weights):
        dx = np.zeros_like(derivative, shape=(*derivative.shape[:-2], *image.shape[-2:]))
        dw = []
        for j, weight in enumerate(weights):
            dw.append(np.sum(derivative * self.tap(image, j), axis=(0, 2, 3), dtype=np.float64))
            dx += self.transpose_tap(derivative * weight[:, None, None], j)
        return dx, np.asarray(dw)


def downsample(image, taps, levels=LEVELS):
    """Downsample an already linearized signal. Real D1 compression is outside this model."""
    pyramid = []
    offsets, weights = taps[:, :2], taps[:, 2].astype(np.float32)
    for _ in range(levels):
        shape = image.shape[-2:]
        destination = tuple((n + 1) // 2 for n in shape)
        image = Filter(shape, destination, offsets, down=True).apply(image, weights)
        pyramid.append(image)
    return pyramid


def impulse(size, phases):
    image = np.zeros((len(phases), size, size), dtype=np.float32)
    for i, (x, y) in enumerate(phases):
        image[i, size // 2 + y, size // 2 + x] = 1
    return image


def window(size, phase, radius):
    x, y = phase
    center = size // 2
    return np.s_[center + y - radius:center + y + radius + 1,
                 center + x - radius:center + x + radius + 1]


class Cascade:
    """One reconstruction chain with local raw-level injection and per-level RGB taps."""
    def __init__(self, full_shape, pyramid, offsets, *, roi=None, pool=None):
        self.pyramid = pyramid
        shapes = [full_shape, *(level.shape[-2:] for level in pyramid)]
        self.filters = [Filter(shapes[i + 1], shapes[i], taps) for i,taps in enumerate(offsets)]
        if roi:
            start, shape = roi
            self.filters[0] = Filter(shapes[1], shape, offsets[0], start=start)
        self.pooled = None if pool is None else Filter(
            shapes[1], tuple(n // pool for n in full_shape), offsets[0], pool=pool)

    def forward(self, weights, injection, pyramid=None):
        pyramid = self.pyramid if pyramid is None else pyramid
        image = pyramid[-1] * injection[-1, :, None, None]
        inputs = [None] * len(self.filters)
        for i in reversed(range(len(self.filters))):
            inputs[i] = image
            image = self.filters[i].apply(image, weights[i])
            if i:
                image += pyramid[i - 1] * injection[i - 1, :, None, None]
        return image, inputs

    def backward(self, derivative, inputs, weights, pooled_derivative):
        dweights, dpyramid = [], []
        for i, filter in enumerate(self.filters):
            if i:
                dpyramid.append(derivative)
            derivative, dw = filter.backward(inputs[i], derivative, weights[i])
            if i == 0:
                dx, pdw = self.pooled.backward(inputs[0], pooled_derivative, weights[0])
                derivative += dx
                dw += pdw
            dweights.append(dw)
        dpyramid.append(derivative)
        return dweights, dpyramid
