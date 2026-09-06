"""Bake periodic, single-scale billows to the 128-cubed R8 cloud shape texture."""

from pathlib import Path

import numpy as np


SIZE = 128
CELLS = 4


def main():
    rng = np.random.Generator(np.random.PCG64(0xC011D))
    centers = rng.uniform(0.1, 0.9, (CELLS, CELLS, CELLS, 3)).astype(np.float32)
    volume = np.empty((SIZE, SIZE, SIZE), dtype=np.uint8)
    y, x = np.mgrid[:SIZE, :SIZE].astype(np.float32)
    positions = np.empty((SIZE, SIZE, 3), dtype=np.float32)
    positions[..., 0] = (x + 0.5) * (CELLS / SIZE)
    positions[..., 1] = (y + 0.5) * (CELLS / SIZE)

    for z in range(SIZE):
        positions[..., 2] = (z + 0.5) * (CELLS / SIZE)
        cell = np.floor(positions).astype(np.int32)
        local = positions - cell
        nearest = np.full((SIZE, SIZE), np.inf, dtype=np.float32)
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    offset = np.array((dx, dy, dz), dtype=np.int32)
                    neighbor = (cell + offset) % CELLS
                    delta = centers[neighbor[..., 2], neighbor[..., 1], neighbor[..., 0]] + offset - local
                    nearest = np.minimum(nearest, np.sum(delta * delta, axis=-1))
        volume[z] = np.rint(np.clip(1.0 - np.sqrt(nearest), 0.0, 1.0) * 255.0).astype(np.uint8)

    output = Path(__file__).resolve().parent.parent / "shaders/textures/cumulus_detail1.bin"
    output.write_bytes(volume.tobytes())
    print(f"{output}: {volume.shape}, R8, {volume.nbytes} bytes")


if __name__ == "__main__":
    main()
