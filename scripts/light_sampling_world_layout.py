"""Bounded storage for complete world prefixes, checked against a full 84-bit trie."""
import argparse
import itertools
import json
import math
from pathlib import Path

BIAS = 1 << 27


def shifts(bits):
    return ((bits + 2) // 3, (bits + 1) // 3, bits // 3)


def full_key(world):
    result = 0
    for axis, coordinate in enumerate(world):
        value = int(coordinate) + BIAS
        assert 0 <= value < 1 << 28
        for bit in range(28):
            result |= ((value >> bit) & 1) << (3 * bit + axis)
    return result


def oracle(bricks, combine, arity):
    """Independent high-to-low integer-key partition; no bounded-layout addresses."""
    bits = arity.bit_length() - 1
    def visit(rows, remaining):
        if len(rows) <= 1:
            return rows[0][1] if rows else None
        buckets = [[] for _ in range(arity)]
        for key, value in rows:
            buckets[(key >> (remaining - bits)) & (arity - 1)].append((key, value))
        return combine([visit(group, remaining - bits) for group in buckets if group])
    return visit([(full_key(p), value) for p, value in bricks.items()], 84)


class WorldLayout:
    def __init__(self, grid, arity, camera_brick):
        self.grid, self.arity = grid, arity
        self.bits = arity.bit_length() - 1
        self.levels = (3 * (grid.bit_length() - 1) + self.bits - 1) // self.bits
        self.origin = tuple(int(x) - grid // 2 for x in camera_brick)
        first = tuple(x + BIAS for x in self.origin)
        self.shift = [shifts(level * self.bits) for level in range(self.levels + 1)]
        self.low = [tuple(x >> s for x, s in zip(first, shift)) for shift in self.shift]
        self.dim = [tuple(((x + grid - 1) >> s) - (x >> s) + 1 for x, s in zip(first, shift))
                    for shift in self.shift]
        self.offset = [0]
        self.capacity = 32
        self.max_counts = [grid ** 3]
        for level in range(1, self.levels + 1):
            maximum = tuple((grid + (1 << s) - 2) // (1 << s) + 1 for s in self.shift[level])
            assert all(d <= m for d, m in zip(self.dim[level], maximum))
            self.max_counts.append(math.prod(maximum))
            self.offset.append(self.capacity)
            self.capacity += arity * self.max_counts[-1]
        assert math.prod(self.dim[-1]) <= 8

    def index(self, cell, level):
        relative = tuple(x - low for x, low in zip(cell, self.low[level]))
        assert all(0 <= x < dim for x, dim in zip(relative, self.dim[level]))
        return relative[0] + self.dim[level][0] * (relative[1] + self.dim[level][1] * relative[2])

    def ordinal(self, cell, level):
        anchor = tuple(x << s for x, s in zip(cell, self.shift[level]))
        return sum(((anchor[bit % 3] >> (bit // 3)) & 1) << i
                   for i, bit in enumerate(range(level * self.bits, (level + 1) * self.bits)))

    def address_cell(self, cell, level):
        if level == self.levels:
            return 2 + self.index(cell, level)
        parent = tuple(x >> (b - a) for x, a, b in zip(cell, self.shift[level], self.shift[level + 1]))
        return self.offset[level + 1] + self.arity * self.index(parent, level + 1) + self.ordinal(cell, level)

    def brick_address(self, world):
        assert all(low <= x < low + self.grid for low, x in zip(self.origin, world))
        return self.address_cell(tuple(x + BIAS for x in world), 0)

    def build(self, bricks, combine):
        storage, active = {}, {}
        for world, value in bricks.items():
            cell = tuple(x + BIAS for x in world)
            address = self.brick_address(world)
            assert address not in storage
            storage[address] = value
            active[cell] = value
        for level in range(1, self.levels + 1):
            delta = tuple(b - a for a, b in zip(self.shift[level - 1], self.shift[level]))
            parents = sorted({tuple(x >> s for x, s in zip(cell, delta)) for cell in active})
            active = {}
            for cell in parents:
                child_base = self.offset[level] + self.arity * self.index(cell, level)
                children = [storage[child_base + j] for j in range(self.arity) if child_base + j in storage]
                value = combine(children)
                address = self.address_cell(cell, level)
                assert address not in storage
                storage[address] = value
                active[cell] = value
        # Mirror the shader's in-place compacted top records, while the oracle uses no storage.
        current = [(cell, self.address_cell(cell, self.levels)) for cell in sorted(active, key=lambda c: self.index(c, self.levels))]
        next_child = 10
        for level in range(self.levels + 1, 84 // self.bits + 1):
            if len(current) <= 1:
                break
            old_shift, new_shift = shifts((level - 1) * self.bits), shifts(level * self.bits)
            delta = tuple(b - a for a, b in zip(old_shift, new_shift))
            parent_keys = [tuple(x >> s for x, s in zip(cell, delta)) for cell, _ in current]
            following = []
            seen = set()
            for i, parent in enumerate(parent_keys):
                if parent in seen:
                    continue
                seen.add(parent)
                indices = [j for j in range(i, len(current)) if parent_keys[j] == parent]
                def ordinal(j):
                    anchor = tuple(x << s for x, s in zip(current[j][0], old_shift))
                    return sum(((anchor[b % 3] >> (b // 3)) & 1) << k
                               for k, b in enumerate(range((level - 1) * self.bits, level * self.bits)))
                indices.sort(key=ordinal)
                children = [storage[current[j][1]] for j in indices]
                if len(children) > 1:
                    for child in children:
                        storage[next_child] = child
                        next_child += 1
                address = 2 + len(following)
                storage[address] = combine(children)
                following.append((parent, address))
            current = following
        assert len(current) <= 1 and next_child <= 24
        storage[1] = storage[current[0][1]] if current else None
        assert all(0 <= address < self.capacity for address in storage)
        return storage[1], storage


def validate():
    def combine(values):
        return values[0] if len(values) == 1 else tuple(values) if values else None
    capacities, cases = {}, 0
    for grid in (16, 32, 64):
        for arity in (2, 4, 8):
            for start in (-1048576, -grid // 2, -1, 0, 1025, 1 << 26):
                base = (start, start + 3, start - 5)
                sparse = {tuple(base[a] + 1 + ((i * (3, 5, 7)[a]) % (grid - 2)) for a in range(3)): ('L', i)
                          for i in range(12)}
                one = {tuple(x + grid // 2 for x in base): ('L', 0)}
                for bricks in ({}, one, sparse):
                    expected = oracle(bricks, combine, arity)
                    for delta in itertools.product((-1, 0, 1), repeat=3):
                        camera = [base[a] + delta[a] + grid // 2 for a in range(3)]
                        layout = WorldLayout(grid, arity, camera)
                        actual, _ = layout.build(bricks, combine)
                        assert actual == expected, (grid, arity, base, delta)
                        capacities[f'{grid}/{arity}'] = layout.capacity
                        cases += 1
    expected_capacities = {'16/2': 10570, '16/4': 7620, '16/8': 7144,
                           '32/2': 74464, '32/4': 51732, '32/8': 46448,
                           '64/2': 558854, '64/4': 380432, '64/8': 333944}
    assert capacities == expected_capacities
    return dict(cases=cases, capacities=capacities, oracle='Independent full 84-bit integer trie, ordered children, unary compression',
                coverage='G16/32/64, arity2/4/8, 162 cohorts x27 windows, empty/single/multiple, negative/zero/large coordinates and >10-bit keys')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))
