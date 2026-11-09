// Rebuild only the native coarse roots; their own descendants stay in place.
// Four-way levels consume two Morton bits, so G32 level 5 has one wider axis.
#if VOXEL_GRID_SIZE == 64
const uint LS_COARSE_LEVEL = 6u; // 256^3 blocks, at most 5^3 roots.
#elif VOXEL_GRID_SIZE == 32
const uint LS_COARSE_LEVEL = 5u; // 256x128x128 blocks, at most 3x5x5 roots.
#else
const uint LS_COARSE_LEVEL = 3u; // 64^3 blocks, at most 5^3 roots.
#endif
const uint LS_COARSE_MAX = 128u;
const uint LS_COARSE_KEYS = 512u; // Three bits per group coordinate.
shared LightSamplingNode ls_coarseAtom[LS_COARSE_MAX];
shared uvec3 ls_coarseCells[LS_COARSE_MAX];
shared uint ls_coarseDense[LS_COARSE_KEYS];
shared uvec2 ls_coarseSorted[LS_COARSE_MAX]; // Morton key and atom index
shared uvec3 ls_coarseQueue[2u * LS_COARSE_MAX];

void ls_initCoarseAnchor() {
    for (uint i = gl_LocalInvocationID.x; i < LS_COARSE_KEYS; i += gl_WorkGroupSize.x) ls_coarseDense[i] = LS_INVALID;
    barrier();
}

bool ls_buildCoarseUpper() {
    uvec3 low, dim;
    ls_worldRange(LS_COARSE_LEVEL, low, dim);
    if (dim.x * dim.y * dim.z > LS_COARSE_MAX || any(greaterThan(dim, uvec3(5u)))) return false;
    uvec3 anchor = uvec3(0xffffffffu);
    uint n = 0u;
    for (uint x = 0u; x < dim.x; ++x) {
        for (uint y = 0u; y < dim.y; ++y) {
            for (uint z = 0u; z < dim.z; ++z) {
                uvec3 cell = low + uvec3(x,y,z);
                LightSamplingNode node = ls_nodes[ls_worldAddress(cell << ls_worldShift(LS_COARSE_LEVEL), LS_COARSE_LEVEL)];
                if (node.boundsMinMass.w == 0.0) continue;
                ls_coarseAtom[n] = node;
                ls_coarseCells[n++] = cell;
                anchor = min(anchor, cell);
            }
        }
    }
    if (n == 0u) return false;
    uint base = ls_levelOffset(LS_COARSE_LEVEL + 1u);
    if (2u * n - 2u > LS_OUTER_NODE_COUNT - base) return false;
    for (uint i = 0u; i < n; ++i) ls_coarseDense[morton3D_30bEncode(ls_coarseCells[i] - anchor)] = i;
    uint sortedCount = 0u;
    for (uint key = 0u; key < LS_COARSE_KEYS; ++key) {
        uint i = ls_coarseDense[key];
        if (i != LS_INVALID) ls_coarseSorted[sortedCount++] = uvec2(key, i);
    }
    if (sortedCount != n) return false;
    uint next = base, tail = 1u;
    ls_coarseQueue[0] = uvec3(0u, n, 1u);
    for (uint head = 0u; head < tail; ++head) {
        uvec3 item = ls_coarseQueue[head];
        uint begin = item.x, end = item.y;
        if (end - begin == 1u) {
            ls_nodes[item.z] = ls_coarseAtom[ls_coarseSorted[begin].y];
            continue;
        }
        uint first = next;
        uint differing = ls_coarseSorted[begin].x ^ ls_coarseSorted[end - 1u].x;
        uint shift = uint(findMSB(differing)) & ~1u;
        for (uint start = begin; start < end;) {
            uint digit = (ls_coarseSorted[start].x >> shift) & 3u;
            uint stop = start + 1u;
            while (stop < end && ((ls_coarseSorted[stop].x >> shift) & 3u) == digit) ++stop;
            ls_coarseQueue[tail++] = uvec3(start, stop, next++);
            start = stop;
        }
        LightSamplingNode node = ls_emptyNode();
        for (uint i = begin; i < end; ++i) node = ls_merge(node, ls_coarseAtom[ls_coarseSorted[i].y], 0u);
        node.boundsMaxLink.w = uintBitsToFloat(ls_childLink(first, next - first));
        ls_nodes[item.z] = node;
    }
    ls_reserved = n;
    return true;
}
