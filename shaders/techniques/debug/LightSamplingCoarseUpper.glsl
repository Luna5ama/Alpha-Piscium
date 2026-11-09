// Experimental G16 upper builder. Called by one lane after native level 3.
// Level 3 of the four-way world trie owns 4^3-brick (64^3-block) subtrees.
const uint LS_COARSE_LEVEL = 3u;
const uint LS_COARSE_MAX = 128u; // 16 consecutive bricks intersect <=5^3 cells.
const uint LS_COARSE_BINS = 12u;
shared LightSamplingNode ls_coarseAtom[LS_COARSE_MAX];
shared ivec3 ls_coarseCenter2[LS_COARSE_MAX];
shared uint ls_coarseOrder[LS_COARSE_MAX];
shared uint ls_coarseTemp[LS_COARSE_MAX];
shared uvec3 ls_coarseQueue[2u * LS_COARSE_MAX];
shared LightSamplingNode ls_coarseBins[LS_COARSE_BINS];
shared LightSamplingNode ls_coarseSuffix[LS_COARSE_BINS];
shared uint ls_coarseBinCount[LS_COARSE_BINS];

uint ls_coarseBucket(int center, int minimum, int span) {
    return min(uint((center - minimum) * int(LS_COARSE_BINS) / span), LS_COARSE_BINS - 1u);
}

float ls_coarseCost(LightSamplingNode node) {
    vec3 size = node.boundsMaxLink.xyz - node.boundsMinMass.xyz;
    return node.boundsMinMass.w * dot(size, size);
}

uint ls_coarseSplit(uint begin, uint end) {
    ivec3 low = ls_coarseCenter2[ls_coarseOrder[begin]];
    ivec3 high = low;
    for (uint i = begin + 1u; i < end; ++i) {
        ivec3 c = ls_coarseCenter2[ls_coarseOrder[i]];
        low = min(low, c);
        high = max(high, c);
    }
    float bestCost = uintBitsToFloat(0x7f800000u);
    int bestAxis = -1;
    uint bestBucket = 0u, bestLeft = 0u;
    for (int axis = 0; axis < 3; ++axis) {
        int span = high[axis] - low[axis];
        if (span == 0) continue;
        for (uint k = 0u; k < LS_COARSE_BINS; ++k) {
            ls_coarseBins[k] = ls_emptyNode();
            ls_coarseBinCount[k] = 0u;
        }
        for (uint i = begin; i < end; ++i) {
            uint atom = ls_coarseOrder[i];
            uint k = ls_coarseBucket(ls_coarseCenter2[atom][axis], low[axis], span);
            ls_coarseBins[k] = ls_merge(ls_coarseBins[k], ls_coarseAtom[atom], 0u);
            ++ls_coarseBinCount[k];
        }
        LightSamplingNode suffix = ls_emptyNode();
        for (uint k = LS_COARSE_BINS; k != 0u; --k) {
            suffix = ls_merge(suffix, ls_coarseBins[k - 1u], 0u);
            ls_coarseSuffix[k - 1u] = suffix;
        }
        LightSamplingNode prefix = ls_emptyNode();
        uint left = 0u;
        for (uint k = 0u; k + 1u < LS_COARSE_BINS; ++k) {
            prefix = ls_merge(prefix, ls_coarseBins[k], 0u);
            left += ls_coarseBinCount[k];
            if (left == 0u || left == end - begin) continue;
            float cost = ls_coarseCost(prefix) + ls_coarseCost(ls_coarseSuffix[k + 1u]);
            // Axes and bins are visited in canonical ascending order.
            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                bestBucket = k;
                bestLeft = left;
            }
        }
    }
    if (bestAxis < 0) return begin + (end - begin) / 2u;
    uint middle = begin + bestLeft;
    uint l = begin, r = middle;
    int span = high[bestAxis] - low[bestAxis];
    for (uint i = begin; i < end; ++i) {
        uint atom = ls_coarseOrder[i];
        uint bucket = ls_coarseBucket(ls_coarseCenter2[atom][bestAxis], low[bestAxis], span);
        if (bucket <= bestBucket) ls_coarseTemp[l++] = atom;
        else ls_coarseTemp[r++] = atom;
    }
    for (uint i = begin; i < end; ++i) ls_coarseOrder[i] = ls_coarseTemp[i];
    return middle;
}

bool ls_buildCoarseUpper() {
    uvec3 low, dim;
    ls_worldRange(LS_COARSE_LEVEL, low, dim);
    if (dim.x * dim.y * dim.z > LS_COARSE_MAX) return false;
    uint n = 0u;
    for (uint x = 0u; x < dim.x; ++x) {
        for (uint y = 0u; y < dim.y; ++y) {
            for (uint z = 0u; z < dim.z; ++z) {
                uvec3 cell = low + uvec3(x,y,z);
                uvec3 corner = cell << ls_worldShift(LS_COARSE_LEVEL);
                LightSamplingNode node = ls_nodes[ls_worldAddress(corner, LS_COARSE_LEVEL)];
                if (node.boundsMinMass.w == 0.0) continue;
                ls_coarseAtom[n] = node;
                ls_coarseCenter2[n] = ivec3(node.boundsMinMass.xyz) + ivec3(node.boundsMaxLink.xyz);
                ls_coarseOrder[n] = n;
                ++n;
            }
        }
    }
    if (n == 0u) return false;
    // Level 3 roots are already copied to shared. Their descendants are
    // in lower levels (or the local tree) and never use this upper range.
    // G16 provides 284 slots here; at most 248 descendants are required.
    uint base = ls_levelOffset(LS_COARSE_LEVEL + 1u);
    if (2u * n - 2u > LS_OUTER_NODE_COUNT - base) return false;
    uint next = base;
    uint tail = 1u;
    ls_coarseQueue[0] = uvec3(0u, n, 1u);
    for (uint head = 0u; head < tail; ++head) {
        uvec3 item = ls_coarseQueue[head];
        uint begin = item.x, end = item.y;
        if (end - begin == 1u) {
            ls_nodes[item.z] = ls_coarseAtom[ls_coarseOrder[begin]];
            continue;
        }
        uint first = next;
        uint middle = ls_coarseSplit(begin, end);
        uint left = middle - begin > 1u ? ls_coarseSplit(begin, middle) : begin;
        uint right = end - middle > 1u ? ls_coarseSplit(middle, end) : middle;
        if (left > begin) ls_coarseQueue[tail++] = uvec3(begin, left, next++);
        ls_coarseQueue[tail++] = uvec3(left, middle, next++);
        if (right > middle) ls_coarseQueue[tail++] = uvec3(middle, right, next++);
        ls_coarseQueue[tail++] = uvec3(right, end, next++);
        LightSamplingNode merged = ls_emptyNode();
        for (uint i = begin; i < end; ++i) merged = ls_merge(merged, ls_coarseAtom[ls_coarseOrder[i]], 0u);
        merged.boundsMaxLink.w = uintBitsToFloat(ls_childLink(first, next - first));
        ls_nodes[item.z] = merged;
    }
    ls_reserved = n; // Native diagnostic: number of roots used by this build.
    return true;
}
