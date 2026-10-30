// SPDX-License-Identifier: BSD-2-Clause
// Adapted from gilbert_d2xyz.py, jakubcerveny/gilbert commit
// 9b080a74c5e3b6fe189785c52742f83ac85ba181. Fixed 16^3 domain, iterative traversal.
// Copyright (c) 2024 abetusk
// Copyright (c) 2018, Jakub Červený
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

int ls_axisLength(ivec3 axis) {
    return abs(axis.x + axis.y + axis.z);
}

ivec3 ls_gilbertPosition(uint index) {
    int remaining = int(index);
    ivec3 origin = ivec3(0);
    ivec3 a = ivec3(16, 0, 0);
    ivec3 b = ivec3(0, 16, 0);
    ivec3 c = ivec3(0, 0, 16);
    while (true) {
        int w = ls_axisLength(a), h = ls_axisLength(b), d = ls_axisLength(c);
        ivec3 da = sign(a), db = sign(b), dc = sign(c);
        if (h == 1 && d == 1) return origin + da * remaining;
        if (w == 1 && d == 1) return origin + db * remaining;
        if (w == 1 && h == 1) return origin + dc * remaining;
        // Signed shifts match Python floor division for negative directions.
        ivec3 a2 = a >> 1, b2 = b >> 1, c2 = c >> 1;
        if ((ls_axisLength(a2) & 1) != 0 && w > 2) a2 += da;
        if ((ls_axisLength(b2) & 1) != 0 && h > 2) b2 += db;
        if ((ls_axisLength(c2) & 1) != 0 && d > 2) c2 += dc;
        int w2 = ls_axisLength(a2), h2 = ls_axisLength(b2), d2 = ls_axisLength(c2);
        ivec3 nextA, nextB, nextC;
        if (2 * w > 3 * h && 2 * w > 3 * d) {
            int size = w2 * h * d;
            if (remaining < size) {
                nextA = a2; nextB = b; nextC = c;
            } else {
                remaining -= size;
                origin += a2;
                nextA = a - a2; nextB = b; nextC = c;
            }
        } else if (3 * h > 4 * d) {
            int size0 = h2 * d * w2;
            int size1 = w * (h - h2) * d;
            if (remaining < size0) {
                nextA = b2; nextB = c; nextC = a2;
            } else if (remaining < size0 + size1) {
                remaining -= size0;
                origin += b2;
                nextA = a; nextB = b - b2; nextC = c;
            } else {
                remaining -= size0 + size1;
                origin += (a - da) + (b2 - db);
                nextA = -b2; nextB = c; nextC = -(a - a2);
            }
        } else if (3 * d > 4 * h) {
            int size0 = d2 * w2 * h;
            int size1 = w * h * (d - d2);
            if (remaining < size0) {
                nextA = c2; nextB = a2; nextC = b;
            } else if (remaining < size0 + size1) {
                remaining -= size0;
                origin += c2;
                nextA = a; nextB = b; nextC = c - c2;
            } else {
                remaining -= size0 + size1;
                origin += (a - da) + (c2 - dc);
                nextA = -c2; nextB = -(a - a2); nextC = b;
            }
        } else {
            int size0 = h2 * d2 * w2;
            int size1 = d * w2 * (h - h2);
            int size2 = w * h2 * (d - d2);
            int size3 = d * (w - w2) * (h - h2);
            if (remaining < size0) {
                nextA = b2; nextB = c2; nextC = a2;
            } else if (remaining < size0 + size1) {
                remaining -= size0;
                origin += b2;
                nextA = c; nextB = a2; nextC = b - b2;
            } else if (remaining < size0 + size1 + size2) {
                remaining -= size0 + size1;
                origin += (b2 - db) + (c - dc);
                nextA = a; nextB = -b2; nextC = -(c - c2);
            } else if (remaining < size0 + size1 + size2 + size3) {
                remaining -= size0 + size1 + size2;
                origin += (a - da) + b2 + (c - dc);
                nextA = -c; nextB = -(a - a2); nextC = b - b2;
            } else {
                remaining -= size0 + size1 + size2 + size3;
                origin += (a - da) + (b2 - db);
                nextA = -b2; nextB = c2; nextC = -(a - a2);
            }
        }
        a = nextA; b = nextB; c = nextC;
    }
}
