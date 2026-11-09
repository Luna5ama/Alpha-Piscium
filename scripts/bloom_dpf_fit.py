"""Fit one RGB Bloom target to the output of a sparse multilevel cascade."""
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, kron, vstack

from bloom_dpf_model import LEVELS, Cascade, Filter, axis_filter, downsample, impulse, reference_taps
from bloom_dpf_target import (
    LOBE_REFERENCE_AREA, PRESETS, Recipe, lobe_channels, lobe_extent, lobe_fit_geometry,
    lobe_longitudinal, preset_seed_taps, reference_shape_tables, reference_tail, resize_mass, target_components,
)


TRAIN_PHASES = ((-96, -71), (-37, 86), (34, 24), (107, -43))
HELD_PHASES = ((-117, 3), (-64, -96), (11, 64), (90, -125))


def stage_offsets(signal, budget):
    """Rank the complete target residual on each stage's own lattice."""
    offsets = []
    for i, extra in enumerate(budget):
        step = 2 ** (i + 1)
        base = [[0.,0.]]
        if not extra or not np.any(signal):
            offsets.append(np.asarray(base))
            continue
        center = (signal.shape[-1] // 2 + step - 1) // step
        mass = resize_mass(signal, center, 1 / step)
        yy, xx = np.indices(mass.shape) - center
        mass *= np.sqrt(1 + np.hypot(xx, yy))
        mass[center,center] = 0
        for _ in range(extra):
            iy, ix = np.unravel_index(mass.argmax(), mass.shape)
            if mass[iy, ix] <= 0:
                break
            x, y = ix - center, iy - center
            base.append([-x, -y])
            mass[(xx-x)**2 + (yy-y)**2 <= 2.5**2] = 0
        offsets.append(np.asarray(base))
    return offsets


def target_mass(recipe, complete):
    if recipe.lobes is not None:
        return complete.sum(axis=(1,2),dtype=np.float64)
    shape_gain = reference_shape_tables()[recipe.preset][0] if recipe.preset != "standard" else 0.
    return recipe.tail_gain + recipe.core_gain + recipe.shape_gain * shape_gain


def stage_signal(complete, mass, outer_radius):
    base = resize_mass(reference_tail(),outer_radius,1.)
    if np.ndim(mass):
        mass = mass[:,None,None]
    return np.max(np.abs(complete - mass*base),axis=0)


def fit_geometry(recipe, radius, outer_radius):
    if recipe.lobes is None:
        return 160 if radius is None else radius, 768 if outer_radius is None else outer_radius, 144*recipe.size
    shape_limit, support = lobe_fit_geometry(recipe)
    radius = max(radius or 160, math.ceil(shape_limit)+16)
    return radius, max(outer_radius or 0,support,radius+32), shape_limit


def measure(output, target, pooled, pooled_target, outer_mask, regions, phases, radius, pool, size, shape_limit):
    axes = np.arange(-radius, radius+1)
    yy, xx = np.meshgrid(axes, axes, indexing="ij")
    distance = np.hypot(xx, yy)
    masks = {
        "core": distance <= 6 * size,
        "shape_scale": (distance > 6 * size) & (distance <= shape_limit),
        "outer": distance > shape_limit,
    }
    entries = []
    for i, (region, phase) in enumerate(zip(regions, phases, strict=True)):
        cropped = output[region]
        error = cropped - target
        entry = {"phase": phase, "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(target)),
                 "mass_rgb": (pooled[i].sum(axis=(1, 2), dtype=np.float64)*pool**2).tolist()}
        for name, mask in masks.items():
            entry[name + "_relative_l2"] = float(np.linalg.norm(error[:, mask]) / max(np.linalg.norm(target[:, mask]), 1e-20))
        mask = outer_mask[i]
        pred, expected = pooled[i][:, mask], pooled_target[i][:, mask]
        entry["pooled_outer_relative_l2"] = float(np.linalg.norm(pred-expected) / max(np.linalg.norm(expected), 1e-20))
        entry["outer_mass_rgb"] = (pred.sum(axis=1, dtype=np.float64)*pool**2).tolist()
        entry["outer_target_mass_rgb"] = (expected.sum(axis=1, dtype=np.float64)*pool**2).tolist()
        centroid = np.stack([(cropped * a).sum(axis=(1, 2), dtype=np.float64) for a in (xx, yy)], axis=1)
        entry["centroid_rgb_xy_in_fine_window"] = (centroid / cropped.sum(axis=(1,2))[:,None]).tolist()
        entries.append(entry)
    return entries


def loss_weights(parts, radius, size, shape_limit):
    target = parts["tail"] + parts["core"] + parts["shape"]
    axes = np.arange(-radius, radius+1)
    distance = np.hypot(axes[:, None], axes)
    energy = np.sum(target**2, dtype=np.float64)
    weights = np.ones_like(target) / energy
    for mask, share in ((distance <= 6 * size, .25), ((distance > 6 * size) & (distance <= shape_limit), .5),
                        (distance > shape_limit, .15)):
        region_energy = np.sum(target[:, mask]**2, dtype=np.float64)
        weights[:, mask] += share / max(region_energy, energy * 1e-4)
    shape = parts["shape"]
    if np.any(shape):
        coverage = shape / np.maximum(target, 1e-12)
        weights += .35 * coverage / max(np.sum(target**2 * coverage, dtype=np.float64), energy * 1e-4)
    return target, weights


def readout_geometry(side, phases, radius):
    center = side // 2
    x, y = np.asarray(phases).T
    lo_x, lo_y = int(min(x)) - radius, int(min(y)) - radius
    shape = (int(max(y)) + radius - lo_y + 1, int(max(x)) + radius - lo_x + 1)
    start = (center + lo_y, center + lo_x)
    regions = [(i, slice(None), slice(v-lo_y-radius,v-lo_y+radius+1),
                slice(u-lo_x-radius,u-lo_x+radius+1)) for i,(u,v) in enumerate(phases)]
    return (start, shape), regions


def coarse_target(target, side, phases, pool, inner_radius):
    radius = target.shape[-1] // 2
    axis = np.arange(2*radius + 1)
    coarse = np.arange(side // pool)*pool + (pool-1)*.5 - side//2
    images, masks = [], []
    for x,y in phases:
        matrices = [csr_matrix((np.full(len(axis), 1/pool, dtype=np.float32),
                     ((side//2 + shift - radius + axis)//pool, axis)),
                     shape=(side//pool,len(axis))) for shift in (x,y)]
        horizontal = axis_filter(target, matrices[0])
        images.append(axis_filter(horizontal.swapaxes(-2,-1), matrices[1]).swapaxes(-2,-1))
        dx, dy = coarse-x, coarse-y
        masks.append((np.hypot(dx[None,:],dy[:,None]) > inner_radius)
                     & (np.maximum(np.abs(dx[None,:]),np.abs(dy[:,None])) < radius - pool))
    return np.stack(images), np.stack(masks)


def coarse_loss(output, target, mask, pool, mass):
    derivative = np.zeros_like(output)
    loss = 0.
    for i,m in enumerate(mask):
        error = output[i][:,m] - target[i][:,m]
        norm = max(float(np.sum(target[i][:,m]**2)), 1e-12)
        scale = .30 / (len(mask) * norm)
        loss += .5 * scale * np.sum(error**2, dtype=np.float64)
        error_mass = error.sum(axis=1, dtype=np.float64) * pool**2
        mass_scale = .10 / (len(mask) * mass**2)
        loss += .5 * np.sum(mass_scale * error_mass**2)
        derivative[i][:,m] = error*scale + (mass_scale * pool**2 * error_mass)[:,None]
    return loss, derivative


class SparseInjection:
    """Same-level bilinear stencil projected through a fixed raw impulse."""
    def __init__(self, raw, offsets):
        self.shape = raw.shape
        phases, h, w = raw.shape
        phase, y, x = np.nonzero(raw)
        source = raw[phase, y, x]
        rows, cols, values = [], [], []
        for i,(ox,oy) in enumerate(offsets):
            ix, iy = int(np.floor(ox)), int(np.floor(oy))
            for dx, u in ((0,ix+1-ox),(1,ox-ix)):
                for dy, v in ((0,iy+1-oy),(1,oy-iy)):
                    if not u*v:
                        continue
                    xx, yy = x-ix-dx, y-iy-dy
                    valid = (xx>=0)&(xx<w)&(yy>=0)&(yy<h)
                    row = (phase[valid]*h+yy[valid])*w+xx[valid]
                    rows.extend(row)
                    cols.extend(np.full(len(row),i))
                    values.extend(source[valid]*(u*v))
        self.matrix = csr_matrix((np.asarray(values,dtype=np.float32),(rows,cols)),
                                 shape=(phases*h*w,len(offsets)))
        self.adjoint = self.matrix.T.tocsr()

    def apply(self, rgb):
        return (self.matrix @ rgb).reshape(*self.shape,3).transpose(0,3,1,2)

    def gradient(self, derivative):
        return self.adjoint @ derivative.transpose(0,2,3,1).reshape(-1,3)


# Offsets and RGB weights must fit in the minimum OpenGL compute shared memory.
MAX_INJECTION_TAPS = 1024


def injection_variances(down, up, first_center):
    """Phase-averaged horizontal variance after a unit injection at each level."""
    offset = down[:,0]
    frac = (offset+.5)%1
    down_variance = np.dot(down[:,2],offset**2+frac*(1-frac))

    # Measure the actual bilinear footprint for an impulse at source texel zero.
    radius = math.ceil(2*(float(np.abs(up[:,0]).max())+1))+1
    x = np.arange(-radius,radius+1)
    tent = np.maximum(1-np.abs(.5*x[:,None]-.25+up[:,0]),0)
    up_variance = np.sum(tent*(x[:,None]-.5)**2*up[:,2])/2
    first_variance = .75*first_center + up_variance*(1-first_center)
    # Each subsequent footprint lives on a grid with twice the spacing.
    pixel_variance = 4.**np.arange(LEVELS)
    return down_variance*np.cumsum(pixel_variance) + first_variance + np.r_[0,up_variance*np.cumsum(pixel_variance[1:])]


def lobe_injection_seed(recipe, variances):
    """Sample each analytic lobe at the level that can still resolve its cross section."""
    profiles = recipe.resolved()
    levels = [{} for _ in range(LEVELS)]
    if not recipe.shape_gain:
        return levels
    for lobe in recipe.lobes:
        if not lobe.gain:
            continue
        width = lobe.width*recipe.size
        length = lobe.length*recipe.size
        power = lobe.profile
        sigma = width*math.sqrt(2**(2/power)*math.gamma(3/power)/math.gamma(1/power))
        level = min(LEVELS-1,max(1,int(np.searchsorted(variances,sigma**2,side="right"))))
        step = 2**level
        spacing = 1.35*step
        channels = tuple(lobe_channels(lobe,recipe.size,profiles["camera"],profiles["prism"]))
        fraction = .30*profiles["prism"]
        reach,_ = lobe_extent(channels,width,profiles["prism"],4)
        gate = min(width,.25*max(x[0] for x in channels))
        along = np.arange(-math.ceil(2*gate/spacing),math.ceil(reach/spacing)+1)*spacing
        weights = np.stack([lobe_longitudinal(along,std,center,band,min(width,.25*std),fraction)
                            for std,center,band in channels],axis=1)
        transverse_area = width*2**(1+1/power)*math.gamma(1+1/power)
        mass = lobe.gain*recipe.shape_gain*(length/3)/LOBE_REFERENCE_AREA*math.sqrt(math.tau)*transverse_area
        weights *= mass/weights.sum(axis=0)

        spread = math.sqrt(max(sigma**2-variances[level-1],0.))
        rows = min(3,math.ceil(2.25*spread/step)) if spread >= .35*step else 0
        if rows and spread**2 <= .625*step**2:
            rows = 1
        cross = step*np.arange(-rows,rows+1)
        if rows == 1:
            side_weight = .5*(spread/step)**2
            cross_weight = np.array([side_weight,1-2*side_weight,side_weight])
        else:
            cross_weight = np.exp(-.5*(np.abs(cross)/spread)**power) if rows else np.ones(1)
            cross_weight /= cross_weight.sum()
        angle = math.radians(lobe.angle)
        c,s = math.cos(angle),math.sin(angle)
        taps = levels[level-1]
        for u,weight in zip(along,weights,strict=True):
            for v,factor in zip(cross,cross_weight,strict=True):
                value = weight*factor
                if value.max() >= mass*2e-5:
                    offset = (-float(u*c-v*s)/step,-float(u*s+v*c)/step)
                    taps[offset] = taps.get(offset,0.) + value
    return levels


def filtered_injection_seed(recipe, signal, down, up, first_center):
    # Same-level offsets use the raw level's texel spacing.
    geometry = stage_offsets(signal,(12,12,8,6,4,2,1,0))
    variances = injection_variances(down,up,first_center)
    authored = lobe_injection_seed(recipe,variances) if recipe.lobes is not None else [{} for _ in range(LEVELS)]
    if recipe.lobes is None and recipe.preset != "standard" and recipe.shape_gain:
        for x,y,*rgb in preset_seed_taps(recipe):
            xy = (x,y)
            authored[0][xy] = authored[0].get(xy,0.) + np.asarray(rgb)
    if recipe.core_gain:
        core_variance = (recipe.core_sigma*recipe.size)**2
        level = int(np.argmin(np.abs(np.log(variances/core_variance))))
        authored[level][(0.,0.)] = authored[level].get((0.,0.),0.) + recipe.core_gain
    result = []
    for i, positions in enumerate(geometry):
        residual_weight = .00002 if recipe.lobes is not None else .001
        initial = {tuple(xy):np.full(3,residual_weight) for xy in positions}
        initial[(0.,0.)] = np.full(3,recipe.tail_gain/LEVELS + residual_weight)
        for xy,weight in authored[i].items():
            initial[xy] = initial.get(xy,0.) + weight
        if i == 0 and recipe.core_gain:
            for x in (-1.,0.,1.):
                for y in (-1.,0.,1.):
                    initial.setdefault((x,y),np.full(3,residual_weight))
        if len(initial) > MAX_INJECTION_TAPS:
            raise ValueError(f"Bloom level {i+1} requires {len(initial)} taps; the shared-memory limit is {MAX_INJECTION_TAPS}")
        result.append((np.array(list(initial)), np.array(list(initial.values()))))
    return result


def refine_center(raw, up_offsets, up_weights, side, regions, image, target, offsets, weights):
    """Refit the near-origin D1 tap group at fixed per-channel mass."""
    radius = 16
    roi, center_regions = readout_geometry(side,TRAIN_PHASES,radius)
    center = target.shape[-1]//2
    crop = np.s_[...,center-radius:center+radius+1,center-radius:center+radius+1]
    desired = target[crop]
    current = np.stack([image[r][crop] for r in regions])
    _,y,x = np.nonzero(raw)
    origin = (y.min()-6,x.min()-6)
    small = raw[:,origin[0]:y.max()+7,origin[1]:x.max()+7]
    op = SparseInjection(small,offsets)
    basis = (op.matrix @ np.eye(len(offsets),dtype=np.float32)).reshape(*small.shape,len(offsets)).transpose(0,3,1,2)
    project = Filter(small.shape[-2:],roi[1],np.asarray(up_offsets)-origin[::-1],start=roi[0])
    axis = np.arange(-radius,radius+1)
    local = 1+2*(np.hypot(axis[:,None],axis)<=3)
    importance = np.tile(local.ravel(),len(regions))
    refined = np.empty_like(weights,dtype=np.float64)
    for channel in range(3):
        projected = project.apply(basis,up_weights[:,channel,None])
        mat = np.stack([projected[b,:,sy,sx] for b,_,sy,sx in center_regions])
        mat = mat.transpose(0,2,3,1).reshape(-1,len(offsets))
        total = weights[:,channel].sum(dtype=np.float64)
        goal = (desired[channel]-current[:,channel]).ravel()+mat@weights[:,channel]
        mat = mat.astype(np.float64)*total
        weighted = mat*importance[:,None]

        def loss(p):
            error = mat@p-goal
            return .5*float(np.dot(importance*error,error))*1e4,(weighted.T@error)*1e4

        start = weights[:,channel]/total
        constraint = {"type":"eq","fun":lambda p:p.sum()-1,"jac":lambda p:np.ones_like(p)}
        fit = minimize(loss,start,jac=True,method="SLSQP",bounds=[(0.,1.)]*len(offsets),
                       constraints=[constraint],options={"maxiter":60,"ftol":1e-10})
        probabilities = np.maximum(fit.x,0.)
        if not np.isfinite(probabilities).all() or not np.any(probabilities):
            raise RuntimeError("The near-origin tap fit returned invalid weights")
        refined[:,channel] = probabilities/probabilities.sum()
    return refined


def refine_fine_injection(raw, side, offsets, weights, up_offsets, up_weights, current, target, importance, cancelled):
    """Fit the D1 stencil against the complete response with the rest of the cascade fixed."""
    h, w = target.shape[-2:]
    footprint = math.ceil(np.max(np.abs(up_offsets))) + 2
    matrices = [[], [], []]
    for i,(x,y) in enumerate(TRAIN_PHASES):
        if cancelled and cancelled():
            raise InterruptedError("Fit cancelled")
        sy, sx = side//2+y-h//2, side//2+x-w//2
        y0, x0 = sy//2-footprint, sx//2-footprint
        small = raw[i:i+1,y0:y0+h//2+2*footprint+2,x0:x0+w//2+2*footprint+2]
        injection = SparseInjection(small,offsets).matrix
        project = Filter(small.shape[-2:],(h,w),up_offsets,start=(sy-2*y0,sx-2*x0))
        taps = [kron(ay[0],ax[0],format="csr") @ injection
                for ay,ax in zip(project.y,project.x,strict=True)]
        for channel in range(3):
            matrices[channel].append(sum(tap*gain for tap,gain in zip(taps,up_weights[:,channel],strict=True)))

    refined = np.empty_like(weights,dtype=np.float64)
    steps = []
    for channel,phases in enumerate(matrices):
        projected = vstack(phases,format="csr").astype(np.float64)
        active = np.diff(projected.indptr) != 0
        projected = projected[active]
        root = np.tile(np.sqrt(importance[channel]).ravel(),len(TRAIN_PHASES))[active]
        previous = weights[:,channel].astype(np.float64)
        total = previous.sum()
        wanted = projected @ previous + (target[channel]-current[:,channel]).ravel()[active]
        weighted = projected.multiply(root[:,None]).tocsr()
        adjoint = weighted.T.tocsr()
        wanted *= root

        def loss(x):
            if cancelled and cancelled():
                raise InterruptedError("Fit cancelled")
            normalized = x/x.sum()
            error = weighted @ (normalized*total)-wanted
            gradient = (adjoint @ error)*(total/x.sum())
            return .5*float(error @ error),gradient-normalized @ gradient

        start = previous*(len(previous)/total)
        fit = minimize(loss,start,jac=True,method="L-BFGS-B",bounds=[(0.,None)]*len(start),
                       options={"maxiter":150,"ftol":1e-12,"gtol":1e-10,"maxcor":16})
        refined[:,channel] = fit.x/fit.x.sum()
        steps.append(fit.nit)
    return refined,steps


def fit_filtered_recipe(recipe, *, radius=None, outer_radius=None, iterations=18, progress=None,
                        cancelled=None, seed=None, return_outer=False):
    """Fit the complete target with same-level stencils inside the reconstruction chain.

    Raw D_i stays alive until the reconstruction of level i. D1 and the
    existing downsample cascade are fixed in this candidate. The output has
    one set of RGB upsample coefficients and one combined injection per level.
    """
    radius, outer_radius, shape_limit = fit_geometry(recipe,radius,outer_radius)
    full = target_components(recipe,outer_radius)
    central=np.s_[...,outer_radius-radius:outer_radius+radius+1,outer_radius-radius:outer_radius+radius+1]
    complete=full["tail"]+full["core"]+full["shape"]
    mass=target_mass(recipe,complete)
    if np.any(mass <= 0):
        raise ValueError("The complete target must have positive mass in each channel")
    parts={k:v[central] for k,v in full.items()}
    target,importance=loss_weights(parts,radius,recipe.size,shape_limit)
    floor=max(float(target.max())*2e-4,1e-12)
    weak=1/np.square(np.maximum(target,floor))
    importance+=.65*weak/np.sum(target**2*weak,dtype=np.float64)
    down,up=reference_taps()
    if seed is None:
        signal=stage_signal(complete,mass,outer_radius)
        up_offsets=[np.vstack(([[0.,0.]],up[:,:2])),*[up[:,:2]]*(LEVELS-1)]
        first_center=.75 if recipe.preset != "standard" else .05
        w0=[np.repeat(np.r_[first_center,(1-first_center)*up[:,2]][:,None],3,axis=1)]
        w0.extend([np.repeat(up[:,2:3],3,axis=1)]*(LEVELS-1))
        injections=filtered_injection_seed(recipe,signal,down,up,first_center)
    else:
        if seed["recipe_sha256"] != recipe.digest() or seed["topology"] != "filtered-injection":
            raise ValueError("Filtered refit seed belongs to a different recipe or topology")
        up_offsets=[np.asarray(stage["offsets"]) for stage in seed["upsample"]]
        w0=[np.asarray(stage["rgb_weights"]) for stage in seed["upsample"]]
        injections=[(np.asarray(stage["offsets"]),np.asarray(stage["rgb_weights"])/mass)
                    for stage in seed["injection"]]
    all_offsets=[xy for xy,_ in injections]
    initial=[np.concatenate([w for _,w in injections]),*w0]
    widths=np.cumsum([0,*(a.size for a in initial)])
    splits=np.cumsum([len(xy) for xy in all_offsets])[:-1]
    x0=np.concatenate([a.ravel() for a in initial])
    all_phases=TRAIN_PHASES+HELD_PHASES
    margin=max(max(abs(x),abs(y)) for x,y in all_phases)
    side=((2*(outer_radius+margin+8)+255)//256)*256
    pool=8
    ones=np.ones((LEVELS,3),dtype=np.float32)

    def prepare(phases):
        roi,regions=readout_geometry(side,phases,radius)
        raw=downsample(impulse(side,phases),down)
        operators=[SparseInjection(d,xy) for d,xy in zip(raw,all_offsets,strict=True)]
        # Set the layout without filtering zero-filled placeholder levels.
        pyramid=[d[:,None] for d in raw]
        model=Cascade((side,side),pyramid,up_offsets,roi=roi,pool=pool)
        coarse,mask=coarse_target(complete,side,phases,pool,shape_limit)
        return model,operators,regions,coarse,mask

    def predict(x,model,operators):
        raw=[x[widths[i]:widths[i+1]].reshape(a.shape) for i,a in enumerate(initial)]
        unit=[a/a.sum(axis=0) for a in raw]
        weights=[w.astype(np.float32) for w in unit[1:]]
        combined=np.split((mass*unit[0]).astype(np.float32),splits)
        injected=[op.apply(w) for op,w in zip(operators,combined,strict=True)]
        image,inputs=model.forward(weights,ones,injected)
        pooled=model.pooled.apply(inputs[0],weights[0])
        return image,pooled,(raw,unit,weights,combined,injected,inputs)

    model,operators,regions,coarse,mask=prepare(TRAIN_PHASES)
    history=[]
    def objective(x):
        if cancelled and cancelled():
            raise InterruptedError("Fit cancelled")
        image,pooled,state=predict(x,model,operators)
        raw,unit,weights,_,injected,inputs=state
        grad=np.zeros_like(image)
        loss=0.
        for r in regions:
            diff=image[r]-target
            grad[r]=diff*importance/len(regions)
            loss+=.5*np.sum(grad[r]*diff,dtype=np.float64)
        outer,pgrad=coarse_loss(pooled,coarse,mask,pool,mass)
        dw,dp=model.backward(grad,inputs,weights,pgrad)
        dv=mass*np.concatenate([op.gradient(g) for op,g in zip(operators,dp,strict=True)])
        gradient=[(d-np.sum(d*u,axis=0))/a.sum(axis=0) for d,u,a in zip([dv,*dw],unit,raw,strict=True)]
        if not history:
            history.append({"iteration":0,"loss":float(loss+outer)})
        return loss+outer,np.concatenate([v.ravel() for v in gradient])

    def callback(intermediate_result):
        n=len(history)
        history.append({"iteration":n,"loss":float(intermediate_result.fun)})
        if progress:
            progress(n,float(intermediate_result.fun))

    bounds = [(1e-9,8.)]*len(x0)
    def optimize(start, budget):
        return minimize(objective,start,jac=True,method="L-BFGS-B",bounds=bounds,callback=callback,
                        options={"maxiter":budget,"ftol":1e-7,"maxls":15})

    result = optimize(x0,iterations)
    iteration_count = result.nit
    refinement_budget = 0
    if seed is None and recipe.core_gain:
        current,_,state = predict(result.x,model,operators)
        _,_,up_weights,combined,_,_ = state
        near = np.linalg.norm(all_offsets[0],axis=1)<=2.5
        probabilities = refine_center(model.pyramid[0][:,0],up_offsets[0],up_weights[0],side,regions,
                                      current,target,all_offsets[0][near],combined[0][near])
        start = result.x.copy()
        local = start[:len(all_offsets[0])*3].reshape(-1,3)
        local[near] = np.maximum(probabilities*local[near].sum(axis=0),1e-9)
        if objective(start)[0] < result.fun:
            refinement_budget = 6
            result = optimize(start,refinement_budget)
            iteration_count += result.nit
    chosen = result.x
    final_loss = float(result.fun)
    fine_refinement = None
    if recipe.lobes is None and np.any(parts["shape"]):
        current,_,state = predict(chosen,model,operators)
        _,_,up_weights,combined,_,_ = state
        coverage = np.max(parts["shape"]/np.maximum(target,floor),axis=0)
        shape_weak = maximum_filter(coverage,size=7)*weak
        fine_importance = importance+2.5*shape_weak/np.sum(target**2*shape_weak,dtype=np.float64)
        probabilities,steps = refine_fine_injection(
            model.pyramid[0][:,0],side,all_offsets[0],combined[0],up_offsets[0],up_weights[0],
            np.stack([current[r] for r in regions]),target,fine_importance,cancelled)
        start = chosen.copy()
        local = start[:len(all_offsets[0])*3].reshape(-1,3)
        local[:] = probabilities*local.sum(axis=0)
        fine_loss = float(objective(start)[0])
        fine_refinement = {"iterations_rgb":steps,"accepted":fine_loss < final_loss,
                           "full_loss_before":final_loss,"full_loss_after":fine_loss}
        if fine_refinement["accepted"]:
            chosen,final_loss = start,fine_loss
    model,operators,regions,coarse,mask=prepare(HELD_PHASES)
    image,pooled,state=predict(chosen,model,operators)
    start,baseline_pooled,_=predict(x0,model,operators)
    _,_,weights,combined,injected,_=state
    if not np.isfinite(image).all() or image.min()<0:
        raise RuntimeError("Nonfinite or negative fitted response")
    metrics=lambda i,p:measure(i,target,p,coarse,mask,regions,HELD_PHASES,radius,pool,recipe.size,shape_limit)
    stages=lambda xy,w:[{"offsets":a.tolist(),"rgb_weights":b.tolist()} for a,b in zip(xy,w,strict=True)]
    breakdown=[]
    for i in range(LEVELS):
        x=injected[i]
        for j in range(i,0,-1):
            x=model.filters[j].apply(x,weights[j])
        pooled_level=model.pooled.apply(x,weights[0])
        breakdown.append({"level":i+1,"tap_count":len(all_offsets[i]),
                          "injection_rgb":combined[i].sum(axis=0,dtype=np.float64).tolist(),
                          "outer_mass_rgb":np.mean([v[:,m].sum(axis=1,dtype=np.float64)*pool**2
                                                    for v,m in zip(pooled_level,mask,strict=True)],axis=0).tolist()})
    responses=np.stack([image[r] for r in regions])
    readout={"fine":responses,"pooled":pooled,"phases":HELD_PHASES,"side":side,"pool":pool} if return_outer else responses
    return {
        "topology":"filtered-injection","recipe":recipe.resolved(),"recipe_sha256":recipe.digest(),"levels":LEVELS,
        "training_phases":TRAIN_PHASES,"holdout_phases":HELD_PHASES,"fine_radius":radius,"outer_radius":outer_radius,
        "outer_pool":pool,"simulation_side":side,"trace":history,"loss":final_loss,
        **({"fine_injection_refinement":fine_refinement} if fine_refinement is not None else {}),
        "iterations":iteration_count,"iteration_limit":iterations+refinement_budget,"converged":bool(result.success),
        "solver":str(result.message),"initial_holdout":metrics(start,baseline_pooled),
        "holdout":metrics(image,pooled),"layers":breakdown,
        **({"desired_mass_rgb":mass.tolist()} if recipe.lobes is not None else {"desired_mass_per_channel":mass}),
        "target_mass_rgb_in_outer_canvas":complete.sum(axis=(1,2),dtype=np.float64).tolist(),
        "downsample":stages([down[:,:2]]*LEVELS,[np.repeat(down[:,2:3],3,axis=1)]*LEVELS),
        "upsample":stages(up_offsets,weights),"injection":stages(all_offsets,combined),
        "weighted_bilinear_fetches_per_full_res_pixel":sum(
            (len(down)+len(xy))/4**(i+1)+len(u)/4**i
            for i,(xy,u) in enumerate(zip(all_offsets,up_offsets,strict=True))),
    },readout

def shader_coefficients(recipe, fitted):
    """Emit stage taps for the filtered-injection shader consumer."""
    if fitted["recipe_sha256"] != recipe.digest() or fitted["topology"] != "filtered-injection":
        raise ValueError("Fit coefficients are stale or use a different topology")

    def vec2(offset):
        x,y = offset
        return f"vec2({x:.9f}, {-y:.9f})"

    def vec3(rgb):
        return "vec3(" + ", ".join(f"{v:.9f}" for v in rgb) + ")"

    def filter_lines(stage):
        return [f"    value += bloom_sample(center + {vec2(xy)}, tile) * {vec3(rgb)};"
                for xy,rgb in zip(stage["offsets"],stage["rgb_weights"],strict=True)]

    lines = ["// Bloom DPF recipe " + recipe.digest(),
             "vec3 bloom_filter(vec2 center, vec4 tile) {",
             "    vec3 value = vec3(0.0);", "#ifdef BLOOM_DOWNSAMPLE"]
    down = fitted["downsample"]
    if any(stage != down[0] for stage in down[1:]):
        raise ValueError("Filtered-injection downsample must use the same taps at every level")
    lines.extend(filter_lines(down[0]))
    lines.append("#else")
    for level,stage in enumerate(fitted["upsample"],1):
        lines.append(f"#{'if' if level == 1 else 'elif'} BLOOM_PASS == {level}")
        lines.extend(filter_lines(stage))
    lines.extend(["#endif", "#endif", "    return value;", "}", ""])

    injection = fitted["injection"]
    for level,stage in enumerate(injection[:-1],2):
        pairs = list(zip(stage["offsets"],stage["rgb_weights"],strict=True))
        lines.append(f"#{'if' if level == 2 else 'elif'} !defined(BLOOM_DOWNSAMPLE) && BLOOM_PASS == {level}")
        if len(pairs) > MAX_INJECTION_TAPS:
            raise ValueError("An injection stage exceeds the shader's shared tap capacity")
        if len(pairs) == 1:
            xy,rgb = pairs[0]
            lines.extend(["void bloom_prepareInjection() {}",
                          "vec3 bloom_injection(vec2 center, vec4 tile) {",
                          f"    return bloom_sampleRaw(center + {vec2(xy)}, tile) * {vec3(rgb)};",
                          "}"])
            continue
        lines.extend([f"const int bloom_injectionCount = {len(pairs)};",
                      "shared vec2 bloom_injectionOffsets[bloom_injectionCount];",
                      "shared vec3 bloom_injectionWeights[bloom_injectionCount];",
                      "void bloom_prepareInjection() {", "    switch (gl_LocalInvocationIndex) {"])
        for lane in range(min(len(pairs),32)):
            lines.append(f"        case {lane}u:")
            for i in range(lane,len(pairs),32):
                xy,rgb = pairs[i]
                lines.extend([f"            bloom_injectionOffsets[{i}] = {vec2(xy)};",
                              f"            bloom_injectionWeights[{i}] = {vec3(rgb)};"])
            lines.append("            break;")
        lines.extend(["    }", "    barrier();", "}",
                      "vec3 bloom_injection(vec2 center, vec4 tile) {",
                      "    vec3 value = vec3(0.0);",
                      "    for (int i = 0; i < bloom_injectionCount; ++i) {",
                      "        value += bloom_sampleRaw(center + bloom_injectionOffsets[i], tile) * bloom_injectionWeights[i];",
                      "    }", "    return value;", "}"])
    lines.append("#endif")
    deepest = injection[-1]
    if deepest["offsets"] != [[0.,0.]]:
        raise ValueError("The deepest injection must be a single center tap")
    lines.extend(["#ifdef BLOOM_FIRST_UPSAMPLE",
                  f"const vec3 bloom_deepestWeight = {vec3(deepest['rgb_weights'][0])};",
                  "#endif", ""])
    return "\n".join(lines)


def builtin_seed(name):
    """Project the reference tables onto the common reconstruction for initial candidates."""
    recipe = Recipe(name)
    down,up = reference_taps()
    def neutral(taps):
        return {"offsets":taps[:,:2].tolist(),"rgb_weights":np.repeat(taps[:,2:3],3,axis=1).tolist()}
    center = {"offsets":[[0.,0.]],"rgb_weights":[[1/LEVELS]*3]}
    injection = [center]*LEVELS
    upsample = [neutral(up)]*LEVELS
    if name != "standard":
        # Keep the shape carrier near the reference while the complete response is refitted.
        gain,taps = reference_shape_tables()[name]
        merged = {(0.,0.):np.full(3,1/LEVELS)}
        for x,y,r,g,b in taps:
            key = (x,y)
            merged[key] = merged.get(key,0.) + gain*np.array([r,g,b])
        injection[0] = {"offsets":[list(xy) for xy in merged],"rgb_weights":[w.tolist() for w in merged.values()]}
        upsample[0] = neutral(np.vstack(([[0.,0.,.98]],np.column_stack((up[:,:2],.02*up[:,2])))))
    return {"topology":"filtered-injection","recipe_sha256":recipe.digest(),
            "downsample":[neutral(down)]*LEVELS,"upsample":upsample,"injection":injection}


KERNEL_INPUTS = Path(__file__).with_name("assets") / "bloom-dpf-kernels.json"


def kernel_rows(stage):
    return [[round(v,9) for v in (*xy,*rgb)]
            for xy,rgb in zip(stage["offsets"],stage["rgb_weights"],strict=True)]


def write_kernel_inputs(path, kernels):
    compact = lambda x: json.dumps(x,separators=(",",":"),allow_nan=False)
    lines = ['{', f'  "downsample":{compact(kernels["downsample"])},', '  "presets":{']
    for i,(name,preset) in enumerate(kernels["presets"].items()):
        if i:
            lines[-1] += ','
        lines.extend([f'    "{name}":{{',
                      f'      "status":{compact(preset["status"])},',
                      f'      "recipe_sha256":{compact(preset["recipe_sha256"])},'])
        for j,kind in enumerate(("upsample","injection")):
            lines.append(f'      "{kind}":[')
            rows = preset[kind]
            lines.extend('        '+compact(stage)+(',' if level+1<len(rows) else '')
                         for level,stage in enumerate(rows))
            lines.append('      ]'+(',' if j == 0 else ''))
        lines.append('    }')
    lines.extend(['  }','}',''])
    path.write_text("\n".join(lines),encoding="utf-8",newline="\n")


def select_kernel_fit(kernels, fit):
    name = fit["recipe"]["preset"]
    if name not in kernels["presets"]:
        raise ValueError("Only built-in preset fits can be selected")
    shader_coefficients(Recipe(name),fit)
    if kernel_rows(fit["downsample"][0]) != kernels["downsample"]:
        raise ValueError("The selected fit does not use the built-in downsample")
    kernels["presets"][name] = {
        "status":"fitted candidate","recipe_sha256":fit["recipe_sha256"],
        **{kind:[kernel_rows(stage) for stage in fit[kind]] for kind in ("upsample","injection")},
    }


def write_shader_kernels(output, kernels):
    """Generate the runtime tables from the explicit coefficient selection."""
    def stage(rows):
        return {"offsets":[row[:2] for row in rows],"rgb_weights":[row[2:] for row in rows]}
    presets = kernels["presets"]
    if set(presets) != set(PRESETS[:-1]):
        raise ValueError("The runtime coefficient file must contain every built-in preset")
    down = [stage(kernels["downsample"])]*LEVELS
    parts = ["// Generated by scripts/bloom_dpf_fit.py."]
    standard = None
    for index,name in enumerate(PRESETS[:-1]):
        entry = presets[name]
        fit = {"topology":"filtered-injection","recipe_sha256":entry["recipe_sha256"],
               "downsample":down,
               **{kind:[stage(rows) for rows in entry[kind]] for kind in ("upsample","injection")}}
        glsl = shader_coefficients(Recipe(name),fit)
        if index == 0:
            standard = glsl
        parts.extend([f"#{'if' if index == 0 else 'elif'} SETTING_BLOOM_SHAPE == {index} // {name}, {entry['status']}",
                      glsl])
    parts.append("#endif\n")
    output.mkdir(parents=True,exist_ok=True)
    (output/"BloomDpfPresets.glsl").write_text("\n".join(parts),encoding="utf-8",newline="\n")
    custom = output/"BloomDpfCustom.glsl"
    if not custom.exists():
        custom.write_text(standard,encoding="utf-8",newline="\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the built-in Bloom DPF shader kernels")
    parser.add_argument("--inputs", type=Path, default=KERNEL_INPUTS)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"shaders/techniques/bloom")
    parser.add_argument("--select-fit", action="append", type=Path, default=[],
                        help="Explicitly select a new fitted coefficient table for one preset")
    args = parser.parse_args()
    kernels = json.loads(args.inputs.read_text(encoding="utf-8"))
    for path in args.select_fit:
        select_kernel_fit(kernels,json.loads(path.read_text(encoding="utf-8")))
    write_shader_kernels(args.output,kernels)
    if args.select_fit:
        write_kernel_inputs(args.inputs,kernels)
