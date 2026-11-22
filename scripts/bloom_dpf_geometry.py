"""Allocate a spatial DPF from a complete linear RGB impulse response."""
import copy
import math

import numpy as np
from scipy.ndimage import gaussian_filter, grey_opening, map_coordinates
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from bloom_dpf_fit import HELD_PHASES
from bloom_dpf_joint import JointDPF, group_params, objective_model
from bloom_dpf_model import LEVELS, reference_taps
from bloom_dpf_policy import arrays, cost, decode_fit, local_batches, output, spatial_readout
from bloom_dpf_target import RasterTarget, resize_mass, target_kernel


def quantize(points, weights, count):
    """Keep unsigned RGB mass while clustering signed, unconstrained 2D locations."""
    if len(points) <= count:
        return points.copy(), weights.copy()
    importance = np.sqrt(weights.max(axis=1))
    centers = [points[np.argmax(importance)]]
    distance = np.full(len(points), np.inf)
    for _ in range(1, count):
        distance = np.minimum(distance, np.sum((points-centers[-1])**2, axis=1))
        centers.append(points[np.argmax(distance*importance)])
    centers = np.asarray(centers)
    for _ in range(16):
        groups = cKDTree(centers).query(points)[1]
        new = centers.copy()
        for i in range(count):
            hits = groups == i
            if np.any(hits):
                new[i] = np.average(points[hits], axis=0, weights=importance[hits])
        if np.max(abs(new-centers)) < 1e-4:
            break
        centers = new
    groups = cKDTree(centers).query(points)[1]
    present = np.unique(groups)
    return centers[present], np.array([weights[groups == i].sum(axis=0) for i in present])


def image_bands(pixels):
    remaining = pixels.copy()
    result = []
    radius = pixels.shape[-1]//2
    for level in range(1, LEVELS):
        step = 2**level
        broad = grey_opening(remaining, size=(1, 2*step+1, 2*step+1))
        detail = np.maximum(remaining-broad, 0.)
        sampled = resize_mass(detail, math.ceil(radius/step), 1/step)
        y,x = np.nonzero(sampled.max(axis=0) > sampled.max()*1e-6)
        center = sampled.shape[-1]//2
        result.append((np.column_stack((center-x, center-y)).astype(float), sampled[:,y,x].T))
        remaining = broad
    result.append((np.zeros((1,2)), remaining.sum(axis=(1,2), dtype=np.float64)[None]))
    return result


def ridge_directions(pixels, radius):
    """Search all orientations. Positive and negative halves retain independent weights."""
    center = pixels.shape[-1]//2
    distance = np.arange(8., min(radius, center)+1)
    angles = np.arange(0., 180., .5)
    angle = np.deg2rad(angles)
    direction = np.column_stack((np.cos(angle), np.sin(angle)))
    values = []
    for sign in (-1, 1):
        bands = []
        for cross in (-3., 0., 3.):
            xy = sign*direction[:,None,:]*distance[None,:,None]
            xy += cross*np.column_stack((-direction[:,1],direction[:,0]))[:,None,:]
            coordinates = (center+xy[...,1], center+xy[...,0])
            bands.append(np.stack([map_coordinates(c, coordinates, order=1, prefilter=False,
                                                   mode="constant", cval=0.) for c in pixels]))
        contrast = np.maximum(bands[1]-.5*(bands[0]+bands[2]), 0.)
        values.append(np.sum(np.sqrt(contrast.max(axis=0)), axis=1))
    score = sum(values)
    if score.max() <= 0:
        return []
    peaks = find_peaks(np.tile(score,3), distance=8, prominence=score.max()*.02)[0]
    peaks = peaks[(peaks >= len(score)) & (peaks < 2*len(score))]-len(score)
    peaks = sorted(peaks, key=lambda i: score[i], reverse=True)
    return [(float(angles[i]), direction[i]) for i in peaks if score[i] >= .06*score.max()][:12]


def geometry_seed(target, radius, outer, read_budget):
    pixels = target_kernel(target, outer)
    mass = pixels.sum(axis=(1,2), dtype=np.float64)
    down, up = reference_taps()
    up_offsets = [np.vstack(([[0.,0.]], up[:,:2])), *[up[:,:2]]*(LEVELS-1)]
    up_weights = [np.repeat(np.r_[.75, .25*up[:,2]][:,None], 3, axis=1),
                  *[np.repeat(up[:,2:3], 3, axis=1)]*(LEVELS-1)]
    # All geometry comes from the sampled image, including the smooth core and halo.
    bands = image_bands(pixels)
    counts = np.ones(LEVELS, dtype=int)
    fixed = sum(len(down)/4**(i+1)+len(u)/4**i for i,u in enumerate(up_offsets))
    spread = []
    for points, weights in bands:
        importance = np.sqrt(weights.max(axis=1))
        center = np.average(points, axis=0, weights=importance) if len(points) else np.zeros(2)
        spread.append(float(np.sum(importance*np.sum((points-center)**2,axis=1))))
    work = fixed+sum(4.**-np.arange(1,LEVELS+1))
    while True:
        benefits = np.array(spread)/(counts*(counts+1))
        benefits[counts >= np.minimum(192, [max(1,len(p)) for p,_ in bands])] = 0
        benefits[-1] = 0
        units = 4.**-np.arange(1,LEVELS+1)
        benefits[work+units > read_budget] = 0
        if not np.any(benefits):
            break
        level = int(np.argmax(benefits/units))
        counts[level] += 1
        work += units[level]
    stages = []
    for (points,weights),count in zip(bands,counts,strict=True):
        if not len(points):
            points, weights = np.zeros((1,2)), np.full((1,3), 1e-12)
        p,w = quantize(points, weights, int(count))
        stages.append(output(p,w))
    physical = sum(np.asarray(s["rgb_weights"]).sum(axis=0) for s in stages)
    for stage in stages:
        stage["rgb_weights"] = (np.asarray(stage["rgb_weights"])*mass/physical).tolist()
    return dict(topology="differentiable-stencil", recipe_sha256=target.digest(), levels=LEVELS,
                depth=LEVELS, fine_radius=radius, outer_radius=outer,
                downsample=[output(down[:,:2],np.repeat(down[:,2:3],3,axis=1)) for _ in range(LEVELS)],
                upsample=[output(p,w) for p,w in zip(up_offsets,up_weights,strict=True)],
                injection=stages, features=[], geometric_counts=counts.tolist())


def carrier_candidate(seed, direction, angle, pitch):
    result = copy.deepcopy(seed)
    points, weights = arrays(result["injection"][0])
    along = points @ direction
    across = abs(points @ np.array([-direction[1],direction[0]]))
    selected = (across <= 1.1) & (abs(along) >= 3.)
    if np.count_nonzero(selected) < 8:
        return None
    ids = np.floor(along[selected]/pitch+.5).astype(int)
    keys = np.unique(ids)
    long_weights = np.array([weights[selected][ids == k].sum(axis=0) for k in keys])
    # The short support fills the long lattice. Joint fitting can move every 2D tap.
    u = np.arange(1-pitch, pitch)/pitch
    short_weights = 1.-abs(u)
    short_weights /= short_weights.sum()
    short = output((u*pitch)[:,None]*direction,
                   np.repeat(short_weights[:,None],3,axis=1))
    result["features"].append(dict(angle=angle, level=1, short=short,
                                   long=output((keys*pitch)[:,None]*direction, long_weights)))
    result["injection"][0] = output(points[~selected], weights[~selected])
    result["topology"] = "factorized-stencil"
    return result


def select_level_counts(target, seed, models, read_budget, cancelled):
    bands = image_bands(target_kernel(target,seed["outer_radius"]))
    counts = np.array(seed["geometric_counts"])
    best = seed
    best_error,_ = composed_score(models,seed)
    trace = []
    # One adjacent-level exchange round bounds the search and tests the composed response.
    for level in range(LEVELS-2):
        for sign in (-1,1):
            if cancelled and cancelled():
                raise InterruptedError("Fit cancelled")
            proposed = counts.copy()
            change = max(1,round(counts[level]/8))
            proposed[level] += sign*change
            proposed[level+1] -= sign*4*change
            if any(n<1 or n>min(192,max(1,len(p)))
                   for n,(p,_) in zip(proposed,bands,strict=True)):
                continue
            candidate = copy.deepcopy(seed)
            for i in (level,level+1):
                p,w = quantize(*bands[i],int(proposed[i]))
                # Thresholding the image bands must not change channel energy during a count exchange.
                mass = np.asarray(seed["injection"][i]["rgb_weights"]).sum(axis=0)
                w *= mass/np.maximum(w.sum(axis=0),1e-30)
                candidate["injection"][i] = output(p,w)
            error,reads = composed_score(models,candidate)
            trace.append(dict(counts=proposed.tolist(),error=error,reads=reads))
            if reads<=read_budget and error<best_error:
                best,best_error = candidate,error
                best["geometric_counts"] = proposed.tolist()
    return best,trace


def composed_score(models, fit):
    state = decode_fit(fit, models[0].mass)
    p,w,ip,iw,*features = state
    values = []
    for model in models:
        image, outer = model.predict(p,w,(ip,iw),features=features[0] if features else ())
        rows = model.metrics(image,outer)
        error = [sum(row[k] for k in ("relative_l2","shape_scale_relative_l2",
                                      "core_relative_l2","pooled_outer_relative_l2")) for row in rows]
        values.append(max(error))
    return max(values), cost((*state[:4],features[0] if features else []), models[0].depth)


class ImageDetailLoss:
    """Compare signed pixel derivatives without authored ray labels or phase averaging."""
    def __init__(self, target):
        self.reference = [np.diff(target,axis=axis) for axis in (-1,-2)]
        self.weights = []
        for derivative in self.reference:
            peak = np.max(abs(derivative),axis=(1,2),keepdims=True)
            scale = np.maximum(abs(derivative), np.maximum(peak*.01,1e-12)).astype(np.float64)
            weight = np.divide(1.,np.square(scale),out=np.zeros_like(scale),where=peak>0)
            weight *= .15/max(float(np.sum(weight*derivative**2)), 1e-20)
            self.weights.append(weight)

    def apply(self, image, regions, delta=None):
        value = 0.
        for region in regions:
            for axis,reference,weight in zip((-1,-2),self.reference,self.weights,strict=True):
                diff = np.diff(image[region], axis=axis)-reference
                value += .5*np.sum(diff**2*weight,dtype=np.float64)/len(regions)
                if delta is not None:
                    gradient = diff*weight/len(regions)
                    lower = [slice(None)]*3
                    upper = lower.copy()
                    lower[axis],upper[axis] = slice(None,-1),slice(1,None)
                    delta[region][tuple(lower)] -= gradient
                    delta[region][tuple(upper)] += gradient
        return float(value)


def fit_kernel(target, *, radius=160, outer_radius=768, progress=None, cancelled=None,
               filter_class=None, iterations=48, read_budget=57.6):
    if not isinstance(target, RasterTarget):
        raise TypeError("Geometry fitting requires a RasterTarget")
    height,width = target.pixels.shape[1:]
    ox,oy = target.origin
    extent = max(ox+.5,width-.5-ox,oy+.5,height-.5-oy)*target.pixel_size
    outer_radius = max(outer_radius,math.ceil(extent-.5))
    native = target_kernel(target, outer_radius)
    residual = np.maximum(native-gaussian_filter(native, (0,4,4)), 0.).max(axis=0)
    y,x = np.nonzero(residual > residual.max()*.001)
    center = outer_radius
    detail_radius = max((np.hypot(x-center,y-center)).max(initial=0), 6.)
    radius = max(radius, math.ceil(detail_radius)+16)
    if outer_radius < radius+32:
        outer_radius = radius+32
        native = target_kernel(target,outer_radius)
    held = [tuple(p % 2**LEVELS for p in phase) for phase in HELD_PHASES]
    batches, monitor = local_batches(LEVELS, held)
    train = [batches[i] for i in (0,1,4,5)]
    selection = monitor[:1]
    assert not set(held).intersection(p for batch in train+selection for p in batch)
    assert not set(p for batch in train for p in batch).intersection(selection[0])
    make_model = lambda phases: JointDPF(target, LEVELS, radius, outer_radius, phases, filter_class=filter_class)
    models = [make_model(phases) for phases in train]
    monitors = [make_model(phases) for phases in selection]
    seed = geometry_seed(target, radius, outer_radius, read_budget)
    initial_score, initial_cost = composed_score(models, seed)
    if progress:
        progress(0,initial_score)
    layout_trace = [dict(carriers=0, error=initial_score, reads=initial_cost)]
    seed,count_trace = select_level_counts(target,seed,models,read_budget,cancelled)
    # Direct taps and image-derived carriers compete on the same composed response.
    for angle,direction in ridge_directions(native, radius):
        if cancelled and cancelled():
            raise InterruptedError("Fit cancelled")
        if len(seed["features"]) == 6:
            break
        current_error,_ = composed_score(models,seed)
        candidates = []
        for pitch in (2,3,4):
            candidate = carrier_candidate(seed, direction, angle, pitch)
            if candidate is None:
                continue
            error,reads = composed_score(models,candidate)
            layout_trace.append(dict(carriers=len(candidate["features"]), angle=angle,
                                     pitch=pitch,error=error,reads=reads))
            if reads <= read_budget and error <= current_error*1.03:
                candidates.append((error + .08*reads/read_budget,candidate))
        if candidates:
            seed = min(candidates,key=lambda pair: pair[0])[1]
    for model in models:
        model.fine_aux = ImageDetailLoss(model.target)
    state = decode_fit(seed,models[0].mass)
    state = (*state[:4],state[4] if len(state)>4 else [])
    x0,bounds,decode,objective = objective_model(models,state,position_span=.6)
    # Retain the reference analysis pyramid. Sampling geometry is fitted in the reconstruction.
    packed_positions,packed_weights = group_params(state[0],state[1],LEVELS)
    down_positions = sum(a.size for a in packed_positions[:LEVELS])
    all_positions = sum(a.size for a in packed_positions)
    down_weights = sum(a.size for a in packed_weights[:LEVELS])
    for start,end in ((0,down_positions),(all_positions,all_positions+down_weights)):
        bounds[start:end] = [(v,v) for v in x0[start:end]]
    history = []
    seed_monitor,_ = composed_score(monitors,seed)
    best = [seed_monitor,x0.copy(),0]

    def callback(intermediate_result):
        if cancelled and cancelled():
            raise InterruptedError("Fit cancelled")
        point = intermediate_result.x
        if len(history)%8 == 0:
            candidate = encode(seed,decode(point)[0],models[0].mass)
            score,_ = composed_score(monitors,candidate)
            if score < best[0]:
                best[:] = score,point.copy(),len(history)+1
        history.append(dict(iteration=len(history)+1,loss=float(intermediate_result.fun)))
        if progress:
            progress(len(history),float(intermediate_result.fun))

    def evaluate(point):
        if cancelled and cancelled():
            raise InterruptedError("Fit cancelled")
        return objective(point)

    result = minimize(evaluate,x0,jac=True,method="L-BFGS-B",bounds=bounds,callback=callback,
                      options=dict(maxiter=iterations,maxls=10,ftol=1e-6))
    final = encode(seed,decode(result.x)[0],models[0].mass)
    final_score,_ = composed_score(monitors,final)
    chosen = result.x
    chosen_iteration = result.nit
    if best[0] < final_score:
        chosen = best[1]
        chosen_iteration = best[2]
        final = encode(seed,decode(chosen)[0],models[0].mass)
    chosen_loss = float(evaluate(chosen)[0])
    response,work = spatial_readout(target,seed,final,filter_class=filter_class,cancelled=cancelled)
    readout_models = [make_model([phase]) for phase in response["phases"]]
    state = decode_fit(final,models[0].mass)
    p,w,ip,iw,*features = state
    rows = []
    for m,fine,pooled in zip(readout_models,response["fine"],response["pooled"],strict=True):
        image = np.zeros((1,3,*m.roi[1]),np.float32)
        image[m.regions[0]] = fine
        rows.extend(m.metrics(image,pooled[None]))
    final.update(recipe=target.resolved(),raster_sha256=target.digest(),
                 training_phases=train,selection_phases=selection,holdout_phases=response["phases"],
                 simulation_side=response["side"],outer_pool=response["pool"],holdout=rows,
                 iterations=result.nit,iteration_limit=iterations,converged=bool(result.success),
                 selected_iteration=chosen_iteration,
                 loss=chosen_loss,solver="Image geometry, composed multiphase spatial DPF",
                 geometry_selection=layout_trace,level_count_selection=count_trace,trace=history,
                 weighted_bilinear_fetches_per_full_res_pixel=work)
    return final,response


def encode(seed, state, mass):
    p,w,ip,iw,*extra = state
    result = dict(seed)
    result.update(downsample=[output(a,b) for a,b in zip(p[:LEVELS],w[:LEVELS],strict=True)],
                  upsample=[output(a,b) for a,b in zip(p[LEVELS:],w[LEVELS:],strict=True)],
                  injection=[output(a,b*mass) for a,b in zip(ip,iw,strict=True)])
    features = extra[0] if extra else []
    result["features"] = [dict(angle=c.angle,level=c.level,short=output(c.short,c.short_weights),
                                long=output(c.long,c.long_weights*mass)) for c in features]
    result["topology"] = "factorized-stencil" if features else "differentiable-stencil"
    return result
