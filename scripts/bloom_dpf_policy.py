"""Fit the complete authored response with a cost-bounded spatial DPF chain."""
import copy
import math
from time import monotonic

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

import bloom_dpf_joint as joint
from bloom_dpf_fit import fit_geometry
from bloom_dpf_joint import AngularRayLoss, Carrier, JointDPF, RayLoss, Stencil, Window


def arrays(stage):
    return np.asarray(stage["offsets"]),np.asarray(stage["rgb_weights"])

def output(p,w):
    return dict(offsets=p.tolist(),rgb_weights=w.tolist())

def axes(recipe,count=None):
    groups = {}
    for lobe in recipe.lobes or ():
        if not lobe.gain:
            continue
        angle = lobe.angle%180
        groups.setdefault(angle,[]).append(lobe)
    strengths = {angle:sum(lobe.gain*lobe.length/lobe.width for lobe in lobes)
                 for angle,lobes in groups.items()}
    ordered = sorted(strengths,key=strengths.get,reverse=True)
    chosen = ordered
    if count is not None and ordered:
        comparable = sum(strengths[angle] >= .5*strengths[ordered[0]] for angle in ordered)
        chosen = ordered[:min(5,max(count,comparable))]
    return [(a,np.asarray([math.cos(math.radians(a)),math.sin(math.radians(a))]),
             max(x.width for x in groups[a])*recipe.size) for a in chosen]

def assign(p,selected):
    distances = []
    for _,direction,width in selected:
        along = p @ direction
        across = np.abs(p @ np.array([-direction[1],direction[0]]))
        distances.append(np.where(abs(along)>=4.5,across/np.maximum(1.4,1.55*width),np.inf))
    distance = np.stack(distances,axis=1)
    group = np.argmin(distance,axis=1)
    return np.where(distance.min(axis=1)<=1.,group,-1)

def aggregate(p,w,groups):
    anchors,weights = [],[]
    scale = np.maximum(w.sum(axis=1),1.e-20)
    for group in np.unique(groups):
        mask = groups==group
        anchors.append(np.average(p[mask],axis=0,weights=scale[mask]))
        weights.append(w[mask].sum(axis=0))
    return np.asarray(anchors),np.asarray(weights)

def merge_direct(p,w,budget,directions=()):
    while len(p)>budget:
        if directions:
            pairs = cKDTree(p).query_pairs(r=4.0,output_type="ndarray")
            sectors = assign(p,directions)
            pairs = pairs[sectors[pairs[:,0]]==sectors[pairs[:,1]]]
        else:
            tree = cKDTree(p)
            pairs = tree.query_pairs(r=2.2,output_type="ndarray")
            if not len(pairs):
                pairs = tree.query_pairs(r=4.0,output_type="ndarray")
        if not len(pairs):
            raise ValueError("Could not meet seed budget without moving a disconnected direct tap")
        a,b = pairs.T
        mass = np.maximum(w.sum(axis=1),1.e-20)
        ab = mass[a]*mass[b]/(mass[a]+mass[b])
        if directions:
            sector_mass = np.bincount(sectors+1,weights=mass,minlength=len(directions)+1)
            ab /= np.sqrt(sector_mass[sectors[a]+1])
        spatial = np.sum((p[a]-p[b])**2,axis=1)
        color = np.sum((w[a]/mass[a,None]-w[b]/mass[b,None])**2,axis=1)
        first,last = pairs[np.argmin(ab*spatial*(1+color*3))]
        if directions:
            p[first] = np.average(p[[first,last]],axis=0,weights=mass[[first,last]])
        else:
            p[first] = (p[first]*mass[first]+p[last]*mass[last])/(mass[first]+mass[last])
        w[first] += w[last]
        p,w = np.delete(p,last,axis=0),np.delete(w,last,axis=0)
    return p,w


def ray_lengths(recipe):
    lengths = {}
    for lobe in recipe.lobes or ():
        if lobe.gain:
            angle = lobe.angle%180
            lengths[angle] = max(lengths.get(angle,0.),lobe.length)
    return lengths


def ray_lattice_seed(recipe,baseline,chosen):
    directions = axes(recipe)
    p,w = arrays(baseline["injection"][0])
    groups = assign(p,chosen)
    budget = 132
    reserve = min(np.count_nonzero(groups==-1),math.ceil(.29*budget))
    partitions = [(angle,direction,p[groups==i] @ direction,w[groups==i])
                  for i,(angle,direction,_) in enumerate(chosen)]
    if any(not len(weights) for _,_,_,weights in partitions):
        return None
    for spacing in range(3,13):
        features = []
        for angle,direction,coord,partition_weights in partitions:
            key = np.floor(coord/spacing+.5).astype(int)
            ids = np.unique(key)
            weights = np.array([partition_weights[key==k].sum(axis=0) for k in ids])
            short = (np.arange(-2,3)*(spacing/5))[:,None]*direction
            features.append(dict(angle=angle,short=output(short,np.full((5,3),.2)),
                                 long=output(ids[:,None]*spacing*direction,weights)))
        direct = budget-sum(len(f["short"]["offsets"])+len(f["long"]["offsets"]) for f in features)
        if direct<max(12,reserve):
            continue
        try:
            dp,dw = merge_direct(p[groups==-1],w[groups==-1],direct,directions)
        except ValueError:
            continue
        return dict(baseline,injection=[output(dp,dw),*baseline["injection"][1:]],
                    topology="factorized-stencil",features=features,tap_budget=budget,seed_spacing=spacing)
    return None


def interpolate_ray_carriers(recipe,lattice,fit,*,all_axes,additional):
    directions = axes(recipe)
    lengths = ray_lengths(recipe)
    features = fit["features"]
    angles = {branch["angle"] for branch in features}
    longest = sorted((angle for angle in angles if angle%90!=0),
                     key=lambda angle:(-lengths[angle],angle))[:2]
    vectors = {angle:direction for angle,direction,_ in directions}
    spacing = lattice["seed_spacing"]

    def triangle(direction,pitch,parts,paired=False):
        u = np.arange(1-parts,parts)/parts
        weights = 1.-abs(u)
        positions = u*pitch
        if paired:
            # Adjacent integer offsets on a cardinal ray share one bilinear fetch.
            start = np.arange(0,len(u),2)
            mass = np.add.reduceat(weights,start)
            positions = np.add.reduceat(positions*weights,start)/mass
            weights = mass
        return output(positions[:,None]*direction,np.repeat((weights/weights.sum())[:,None],3,axis=1))

    for reference,branch in zip(lattice["features"],features,strict=True):
        angle = branch["angle"]
        if not all_axes and angle not in longest:
            continue
        aligned = angle%90==0
        parts = spacing if aligned or angle in longest else max(3,spacing-1)
        branch["short"] = triangle(vectors[angle],spacing,parts,paired=aligned)
        branch["long"]["offsets"] = reference["long"]["offsets"]
    if additional is not None:
        p,w = arrays(fit["injection"][0])
        angle,direction,_ = directions[additional]
        hits = assign(p,directions)==additional
        coord = p[hits] @ direction
        pitch = max(3,math.ceil(np.ptp(coord)/max(1,len(coord)-3)))
        key = np.floor(coord/pitch+.5).astype(int)
        ids = np.unique(key)
        weights = np.stack([w[hits][key==k].sum(axis=0) for k in ids])
        long = output((ids*pitch)[:,None]*direction,weights)
        features.append(dict(angle=angle,short=triangle(direction,pitch,pitch),long=long))
        fit["injection"][0] = output(p[~hits],w[~hits])
    return fit

def build_carrier_seed(recipe, old, warm, budget, spacing=None, features=3):
    warm = copy.deepcopy(warm)
    assert old["recipe_sha256"]==warm["recipe_sha256"]==recipe.digest()
    p,w = arrays(old["injection"][0])
    q,v = arrays(warm["injection"][0])
    selected = axes(recipe,features)
    selected = [direction for direction in selected
                if np.any(assign(p,[direction]) == 0)]
    if not selected:
        warm.pop("features",None)
        return warm
    group = assign(p,selected)
    selected = [direction for i,direction in enumerate(selected) if np.any(group == i)]
    group,learned = assign(p,selected),assign(q,selected)
    previous = {x["angle"]:x for x in warm.get("features",())}
    if previous:
        qd,vd = q.copy(),v.copy()
        carriers = [arrays(previous[angle]["long"]) for angle,_,_ in selected]
        direct_mass = vd.sum(axis=0)
    else:
        scale = v.sum(axis=0)/w.sum(axis=0)
        qd,vd = q[learned==-1],v[learned==-1]
        direct_mass = w[group==-1].sum(axis=0)*scale
        carriers = [(p[group==i],w[group==i]*scale) for i in range(len(selected))]

    def allocate(spacing, short_limit):
        features = []
        for (angle,direction,_),(position,weight) in zip(selected,carriers,strict=True):
            along = position @ direction
            key = np.floor(along/spacing+.5).astype(int)
            long,lw = aggregate(position,weight,key)
            if short_limit:
                count = min(short_limit,max(3,math.ceil(spacing/1.12)|1))
                nodes = (np.arange(count)+.5)/count-.5
                short = (nodes*spacing)[:,None]*direction
                sw = np.full((count,3),1./count)
            else:
                short = np.asarray([-spacing/2,0.,spacing/2])[:,None]*direction
                sw = np.repeat(np.array([[.26],[.48],[.26]]),3,axis=1)
            features.append(dict(angle=angle,short=output(short,sw),long=output(long,lw)))
        space = budget-sum(len(x["short"]["offsets"])+len(x["long"]["offsets"]) for x in features)
        return features,space

    if len(selected) > 3:
        spacing = None
    if spacing is None:
        preferred = min(len(qd),max(24,round(.30*budget)))
        expanded = min(len(qd),24)
        limits = ((6.61,7,preferred),(9.51,5,expanded)) if len(selected) <= 3 else ((9.51,5,expanded),)
        for maximum,short_limit,minimum_direct in limits:
            for spacing in np.arange(1.8,maximum,.10):
                features,space = allocate(float(spacing),short_limit)
                if space >= minimum_direct:
                    break
            else:
                continue
            break
        else:
            raise ValueError("No carrier allocation fits the requested sampling budget")
    else:
        features,space = allocate(spacing,0)
    if space<12:
        raise ValueError("Too little budget for the direct center and small lobes")
    qd,vd = merge_direct(qd,vd,space)
    vd *= direct_mass/vd.sum(axis=0)
    warm.setdefault('depth',old['levels'])
    warm["topology"] = "factorized-stencil"
    warm["injection"][0] = output(qd,vd)
    warm["features"] = features
    warm["seed_spacing"] = float(spacing)
    warm["tap_budget"] = budget
    for stale in ("holdout","trace","medians","selection"):
        warm.pop(stale,None)
    return warm


def decode_fit(data,mass):
    d,dw = joint.arrays(data["downsample"])
    u,uw = joint.arrays(data["upsample"])
    p,w = joint.arrays(data["injection"])
    result = d+u,dw+uw,[np.round(a,9) for a in p],[a/mass for a in w]
    if data.get("features"):
        branches = []
        for branch in data["features"]:
            short,sw = joint.arrays([branch["short"]])
            long,lw = joint.arrays([branch["long"]])
            level = branch.get("level",1)
            if level not in (1,2) or level >= len(p):
                raise ValueError("Directional carrier level is outside the reconstruction chain")
            branches.append(Carrier(branch["angle"],level,short[0],sw[0],long[0],lw[0]/mass))
        result += (branches,)
    return result

def local_batches(depth, held):
    period = 2**depth
    stride = period//4
    used = set(held)
    rng = np.random.default_rng(651894)

    def cluster(x,y,px,py):
        anchors = [(x+dx+2*px,y+dy+2*py)
                   for dy in range(0,stride,4) for dx in range(0,stride,4)]
        for index in rng.permutation(len(anchors)):
            cx,cy = anchors[index]
            batch = [(cx+dx,cy+dy) for dy in range(2) for dx in range(2)]
            if not used.intersection(batch):
                used.update(batch)
                return batch
        raise ValueError("The fit geometry has no unused phase block")

    train = [cluster(x,y,(x//stride)%2,(y//stride)%2)
             for y in range(0,period,stride) for x in range(0,period,stride)]
    monitor = [cluster(stride*k,stride*((3*k+1)%4),k%2,k//2) for k in range(4)]
    return train,monitor


def search_probes(depth,held):
    period = 2**depth
    used = set(held)
    rng = np.random.default_rng(318897)
    cells = [(x,y) for y in range(0,period-4,4) for x in range(0,period-4,4)]
    probes = []
    for ox,oy in ((1,1),(3,1),(1,3)):
        for index in rng.permutation(len(cells)):
            cx,cy = cells[index]
            batch = [(cx+ox+dx,cy+oy+dy) for dy in range(2) for dx in range(2)]
            if not used.intersection(batch):
                probes.append(batch)
                used.update(batch)
                break
        else:
            raise ValueError("The fit geometry has no independent D2 phase block")
    return probes[0],probes[1]+probes[2]


def stage_list(state,depth):
    p,w,ip,iw,branches = state
    d,dw = joint.group_params(p,w,depth)
    edits = []
    for i,(pos,weight) in enumerate(zip(d,dw,strict=True)):
        if i<depth:
            edits.append(("down",i,np.vstack((np.zeros((1,2)),pos)),weight,4.**-(i+1),True))
        else:
            edits.append(("up",i-depth,pos,weight,4.**-(i-depth),False))
    for i,(pos,weight) in enumerate(zip(ip,iw,strict=True)):
        if i+1<depth:
            edits.append(("raw",i,pos,weight,4.**-(i+1),False))
    for i,branch in enumerate(branches):
        unit = 4.**-branch.level
        edits.extend((("short",i,branch.short,branch.short_weights,unit,False),
                      ("long",i,branch.long,branch.long_weights,unit,False)))
    return edits

def replace(state,depth,entry,position,weight):
    p,w,ip,iw,branches = [v.copy() for v in state]
    kind,index,_,_,_,_ = entry
    if kind=="down":
        p[index] = np.vstack((np.zeros((1,2)),(position[1:,None,:]*joint.DIAGONALS).reshape(-1,2)))
        w[index] = np.vstack((weight[:1],np.repeat(weight[1:]/4,4,axis=0)))
    elif kind=="up":
        p[depth+index],w[depth+index] = position,weight
    elif kind=="raw":
        ip[index],iw[index] = position,weight
    elif kind=="short":
        branches[index] = branches[index]._replace(short=position,short_weights=weight)
    else:
        branches[index] = branches[index]._replace(long=position,long_weights=weight)
    return p,w,ip,iw,branches

def cost(state,depth):
    return sum(n*unit for _,_,p,_,unit,grouped in stage_list(state,depth)
               for n in (1+4*(len(p)-1) if grouped else len(p),))

def merge(entry):
    _,_,p,w,_,grouped = entry
    if len(p)<2:
        return None
    a,b = np.triu_indices(len(p),1)
    mass = np.maximum(np.sum(w,axis=1),1.e-20)
    chroma = w/mass[:,None]
    loss = np.sum((p[a]-p[b])**2,axis=1)*mass[a]*mass[b]/(mass[a]+mass[b])
    loss *= 1+3*np.sum((chroma[a]-chroma[b])**2,axis=1)
    first,last = a[np.argmin(loss)],b[np.argmin(loss)]
    q,v = p.copy(),w.copy()
    if not grouped or first:
        q[first] = (p[first]*mass[first]+p[last]*mass[last])/(mass[first]+mass[last])
    v[first] += w[last]
    return np.delete(q,last,axis=0),np.delete(v,last,axis=0)

def reduce_taps(stage,count):
    kind,index,positions,weights,unit,grouped = stage
    while len(positions)>count:
        positions,weights = merge((kind,index,positions,weights,unit,grouped))
    return positions,weights


def split(entry,index):
    kind,_,p,w,_,grouped = entry
    q,v = p.copy(),w.copy()
    shift = np.asarray([.35,.0])
    if grouped and index==0:
        q = np.vstack((q,[[.74,.74]]))
        v = np.vstack((v,v[0]*.12))
        v[0] *= .88
    else:
        if kind=="long" and np.linalg.norm(p[index])>1.e-4:
            shift = .45*p[index]/np.linalg.norm(p[index])
        q = np.vstack((q,p[index]+shift))
        q[index] = p[index]-shift
        v[index] *= .5
        v = np.vstack((v,v[index]))
    return q,v

def motion_error(model,image):
    responses = [image[region] for region in model.regions]
    target = model.target
    energy = residual = 0.
    for i,(x,y) in enumerate(model.phases):
        for j,(tx,ty) in enumerate(model.phases[i+1:],i+1):
            dx,dy = tx-x,ty-y
            if abs(dx)+abs(dy)!=1:
                continue
            reference = np.roll(target,(dy,dx),axis=(-2,-1))-target
            observed = np.roll(responses[j],(dy,dx),axis=(-2,-1))-responses[i]
            residual += np.sum((observed-reference)[:,1:-1,1:-1]**2,dtype=np.float64)
            energy += np.sum(reference[:,1:-1,1:-1]**2,dtype=np.float64)
    return float(np.sqrt(residual/max(energy,1.e-20)))


def score_data(model,ray,state):
    p,w,ip,iw,features = state
    image,outer = model.predict(p,w,(ip,iw),features=features)
    metrics = np.asarray([[e[k] for k in joint.READOUT] for e in model.metrics(image,outer)])
    return (metrics,ray.apply(image,model.regions),motion_error(model,image),
            ray.intensities(image,model.regions))


def ray_deficit(intensity,reference):
    if not len(reference):
        return 0.
    floor = np.minimum(reference,1.)
    deficit = np.maximum(floor-intensity,0.)/np.maximum(floor,.1)
    return float(np.percentile(deficit,90))


def compare(model,ray,reference,state,preserve=None):
    a,b,motion,intensity = score_data(model,ray,state)
    ratios = a/np.maximum(reference[0],1.e-8)
    score = float(np.median(ratios,axis=0) @ [.25,.25,.28,.22]+.10*np.percentile(ratios,90)
                  +.10*b/max(reference[1],1.e-7)+.08*motion/max(reference[2],1.e-7))
    return score if preserve is None else score+.25*ray_deficit(intensity,preserve)


def carrier_layout_score(recipe,baseline,filter_class):
    depth = baseline["depth"]
    radius,outer,_ = fit_geometry(recipe,baseline["fine_radius"],baseline["outer_radius"])
    _,held = joint.phase_sets(depth)
    _,phases = search_probes(depth,held)
    model = JointDPF(recipe,depth,radius,outer,phases,filter_class=filter_class)
    ray = AngularRayLoss(recipe,radius,model.target)

    def measure_fit(fit):
        p,w,ip,iw,*extra = decode_fit(fit,model.mass)
        branches = extra[0] if extra else ()
        image,far = model.predict(p,w,(ip,iw),features=branches)
        metrics = np.asarray([[row[k] for k in joint.READOUT] for row in model.metrics(image,far)])
        return (metrics,ray.apply(image,model.regions),motion_error(model,image),
                ray.roughness(image,model.regions),ray.intensities(image,model.regions),
                cost((p,w,ip,iw,branches),depth))

    error,ray_error,motion,roughness,intensity,_ = measure_fit(baseline)
    visible = intensity>.1
    # The weighted-read ceiling screens layouts before runtime measurement.
    max_reads = 58.

    def score(fit):
        e,r,m,h,i,work = measure_fit(fit)
        ratio = e/np.maximum(error,1.e-8)
        target = np.median(ratio,axis=0) @ [.25,.25,.28,.22] + .10*np.percentile(ratio,90)
        missing = np.maximum(intensity[visible]-i[visible],0.)/intensity[visible]
        coverage = float(np.mean(missing)) if missing.size else 0.
        quality = (target+.10*r/max(ray_error,1.e-6)+.08*m/max(motion,1.e-6)
                   +.40*h/max(roughness,.03)+.70*coverage)
        return dict(score=float(quality+.03*work/max_reads),target=float(target),roughness=h,
                    ray_loss=r,motion=m,coverage_deficit=coverage,reads=work,eligible=bool(work<=max_reads))
    return score


def select_ray_lattice(recipe,baseline,score,cancelled):
    directions = axes(recipe)
    lengths = ray_lengths(recipe)
    by_length = sorted(directions,key=lambda item:lengths[item[0]],reverse=True)
    seen,layouts = set(),[]
    for chosen in (by_length[:5],directions[:5],directions[:3]):
        key = tuple(sorted(angle for angle,_,_ in chosen))
        if not key or key in seen:
            continue
        seen.add(key)
        if cancelled and cancelled():
            raise InterruptedError("Fit cancelled")
        seed = ray_lattice_seed(recipe,baseline,chosen)
        if seed is not None:
            detail = score(seed)
            if detail["eligible"]:
                layouts.append((detail["score"],seed,detail))
    try:
        compact = build_carrier_seed(recipe,baseline,baseline,132)
    except ValueError:
        compact = None
    if compact is not None and compact.get("features"):
        detail = score(compact)
        if detail["eligible"]:
            layouts.append((detail["score"],compact,detail))
    if not layouts:
        return None,[]
    _,selected,_ = min(layouts,key=lambda item:item[0])
    history = [dict(angles=[f["angle"] for f in seed["features"]],spacing=seed["seed_spacing"],
                    interpolation=seed is not compact,selected=seed is selected,**detail)
               for _,seed,detail in layouts]
    return (None if selected is compact else selected),history


def select_ray_interpolation(recipe,lattice,trained,score,cancelled):
    choices = [("trained",trained)]
    directions = axes(recipe)
    p,_ = arrays(trained["injection"][0])
    groups = assign(p,directions)
    used = {b["angle"] for b in trained["features"]}
    additions = [i for i,(angle,_,_) in enumerate(directions)
                 if angle not in used and np.count_nonzero(groups==i)>=5]
    for all_axes in (False,True):
        for extra in (None,*additions):
            label = "all axes" if all_axes else "long oblique axes"
            if extra is not None:
                label += f", plus {directions[extra][0]:g} degrees"
            candidate = interpolate_ray_carriers(recipe,lattice,copy.deepcopy(trained),
                                                  all_axes=all_axes,additional=extra)
            choices.append((label,candidate))
    grades = []
    for name,fit in choices:
        if cancelled and cancelled():
            raise InterruptedError("Fit cancelled")
        grades.append((name,fit,score(fit)))
    _,selected,detail = min((entry for entry in grades if entry[2]["eligible"]),
                            key=lambda entry:entry[2]["score"])
    selected["coefficient_training_step"] = selected.pop("best_step")
    selected["selection"] = detail["score"]
    selected["carrier_layout_selection"] = [dict(layout=label,selected=fit is selected,**row)
                                            for label,fit,row in grades]
    return selected


def carrier_definition(branch,mass):
    data = dict(angle=branch.angle,short=output(branch.short,branch.short_weights),
                long=output(branch.long,branch.long_weights*mass))
    if branch.level != 1:
        data["level"] = branch.level
    return data


def encode(data,state,depth,mass,trace):
    p,w,ip,iw,features = state
    stage = lambda a,b:dict(offsets=a.tolist(),rgb_weights=b.tolist())
    data["downsample"] = [stage(a,b) for a,b in zip(p[:depth],w[:depth],strict=True)]
    data["upsample"] = [stage(a,b) for a,b in zip(p[depth:],w[depth:],strict=True)]
    data["injection"] = [stage(a,b*mass) for a,b in zip(ip,iw,strict=True)]
    if features:
        data["features"] = [carrier_definition(branch,mass) for branch in features]
        data["topology"] = "factorized-stencil"
    else:
        data.pop("features",None)
        data["topology"] = "differentiable-stencil"
    for field in ("holdout","medians","trace","selection","ray_loss","loss",
                  "weighted_bilinear_fetches_per_full_res_pixel"):
        data.pop(field,None)
    data["structure_search"] = trace
    return data

def carrier_taps(branch):
    positions = branch.long[:,None,:]+branch.short[None,:,:]
    weights = branch.long_weights[:,None,:]*branch.short_weights[None,:,:]
    return positions.reshape(-1,2),weights.reshape(-1,3)


def structure_moves(state,recipe,depth):
    p,w,ip,iw,branches = state
    for i,branch in enumerate(branches):
        other = branches[:i]+branches[i+1:]
        level = branch.level
        layer = level-1
        if depth>2:
            factor = 2.**(level-(3-level))
            migrated = branch._replace(level=3-level,short=branch.short*factor,long=branch.long*factor)
            yield ("level",i,3-level),(p,w,ip,iw,other+[migrated])
            if level==1 and len(branches)<5 and len(branch.long)>=6:
                distance = np.linalg.norm(branch.long,axis=1)
                energy = branch.long_weights.sum(axis=1)
                masks = (("outer",distance>np.median(distance)),
                         ("low-energy",energy<np.median(energy)))
                for portion,coarse in masks:
                    if 2 <= coarse.sum() <= len(distance)-2:
                        fine = branch._replace(long=branch.long[~coarse],long_weights=branch.long_weights[~coarse])
                        far = migrated._replace(long=migrated.long[coarse],long_weights=branch.long_weights[coarse])
                        yield ("split-level",i,portion),(p,w,ip,iw,other+[fine,far])
        a,b = carrier_taps(branch)
        raw = (np.vstack((ip[layer],a)),np.vstack((iw[layer],b)))
        limit = len(ip[layer])+len(branch.long)+len(branch.short)-1
        try:
            raw = merge_direct(*raw,limit)
        except ValueError:
            pass
        else:
            candidate = (p,w,ip.copy(),iw.copy(),other)
            candidate[2][layer],candidate[3][layer] = raw
            yield ("remove",i),candidate
        for j,second in enumerate(branches[i+1:],i+1):
            difference = abs(branch.angle-second.angle)%180
            if level!=second.level or min(difference,180-difference)>45:
                continue
            first,second = sorted((branch,second),key=lambda c:c.long_weights.sum(),reverse=True)
            def center(carrier):
                weights = carrier.short_weights @ carrier.long_weights.sum(axis=0)
                return np.average(carrier.short,axis=0,weights=weights)
            long = np.vstack((first.long,second.long+center(second)-center(first)))
            weights = np.vstack((first.long_weights,second.long_weights))
            merged = first._replace(long=long,long_weights=weights)
            yield ("merge-direction",i,j),(p,w,ip,iw,[c for k,c in enumerate(branches) if k not in (i,j)]+[merged])

    if len(branches)<5:
        represented = {c.angle for c in branches}
        for angle,direction,width in axes(recipe):
            if angle in represented:
                continue
            hits = assign(ip[0],[(angle,direction,width)])==0
            if hits.sum()<5 or np.count_nonzero(~hits)<8:
                continue
            a,b = ip[0][hits],iw[0][hits]
            long,weights = aggregate(a,b,np.round(a @ direction/3).astype(int))
            short = np.array([-.8,0.,.8])[:,None]*direction
            short_weights = np.broadcast_to(np.array([[.24],[.52],[.24]]),(3,3))
            carrier = Carrier(angle,1,short,short_weights,long,weights)
            candidate = (p,w,ip.copy(),iw.copy(),branches+[carrier])
            candidate[2][0],candidate[3][0] = ip[0][~hits],iw[0][~hits]
            yield ("add-direction",angle),candidate


def tap_scores(stage,gradient,depth):
    kind,index,_,weights,_,_ = stage
    _,dp,dw,dip,diw,*extra = gradient
    if kind=="down":
        a,b = joint.group_gradient(dp[index],dw[index])
        a = np.vstack((np.zeros((1,2)),a))
    elif kind=="up":
        a,b = dp[depth+index],dw[depth+index]
    elif kind=="raw":
        a,b = dip[index],diw[index]
    else:
        feature = extra[0][index]
        a,b = feature[:2] if kind=="short" else feature[2:]
    rgb = np.sum(weights,axis=0)
    mean = np.sum(weights*b,axis=0)/np.maximum(rgb,1.e-12)
    return np.linalg.norm(a,axis=1)+np.linalg.norm(weights*(b-mean),axis=1)


def adam_step(x,gradient,mean,var,step,rate,limits):
    mean = .9*mean+.1*gradient
    var = .999*var+.001*gradient*gradient
    update = rate*(mean/(1-.9**step))/(np.sqrt(var/(1-.999**step))+1e-8)
    return np.clip(x-update,limits[:,0],limits[:,1]),mean,var


def refit_structure(model,candidate,cancelled=None):
    x,bounds,decode,evaluate = joint.objective_model(model,candidate,position_span=.35)
    limits = np.asarray(bounds)
    limits[:,0] = np.maximum(limits[:,0],x-.55)
    limits[:,1] = np.minimum(limits[:,1],x+.55)
    mean,var = np.zeros_like(x),np.zeros_like(x)
    best,best_loss = x.copy(),np.inf
    for step in range(1,9):
        if cancelled and cancelled():
            raise InterruptedError("Fit cancelled")
        value,gradient = evaluate(x)
        if not (np.isfinite(value) and np.isfinite(gradient).all()):
            raise FloatingPointError("Nonfinite stencil evaluation")
        if value<best_loss:
            best,best_loss = x.copy(),value
        x,mean,var = adam_step(x,gradient,mean,var,step,.012,limits)
    value,_ = evaluate(x)
    fitted = decode(x if value<best_loss else best)[0]
    return fitted if len(fitted)==5 else (*fitted,[])


def select_structure(recipe, old, seed, *, rounds=4, filter_class=None, cancelled=None):
    data = copy.deepcopy(seed)
    assert recipe.digest()==old["recipe_sha256"]==data["recipe_sha256"]
    depth = data["depth"]
    radius,outer,_ = fit_geometry(recipe,old["fine_radius"],old["outer_radius"])
    _,held = joint.phase_sets(depth)
    phase_train,phase_selection = search_probes(depth,held)
    train = joint.JointDPF(recipe,depth,radius,outer,phase_train,filter_class=filter_class)
    select = joint.JointDPF(recipe,depth,radius,outer,phase_selection,filter_class=filter_class)
    ray = RayLoss(recipe,radius,train.target)
    state = decode_fit(data,train.mass)
    if len(state)==4:
        state += ([],)
    original = decode_fit(old,train.mass)+((),)
    train_ref,select_ref = score_data(train,ray,original),score_data(select,ray,original)
    preserve = score_data(select,ray,state)[3]
    train.set_quality_floor(*train.predict(original[0],original[1],original[2:4]))
    train.fine_aux = ray
    initial_work = cost(state,depth)

    def work(candidate):
        return cost(candidate,depth)+sum(4.**-c.level for c in candidate[4])

    def rank(error,candidate):
        return error+.22*work(candidate)/initial_work

    trace = []
    start = monotonic()
    for round_id in range(rounds):
        if cancelled and cancelled():
            raise InterruptedError("Fit cancelled")
        before = compare(select,ray,select_ref,state,preserve)
        gradient = train.predict(state[0],state[1],state[2:4],gradient=True,features=state[4])
        stages = stage_list(state,depth)
        pruning,expanding = [],[]
        for stage in stages:
            merged = merge(stage)
            if merged is not None:
                pruned = replace(state,depth,stage,*merged)
                score = rank(compare(train,ray,train_ref,pruned),pruned)
                pruning.append((score,stage[:2],pruned))
            priority = float(np.max(tap_scores(stage,gradient,depth)))
            expanding.append((priority,stage))
        pruning.sort(key=lambda c:c[0])
        expanding.sort(key=lambda c:c[0],reverse=True)
        candidates = [(("merge-tap",*edit),candidate) for _,edit,candidate in pruning[:2]]
        if pruning:
            _,drop,pruned = pruning[0]
            stages_after = {stage[:2]:stage for stage in stage_list(pruned,depth)}
            for _,stage in expanding[:4]:
                kind = stage[:2]
                if kind==drop:
                    continue
                target = stages_after[kind]
                index = int(np.argmax(tap_scores(stage,gradient,depth)))
                moved = replace(pruned,depth,target,*split(target,index))
                if cost(moved,depth)<=initial_work:
                    candidates.append((("redistribute",drop,kind),moved))
        budget_cuts = []
        for stage in stages:
            kind,index,positions,weights,unit,_ = stage
            if kind not in ("raw","long") or unit<.25 or len(positions)<16:
                continue
            removed = min(8,len(positions)//4)
            taps = reduce_taps(stage,len(positions)-removed)
            cut = replace(state,depth,stage,*taps)
            value = rank(compare(train,ray,train_ref,cut),cut)
            budget_cuts.append((value,("prune-block",kind,index,removed),cut))
        candidates.extend((edit,candidate) for _,edit,candidate in sorted(budget_cuts,key=lambda x:x[0])[:2])
        proposals = {}
        for edit,candidate in structure_moves(state,recipe,depth):
            excess = cost(candidate,depth)-initial_work
            if excess>1.e-9:
                if edit[0]!="add-direction":
                    continue
                stage = stage_list(candidate,depth)[2*depth]
                count = len(stage[2])-math.ceil(excess/stage[4])
                if count<8:
                    continue
                candidate = replace(candidate,depth,stage,*reduce_taps(stage,count))
            score = rank(compare(train,ray,train_ref,candidate),candidate)
            proposals.setdefault(edit[0],[]).append((score,edit,candidate))
        for family,options in proposals.items():
            choices = sorted(options,key=lambda x:x[0])[:2 if family=="level" else 1]
            candidates.extend((edit,candidate) for _,edit,candidate in choices)

        winner,best = state,rank(before,state)
        selected_edit = None
        for edit,candidate in [(("refit",),state),*candidates]:
            unrefined = before if candidate is state else compare(select,ray,select_ref,candidate,preserve)
            refined = refit_structure(train,candidate,cancelled)
            error = compare(select,ray,select_ref,refined,preserve)
            improved = error < unrefined
            selected,error = (refined,error) if improved else (candidate,unrefined)
            value = rank(error,selected)
            trace.append(dict(round=round_id,edit=edit,selection=error,work=work(selected),
                              before=before,unrefined=unrefined,selected_refit=improved,
                              score=value,elapsed=monotonic()-start))
            if value < best-.00025:
                winner,best,selected_edit = selected,value,edit
        trace.append(dict(round=round_id,winner=selected_edit,score=best,work=work(winner)))
        state = winner
        if selected_edit in (None,("refit",)):
            break
    return encode(data,state,depth,train.mass,trace)


def prune_fine_budget(recipe, baseline_fit, selected, *, filter_class=None, cancelled=None):
    depth = selected["depth"]
    _, held = joint.phase_sets(depth)
    _, phases = search_probes(depth, held)
    radius,outer,_ = fit_geometry(recipe,baseline_fit["fine_radius"],baseline_fit["outer_radius"])
    model = JointDPF(recipe,depth,radius,outer,phases,
                     filter_class=filter_class)
    ray = RayLoss(recipe,model.radius,model.target)
    state = decode_fit(selected,model.mass)
    if len(state) == 4:
        state += ([],)
    baseline,ray_base,motion_base,signal = score_data(model,ray,state)
    trace = selected["structure_search"].copy()

    for round_id in range(5):
        best = None
        attempted = set()
        stages = stage_list(state,depth)
        # Keep cumulative error against the input to this pass. When a block is
        # too expensive in quality, give smaller edits a chance at that stage.
        for limit in (6,3,2,1):
            seen = set()
            for stage in stages:
                kind,index,positions,weights,unit,grouped = stage
                if kind not in ("raw","long") or unit != .25 or len(positions) < 9:
                    continue
                removed = min(limit,len(positions)//3)
                key = kind,index,removed
                if key in attempted:
                    continue
                seen.add(key)
                masks = [np.zeros(len(positions),dtype=bool)]
                if kind == "raw":
                    mass = weights.sum(axis=1)
                    masks.append(mass >= np.quantile(mass,.68))
                for keep in masks:
                    if cancelled and cancelled():
                        raise InterruptedError("Fit cancelled")
                    count = np.count_nonzero(~keep)-removed
                    if count < 1:
                        continue
                    eligible = (kind,index,positions[~keep],weights[~keep],unit,grouped)
                    offsets,rgb = reduce_taps(eligible,count)
                    candidate = replace(state,depth,stage,np.vstack((positions[keep],offsets)),
                                        np.vstack((weights[keep],rgb)))
                    measured,shape,motion,power = score_data(model,ray,candidate)
                    ratios = measured/np.maximum(baseline,1.e-8)
                    deficit = ray_deficit(power,signal)
                    ray_ratio = shape/max(ray_base,1.e-8) if ray_base else 1.
                    motion_ratio = motion/max(motion_base,1.e-8) if motion_base else 1.
                    worst = np.max(ratios,axis=0)
                    if (np.any(worst > 1.05) or deficit > .075 or ray_ratio > 1.15 or motion_ratio > 1.05):
                        continue
                    score = (np.median(ratios,axis=0) @ [.2,.18,.4,.22]+.15*np.percentile(ratios,90)
                             +.12*ray_ratio+.09*motion_ratio+.35*deficit)
                    row = dict(budget_round=round_id,edit=["prune-block",kind,index,removed],
                               protected=int(keep.sum()),score=float(score),reads=cost(candidate,depth),
                               worst=worst.tolist(),ray_ratio=ray_ratio,ray_deficit=deficit)
                    if best is None or score < best[0]:
                        best = (score,candidate,row)
            attempted.update(seen)
            if best is not None:
                break

        if best is None:
            break
        _,state,row = best
        trace.append(row)
    if len(trace) == len(selected["structure_search"]):
        return selected
    return encode(selected.copy(),state,depth,model.mass,trace)


def carrier_arrays(data):
    d = data["injection"][0]
    p = np.asarray(d["offsets"])
    w = np.asarray(d["rgb_weights"])
    branches = []
    for f in data.get("features",()):
        short,long = f["short"],f["long"]
        branches.append(tuple(np.asarray(x) for x in (short["offsets"],short["rgb_weights"],
                                                      long["offsets"],long["rgb_weights"])))
    return p,w,branches

def setup(side):
    center = side//2
    point = Window((center,center),(1,1))
    image = Window((0,0),(side,side))
    impulse = np.ones((1,1,1,1),np.float32)
    def render(p,w,branches,adjoint=None):
        direct = Stencil(point,image,p)
        output = direct.apply(impulse,w)
        operators = []
        for short,sw,long,lw in branches:
            lo = np.floor(center-np.max(short[:,::-1],axis=0)-1).astype(int)
            hi = np.ceil(center-np.min(short[:,::-1],axis=0)+1).astype(int)
            win = Window(tuple(lo),tuple(hi-lo+1))
            first = Stencil(point,win,short)
            carrier = first.apply(impulse,sw)
            second = Stencil(win,image,long)
            second.apply(carrier,lw,output)
            operators.append((first,second,carrier,sw,lw))
        if adjoint is None:
            return output
        loss,delta = adjoint(output)
        _,dp,dw = direct.backward(impulse,delta,w)
        gradients = []
        for first,second,carrier,sw,lw in operators:
            dr,dl,dlw = second.backward(carrier,delta,lw)
            _,ds,dsw = first.backward(impulse,dr,sw)
            gradients.append((ds,dsw,dl,dlw))
        return loss,dp,dw,gradients
    return render



def refine_carrier(first, seed, *, iterations=160, cancelled=None):
    data = copy.deepcopy(seed)
    assert first["recipe_sha256"]==data["recipe_sha256"]
    pos,w,features = carrier_arrays(data)
    sp,sw,lp,lw = map(list,zip(*features,strict=True))
    total = np.concatenate([w,*lw]).sum(axis=0)
    assert np.all(total>0)
    z = np.concatenate([w,*lw])/total
    count = len(features)
    split_taps = np.cumsum([len(a) for a in [w,*lw]])[:-1]
    variables = [pos,*sp,*lp,*[np.log(np.maximum(a,1e-9)) for a in sw],np.log(np.maximum(z,1e-9))]
    edges = np.cumsum([a.size for a in variables])[:-1]
    x = np.concatenate([a.ravel() for a in variables])
    bounds = []
    for i,a in enumerate(variables):
        radius = 1.45 if i<1+2*count else 4.
        bounds.extend((q-radius,q+radius) for q in a.ravel())
    max_offset = max(float(np.max(abs(q))) for q in [pos,*lp,*sp])
    side = 2*max(128,int(np.ceil(max_offset+16)))+1
    render = setup(side)
    target = render(*carrier_arrays(first))
    peak = float(target.max())
    strength = 1./max(float(np.sum(target**2,dtype=np.float64)),1.e-20)
    weak = 1./np.maximum(target,peak*.015)**1.2
    weak *= .6/max(float(np.sum(target**2*weak,dtype=np.float64)),1.e-20)
    weights = strength+weak

    def read(params):
        q = [b.reshape(a.shape) for b,a in zip(np.split(params,edges),variables,strict=True)]
        p = q[0]
        short,long = q[1:1+count],q[1+count:1+2*count]
        sw = [JointDPF.normalize(a) for a in q[1+2*count:-1]]
        probs = JointDPF.normalize(q[-1])
        levels = np.split(probs,split_taps)
        return p,levels[0]*total,list(zip(short,sw,long,[a*total for a in levels[1:]],strict=True)),probs

    def objective(params):
        if cancelled and cancelled():
            raise InterruptedError('Fit cancelled')
        p,w,branch,probs = read(params)
        def adjoint(output):
            error = output-target
            delta = error*weights
            return .5*np.sum(error*delta,dtype=np.float64),delta
        f,dp,dw,g = render(p,w,branch,adjoint)
        dshort,dsw,dlong,dlw = map(list,zip(*g,strict=True))
        dsw = [b*(a-np.sum(a*b,axis=0)) for (_,b,_,_),a in zip(branch,dsw,strict=True)]
        combined = np.concatenate([dw,*dlw])*total
        dz = probs*(combined-np.sum(probs*combined,axis=0))
        return f,np.concatenate([a.ravel() for a in (dp,*dshort,*dlong,*dsw,dz)])

    initial = float(objective(x)[0])
    start = monotonic()
    result = minimize(objective,x,jac=True,method="L-BFGS-B",bounds=bounds,
                      options=dict(maxiter=iterations,maxls=12,ftol=1.e-9))
    p,w,branches,_ = read(result.x)
    stage = lambda p,w:dict(offsets=p.tolist(),rgb_weights=w.tolist())
    data["injection"][0] = stage(p,w)
    data["features"] = [dict(angle=old["angle"],short=stage(sp,sw),long=stage(lp,lw))
                        for old,(sp,sw,lp,lw) in zip(data["features"],branches,strict=True)]
    data["carrier_seed_refit"] = dict(before=initial,after=float(result.fun),steps=int(result.nit))
    return data


def train_chain(recipe, old, seed, *, steps, interval=64, span=1.2, ray_strength=.08,
                ray_loss=RayLoss, filter_class=None, progress=None, cancelled=None):
    assert old["recipe_sha256"]==seed["recipe_sha256"]==recipe.digest()
    depth = old["levels"]
    radius,outer,_ = fit_geometry(recipe,old["fine_radius"],old["outer_radius"])
    _,held = joint.phase_sets(depth)
    train,monitor = local_batches(depth,held)

    def model(phases):
        return joint.JointDPF(recipe,depth,radius,outer,phases,filter_class=filter_class)

    trains = [model(batch) for batch in train]
    monitors = [model(batch) for batch in monitor]
    ray = ray_loss(recipe,radius,trains[0].target,ray_strength)
    original = decode_fit(old,trains[0].mass)
    start = decode_fit(seed,trains[0].mass)
    def predict(m,coefficients):
        p,w,ip,iw,*extra = coefficients
        return m.predict(p,w,(ip,iw),features=extra[0] if extra else ())
    for m in trains:
        m.set_quality_floor(*predict(m,original))
        m.fine_aux = ray

    def assess(models,coefficients):
        rows = []
        ray_values = []
        motion = []
        signals = []
        for m in models:
            image,outer = predict(m,coefficients)
            rows.extend(m.metrics(image,outer))
            ray_values.append(ray.apply(image,m.regions))
            motion.append(motion_error(m,image))
            signals.append(ray.intensities(image,m.regions))
        metrics = np.asarray([[row[k] for k in joint.READOUT] for row in rows])
        return rows,metrics,float(np.mean(ray_values)),float(np.mean(motion)),np.mean(signals,axis=0)

    _,legacy_metrics,legacy_rays,legacy_motion,_ = assess(monitors,original)
    _,_,_,_,preserve = assess(monitors,start)
    def selection(coefficients):
        _,metrics,rays,motion,intensity = assess(monitors,coefficients)
        ratio = metrics/np.maximum(legacy_metrics,1.e-8)
        medians = np.median(ratio,axis=0)
        score = (medians @ [.25,.25,.28,.22] + .10*np.percentile(ratio,90)
                 + .10*rays/max(legacy_rays,1.e-6) + .08*motion/max(legacy_motion,1.e-6)
                 + .25*ray_deficit(intensity,preserve))
        return float(score),dict(relative_median=medians.tolist(),ray_loss=rays,motion_error=motion,
                                 median=np.median(metrics,axis=0).tolist())

    x,limits,decode,objective = joint.objective_model(trains,start,position_span=span)
    limits = np.asarray(limits)
    packed_pos,packed_weight = joint.group_params(start[0],start[1],depth)
    ds,us = sum(a.size for a in packed_pos[:depth]),sum(a.size for a in packed_pos[depth:])
    weights = sum(a.size for a in packed_weight)
    positions = sum(a.size for a in start[2])
    sizes = (ds,us,weights,positions,x.size-ds-us-weights-positions)
    lr = np.concatenate([np.full(n,r) for n,r in zip(sizes,(.010,.018,.022,.030,.030),strict=True)])
    rng = np.random.default_rng(57720)
    elapsed = monotonic()
    initial,detail = selection(decode(x)[0])

    best,best_score,best_step = x.copy(),initial,0
    mean,var = np.zeros_like(x),np.zeros_like(x)
    rates,plateau,trace = 1.,0,[]
    for step in range(1,steps+1):
        if cancelled and cancelled():
            raise InterruptedError('Fit cancelled')
        index = (step-1)%len(trains)
        if not index:
            order = rng.permutation(len(trains))
        value,g = objective(x,[trains[order[index]]])
        if not (np.isfinite(value) and np.isfinite(g).all()):
            raise FloatingPointError("Nonfinite stencil evaluation")
        x,mean,var = adam_step(x,g,mean,var,step,rates*lr,limits)
        if step%interval and step!=steps:
            continue
        score,detail = selection(decode(x)[0])
        item = dict(step=step,train=float(value),selection=score,elapsed=monotonic()-elapsed,rate=rates,**detail)
        if progress:
            progress(step,float(value))
        trace.append(item)
        if score<best_score*.999:
            best,best_score,best_step = x.copy(),score,step
            plateau = 0
        else:
            plateau += 1
            if plateau in (2,4):
                rates *= .5
                x = best.copy()
                mean.fill(0.)
                var.fill(0.)
        if plateau>=6:
            break

    coefficients = decode(best)[0]
    validation = [model(held[i:i+4]) for i in range(0,len(held),4)]
    rows,metrics,rays,_,_ = assess(validation,coefficients)
    p,w,ip,iw,*extra = coefficients
    encode = lambda offsets,weights:[dict(offsets=a.tolist(),rgb_weights=b.tolist()) for a,b in zip(offsets,weights,strict=True)]
    result = dict(topology="differentiable-stencil",depth=depth,recipe_sha256=recipe.digest(),
        downsample=encode(p[:depth],w[:depth]),upsample=encode(p[depth:],w[depth:]),
        injection=encode(ip,[a*trains[0].mass for a in iw]),seed="automatic",
        train=train,monitor=monitor,holdout=rows,ray_loss=rays,
        steps=step,best_step=best_step,initial=initial,selection=best_score,trace=trace,
        medians=np.median(metrics,axis=0).tolist())
    if extra:
        result["topology"] = "factorized-stencil"
        result["features"] = [carrier_definition(branch,trains[0].mass) for branch in extra[0]]
    if "structure_search" in seed:
        result["structure_search"] = seed["structure_search"]
    return result



def spatial_readout(recipe, baseline, selected, *, filter_class=None, cancelled=None, ray_loss=None):
    from bloom_dpf_fit import HELD_PHASES

    depth = baseline["levels"]
    radius, outer, _ = fit_geometry(recipe,baseline["fine_radius"],baseline["outer_radius"])
    models = [JointDPF(recipe,depth,radius,outer,[phase],filter_class=filter_class)
              for phase in HELD_PHASES]
    state = decode_fit(selected,models[0].mass)
    p,w,ip,iw,*extra = state
    features = extra[0] if extra else ()
    fine, pooled, rows, ray_values = [], [], [], []
    ray = ray_loss(recipe,radius,models[0].target) if ray_loss is not None else None
    for model in models:
        if cancelled and cancelled():
            raise InterruptedError("Fit cancelled")
        image, far = model.predict(p,w,(ip,iw),features=features)
        fine.append(image[model.regions[0]])
        pooled.append(far[0])
        if ray is not None:
            rows.extend(model.metrics(image,far))
            ray_values.append(ray.apply(image,model.regions))
    if ray is not None:
        selected.update(holdout=rows,ray_loss=float(np.mean(ray_values)),
                        medians=np.median([[row[k] for k in joint.READOUT] for row in rows],axis=0).tolist())
    phases = [model.phases[0] for model in models]
    return (dict(fine=np.stack(fine),pooled=np.stack(pooled),phases=phases,
                 side=models[0].side,pool=models[0].pool),
            cost((p,w,ip,iw,features),depth))


def fit_spatial_recipe(recipe, *, radius, outer_radius, progress=None, cancelled=None, filter_class=None):
    from bloom_dpf_geometry import fit_kernel
    from bloom_dpf_target import RasterTarget, target_kernel

    if radius is None or outer_radius is None:
        radius,outer_radius,_ = fit_geometry(recipe,radius,outer_radius)
    target = recipe if isinstance(recipe,RasterTarget) else RasterTarget(target_kernel(recipe,outer_radius))
    fitted, images = fit_kernel(target,radius=radius,outer_radius=outer_radius,
        progress=progress,cancelled=cancelled,filter_class=filter_class)
    fitted.update(recipe=recipe.resolved(),recipe_sha256=recipe.digest())
    return fitted,images


def optimize_spatial_recipe(recipe, baseline, *, progress=None, cancelled=None, filter_class=None):
    baseline = dict(baseline,depth=baseline["levels"])
    start = baseline["iterations"]
    limit = baseline["iteration_limit"]

    def report(step, loss):
        if progress:
            progress(start+step,loss)

    lattice = None
    if recipe.shape_gain and any(lobe.gain for lobe in recipe.lobes or ()):
        score = carrier_layout_score(recipe,baseline,filter_class)
        lattice,seed_grades = select_ray_lattice(recipe,baseline,score,cancelled)
    readout_loss = None
    if lattice is not None:
        seed = refine_carrier(baseline,lattice,iterations=64,cancelled=cancelled)
        steps = 256
        selected = train_chain(recipe,baseline,seed,steps=steps,interval=32,span=.6,
                               ray_loss=AngularRayLoss,filter_class=filter_class,progress=report,cancelled=cancelled)
        selected = select_ray_interpolation(recipe,lattice,selected,score,cancelled)
        selected["carrier_seed_selection"] = seed_grades
        readout_loss = AngularRayLoss
        solver = "Spatial DPF, carrier layout selected on held phases"
    else:
        seed = baseline
        if recipe.shape_gain and any(lobe.gain for lobe in recipe.lobes or ()):
            seed = build_carrier_seed(recipe,baseline,baseline,224,2.8)
        if seed.get("features"):
            seed = refine_carrier(baseline,seed,cancelled=cancelled)
            warm = train_chain(recipe,baseline,seed,steps=832,filter_class=filter_class,
                               progress=report,cancelled=cancelled)
            start += warm["steps"]
            limit += 832
            seed = build_carrier_seed(recipe,baseline,warm,132)
            seed = refine_carrier(warm,seed,cancelled=cancelled)
            steps = 1024
        else:
            steps = 384
        structure = select_structure(recipe,baseline,seed,filter_class=filter_class,cancelled=cancelled)
        structure = prune_fine_budget(recipe,baseline,structure,filter_class=filter_class,cancelled=cancelled)
        selected = train_chain(recipe,baseline,structure,steps=steps,filter_class=filter_class,
                               progress=report,cancelled=cancelled)
        solver = "Spatial DPF, best selection step "+str(selected["best_step"])
    response,work = spatial_readout(recipe,baseline,selected,filter_class=filter_class,
                                    cancelled=cancelled,ray_loss=readout_loss)
    selected.update(recipe=recipe.resolved(),levels=baseline["levels"],holdout_phases=response["phases"],
                    fine_radius=baseline["fine_radius"],outer_radius=baseline["outer_radius"],
                    simulation_side=response["side"],outer_pool=response["pool"],
                    iterations=start+selected["steps"],iteration_limit=limit+steps,
                    loss=selected["selection"],converged=selected["steps"]<steps,
                    solver=solver,
                    weighted_bilinear_fetches_per_full_res_pixel=work)
    return selected,response
