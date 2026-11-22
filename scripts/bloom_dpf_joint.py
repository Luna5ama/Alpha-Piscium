"""Forward and adjoint evaluation for a normalized spatial Bloom DPF chain."""
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple
import math

import numpy as np
from scipy.sparse import csr_matrix

import bloom_dpf_model as model
from bloom_dpf_fit import coarse_loss, coarse_target, loss_weights, measure, readout_geometry, target_mass
from bloom_dpf_target import RasterTarget, lobe_fit_geometry, target_components

target_components = lru_cache(maxsize=1)(target_components)



def axis_terms(position):
    nearest = round(position)
    if abs(position-nearest) <= 16*math.ulp(max(1.,abs(position))):
        position = nearest
    low = math.floor(position)
    fraction = position-low
    if fraction==0:
        return ((low-1,0.,-.5),(low,1.,0.),(low+1,0.,.5))
    return ((low,1-fraction,-1.),(low+1,fraction,1.))


class Stencil:
    def __init__(self, source, destination, offsets):
        self.source = source
        self.destination = destination
        self.taps = []
        for ox,oy in offsets:
            terms = []
            for y,wy,dy in axis_terms(oy):
                for x,wx,dx in axis_terms(ox):
                    coefficient = wx*wy
                    gx,gy = dx*wy,wx*dy
                    if not (coefficient or gx or gy):
                        continue
                    shift = (y,x)
                    start = tuple(max(a,b-k) for a,b,k in zip(destination.start,source.start,shift,strict=True))
                    end = tuple(min(a,b-k) for a,b,k in zip(destination.end,source.end,shift,strict=True))
                    if any(a>b for a,b in zip(start,end,strict=True)):
                        continue
                    out = (...,*(slice(a-o,b-o+1) for a,b,o in zip(start,end,destination.start,strict=True)))
                    inp = (...,*(slice(a+k-o,b+k-o+1) for a,b,k,o in zip(start,end,shift,source.start,strict=True)))
                    terms.append((inp,out,coefficient,gx,gy))
            self.taps.append(terms)

    def apply(self, source, weights, dest=None):
        if dest is None:
            dest = np.zeros((len(source),3,*self.destination.shape),np.float32)
        for w,terms in zip(weights,self.taps,strict=True):
            for inp,out,value,_,_ in terms:
                if value:
                    dest[out] += source[inp]*(w*value)[:,None,None]
        return dest

    def backward(self, source, dest_gradient, weights):
        raw_gradient = np.zeros_like(dest_gradient,shape=(len(source),3,*self.source.shape))
        weight_gradient = np.zeros_like(weights,dtype=np.float64)
        offset_gradient = np.zeros((len(weights),2),np.float64)
        for index,(w,terms) in enumerate(zip(weights,self.taps,strict=True)):
            for inp,out,value,gx,gy in terms:
                delta = dest_gradient[out]
                product = np.sum(source[inp]*delta,axis=(0,2,3),dtype=np.float64)
                if value:
                    raw_gradient[inp] += delta*(w*value)[:,None,None]
                    weight_gradient[index] += product*value
                if gx or gy:
                    position_gradient = product @ w
                    offset_gradient[index] += (gx*position_gradient,gy*position_gradient)
        return raw_gradient,offset_gradient,weight_gradient


@dataclass(frozen=True)
class Window:
    start: tuple
    shape: tuple

    @property
    def end(self):
        return tuple(a+n-1 for a,n in zip(self.start,self.shape,strict=True))

    def union(self, other):
        start = tuple(min(a,b) for a,b in zip(self.start,other.start,strict=True))
        end = tuple(max(a,b) for a,b in zip(self.end,other.end,strict=True))
        return Window(start,tuple(b-a+1 for a,b in zip(start,end,strict=True)))




def support(source, positions, src_side, dst_side, down):
    scale,bias = (1.,0.) if down is None else ((2.,.5) if down else (.5,-.25))
    start,end = [],[]
    for axis,(lo,hi) in enumerate(zip(source.start,source.end,strict=True)):
        shift = positions[:,1-axis]
        left = max(0,math.ceil((lo-1-bias-float(shift.max()))/scale))
        right = min(dst_side-1,math.floor((hi+1-bias-float(shift.min()))/scale))
        if down is False:
            if lo==0:
                left = 0
            if hi==src_side-1:
                right = dst_side-1
        start.append(min(left,dst_side-1))
        end.append(max(right,start[-1]))
    return Window(tuple(start),tuple(b-a+1 for a,b in zip(start,end,strict=True)))


class Carrier(NamedTuple):
    angle: float
    level: int
    short: np.ndarray
    short_weights: np.ndarray
    long: np.ndarray
    long_weights: np.ndarray


class JointDPF:
    def __init__(self, recipe, depth, radius, outer, phases, *, filter_class=None):
        self.filter_class = filter_class or model.Filter
        self.depth = depth
        self.radius = radius
        self.outer = outer
        self.size = recipe.size
        self.period = 2**depth
        self.side = 2*self.period*math.ceil((outer+self.period+8)/self.period)
        self.pool = 8
        self.shape_limit = (radius-16 if isinstance(recipe, RasterTarget) else
                            lobe_fit_geometry(recipe)[0] if recipe.lobes is not None else 144*recipe.size)
        self.phases = tuple((x%self.period, y%self.period) for x,y in phases)
        self.roi,self.regions = readout_geometry(self.side,self.phases,radius)
        coords = np.asarray([(self.side//2+y,self.side//2+x) for x,y in self.phases])
        lo,hi = coords.min(axis=0),coords.max(axis=0)
        self.input_window = Window(tuple(lo),tuple(hi-lo+1))
        self.impulse = np.zeros((len(coords),1,*self.input_window.shape),np.float32)
        self.impulse[np.arange(len(coords)),0,coords[:,0]-lo[0],coords[:,1]-lo[1]] = 1.
        all_parts = target_components(recipe,outer)
        complete = sum(all_parts.values())
        self.mass = np.broadcast_to(target_mass(recipe,complete),(3,)).astype(np.float32)
        if np.any(self.mass <= 0):
            raise ValueError("The DPF target has zero mass")
        win = np.s_[...,outer-radius:outer+radius+1,outer-radius:outer+radius+1]
        self.target,self.importance = loss_weights({k:v[win] for k,v in all_parts.items()},radius,recipe.size,self.shape_limit)
        floor = max(float(self.target.max())*2e-4,1e-12)
        weak = 1/np.square(np.maximum(self.target,floor))
        self.importance += .65*weak/np.sum(self.target**2*weak,dtype=np.float64)
        self.coarse,self.outer_mask = coarse_target(complete,self.side,self.phases,self.pool,self.shape_limit)
        self.quality_floor = None
        self.fine_aux = None
    @staticmethod
    def normalize(logits):
        exponent = np.exp(logits-logits.max(axis=0))
        return (exponent/exponent.sum(axis=0)).astype(np.float32)

    def predict(self, positions, weights, injection, *, gradient=False, features=()):
        raw = [self.impulse]
        raw_windows = [self.input_window]
        down = []
        for i,(p,w) in enumerate(zip(positions[:self.depth],weights[:self.depth],strict=True)):
            source = raw[-1]
            window = raw_windows[-1]
            src_side,dst_side = self.side>>i,self.side>>(i+1)
            dst = support(window,p,src_side,dst_side,True)
            op = self.filter_class(window.shape,dst.shape,p,down=True,start=dst.start,
                              source_start=window.start,full_source=(src_side,)*2)
            down.append(op)
            raw.append(op.apply(source,w))
            raw_windows.append(dst)

        stencil = [None]*self.depth
        feature_windows = [[] for _ in range(self.depth)]
        feature_state = [[] for _ in range(self.depth)]
        feature_gradients = [None] * len(features)
        offsets,probabilities = injection
        injection_weights = [p*self.mass for p in probabilities]
        windows = [support(r,p,self.side>>i,self.side>>i,None)
                   for i,(r,p) in enumerate(zip(raw_windows[1:],offsets,strict=True),1)]
        for index, branch in enumerate(features):
            level = branch.level
            size = self.side >> level
            carrier_window = support(raw_windows[level],branch.short,size,size,None)
            feature_windows[level].append((index,branch,carrier_window))
            windows[level-1] = windows[level-1].union(support(carrier_window,branch.long,size,size,None))
        window = windows[-1]
        stencil[-1] = Stencil(raw_windows[-1],window,offsets[-1])
        image = stencil[-1].apply(raw[-1],injection_weights[-1])
        up = [None]*self.depth
        inputs = [None]*self.depth
        for level in range(self.depth-1,-1,-1):
            p,w = positions[self.depth+level],weights[self.depth+level]
            inputs[level] = image
            src_side,dst_side = self.side>>(level+1),self.side>>level
            if level==0:
                start,shape = self.roi
                dst = Window(start,shape)
                pooled = self.filter_class(window.shape,(self.side//self.pool,)*2,p,pool=self.pool,
                                      source_start=window.start,full_source=(src_side,)*2)
                outer = pooled.apply(image,w)
            else:
                dst = support(window,p,src_side,dst_side,False).union(windows[level-1])
            op = self.filter_class(window.shape,dst.shape,p,start=dst.start,
                              source_start=window.start,full_source=(src_side,)*2)
            up[level] = op
            image = op.apply(image,w)
            if level:
                stencil[level-1] = Stencil(raw_windows[level],dst,offsets[level-1])
                stencil[level-1].apply(raw[level],injection_weights[level-1],image)
                for index,branch,carrier_window in feature_windows[level]:
                    first = Stencil(raw_windows[level],carrier_window,branch.short)
                    carrier = first.apply(raw[level],branch.short_weights)
                    second = Stencil(carrier_window,dst,branch.long)
                    long_weights = branch.long_weights*self.mass
                    second.apply(carrier,long_weights,image)
                    if gradient:
                        feature_state[level].append((index,first,second,carrier,branch.short_weights,long_weights))
            window = dst
        if not gradient:
            return image,outer

        delta = np.zeros_like(image)
        loss = 0.
        for region in self.regions:
            error = image[region]-self.target
            delta[region] = error*self.importance/len(self.regions)
            loss += .5*np.sum(error*delta[region],dtype=np.float64)
        if self.fine_aux is not None:
            loss += self.fine_aux.apply(image,self.regions,delta)
        pooled_loss,pooled_delta = coarse_loss(outer,self.coarse,self.outer_mask,self.pool,self.mass)
        loss += pooled_loss
        if self.quality_floor is not None:
            loss += self.quality_loss(image,outer,delta,pooled_delta)
        dpos = [None]*len(positions)
        dweight = [None]*len(weights)
        draw = [None]*len(raw)
        dipos = [None]*self.depth
        diweight = [None]*self.depth
        for i,op in enumerate(up):
            if i:
                draw[i],dipos[i-1],diweight[i-1] = stencil[i-1].backward(raw[i],delta,injection_weights[i-1])
                for index,first,second,carrier,sw,lw in feature_state[i]:
                    dcarrier,dlong,dlw = second.backward(carrier,delta,lw)
                    draw_raw,dshort,dsw = first.backward(raw[i],dcarrier,sw)
                    draw[i] += draw_raw
                    feature_gradients[index] = (dshort,dsw,dlong,dlw*self.mass)
            k = self.depth+i
            delta,dweight[k],dpos[k] = op.backward_positions(inputs[i],delta,weights[k])
            if i==0:
                dx,dw,dp = pooled.backward_positions(inputs[0],pooled_delta,weights[k])
                delta += dx
                dweight[k] += dw
                dpos[k] += dp
        draw[-1],dipos[-1],diweight[-1] = stencil[-1].backward(raw[-1],delta,injection_weights[-1])
        for i in range(self.depth-1,-1,-1):
            delta,dweight[i],dpos[i] = down[i].backward_positions(raw[i],draw[i+1],weights[i],input_gradient=i>0)
            if i:
                draw[i] = delta if draw[i] is None else draw[i]+delta
        result = loss,dpos,dweight,dipos,[v*self.mass for v in diweight]
        return (*result,feature_gradients) if features else result

    def set_quality_floor(self, image, outer):
        axis = np.arange(-self.radius,self.radius+1)
        distance = np.hypot(axis[:,None],axis[None,:])
        masks = (np.ones_like(distance,dtype=bool),distance<=6*self.size,
                 (distance>6*self.size)&(distance<=self.shape_limit))
        reference = []
        for i,region in enumerate(self.regions):
            error = image[region]-self.target
            far = outer[i]-self.coarse[i]
            energies = [np.sum(error[:,m]**2,dtype=np.float64) for m in masks]
            energies.append(np.sum(far[:,self.outer_mask[i]]**2,dtype=np.float64))
            reference.append(energies)
        self.quality_floor = masks,reference

    def quality_loss(self, image, outer, delta, outer_delta):
        masks,reference = self.quality_floor
        result = 0.
        for i,region in enumerate(self.regions):
            error = image[region]-self.target
            far = outer[i]-self.coarse[i]
            pairs = [(error,delta[region],mask) for mask in masks]
            pairs.append((far,outer_delta[i],self.outer_mask[i]))
            for (e,d,m),scale,limit in zip(pairs,(.04,.32,.16,.24),reference[i],strict=True):
                limit = max(float(limit),1e-20)
                values = e[:,m]
                excess = max(np.sum(values**2,dtype=np.float64)/limit-.97,0.)
                scale /= len(self.regions)
                result += scale*excess**2
                d[:,m] += (4*scale*excess/limit)*values
        return result

    def metrics(self, image, outer):
        return measure(image,self.target,outer,self.coarse,self.outer_mask,self.regions,self.phases,
                       self.radius,self.pool,self.size,self.shape_limit)


def phase_sets(depth):
    period = 2**depth
    all_phases = [(x,y) for y in range(period) for x in range(period)]
    if depth <= 3:
        batches = [[(2*x+px,2*y+py) for py in range(2) for px in range(2)]
                   for y in range(period//2) for x in range(period//2)]
        return batches, all_phases
    rng = np.random.default_rng(0xB100)
    cells = {}
    for ly in range(4):
        for lx in range(4):
            phases = [(x,y) for x,y in all_phases if x%4==lx and y%4==ly]
            order = rng.permutation(len(phases))
            cells[lx,ly] = [phases[i] for i in order]
    train = [[cells[lx+px,ly+py][k] for py in range(2) for px in range(2)]
             for k in range(4) for ly in (0,2) for lx in (0,2)]
    held = [cells[lx,ly][4+k] for k in range(4) for ly in range(4) for lx in range(4)]
    return train,held


DIAGONALS = np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])


READOUT = ("relative_l2","core_relative_l2","shape_scale_relative_l2","pooled_outer_relative_l2")


def arrays(stages):
    return [np.asarray(x["offsets"]) for x in stages],[np.asarray(x["rgb_weights"]) for x in stages]


def group_params(positions, weights, depth):
    p,w = [],[]
    for offsets,probs in zip(positions[:depth],weights[:depth],strict=True):
        groups = offsets[1:].reshape(-1,4,2)
        p.append(np.max(abs(groups),axis=1))
        grouped = probs[1:].reshape(-1,4,3).sum(axis=1)
        w.append(np.vstack((probs[:1],grouped)))
    return p+positions[depth:],w+weights[depth:]


def expand(packed, probs, depth):
    pos,weights = packed.copy(),probs.copy()
    for i in range(depth):
        offsets = (packed[i][:,None,:]*DIAGONALS).reshape(-1,2)
        pos[i] = np.vstack((np.zeros((1,2)),offsets))
        weights[i] = np.vstack((probs[i][:1],np.repeat(probs[i][1:]/4,4,axis=0)))
    return pos,weights


def group_gradient(dp,dw):
    return ((dp[1:].reshape(-1,4,2)*DIAGONALS).sum(axis=1),
            np.vstack((dw[:1],dw[1:].reshape(-1,4,3).mean(axis=1))))


def objective_model(model, state, position_span=.75):
    batches = model if isinstance(model,list) else [model]
    depth = batches[0].depth
    positions,weights,ip,iw,*extra = state
    features = extra[0] if extra else ()
    feature_count = len(features)
    sp = [c.short for c in features]
    sw = [c.short_weights for c in features]
    lp = [c.long for c in features]
    lw = [c.long_weights for c in features]
    p,w = group_params(positions,weights,depth)
    z = np.concatenate([*iw,*lw])
    limits = np.cumsum([len(q) for q in (*iw,*lw)])[:-1]
    def logs(v):
        return np.log(np.maximum(v,1e-9))
    params = [*p,*[logs(v) for v in w],*ip,*sp,*lp,*[logs(v) for v in sw],logs(z)]
    splits = np.cumsum([a.size for a in params])[:-1]
    x0 = np.concatenate([a.ravel() for a in params])
    bounds = []
    for i,a in enumerate(params):
        if i<2*depth:
            delta = position_span
        elif 4*depth<=i<5*depth+2*feature_count:
            delta = 0. if i==5*depth-1 else position_span
        else:
            delta = 4.
        bounds.extend((v-delta,v+delta) for v in a.ravel())

    def decode(x):
        s = [a.reshape(p.shape) for a,p in zip(np.split(x,splits),params,strict=True)]
        probabilities = [JointDPF.normalize(x) for x in s[2*depth:4*depth]]
        pos,weights = expand(s[:2*depth],probabilities,depth)
        stencil = JointDPF.normalize(s[-1])
        raw = np.split(stencil,limits)
        state = (pos,weights,s[4*depth:5*depth],raw[:depth])
        if features:
            start = 5*depth
            short = s[start:start+feature_count]
            long = s[start+feature_count:start+2*feature_count]
            sw = [JointDPF.normalize(q) for q in s[start+2*feature_count:-1]]
            state += ([Carrier(c.angle,c.level,sp,sw,lp,lw)
                       for c,sp,sw,lp,lw in zip(features,short,sw,long,raw[depth:],strict=True)],)
        return state,probabilities,stencil

    def function(x, selected_batches=batches):
        (p,w,ip,iw,*extra),q,stencil = decode(x)
        branches = extra[0] if extra else ()
        results = [m.predict(p,w,(ip,iw),gradient=True,features=branches) for m in selected_batches]
        value = np.mean([r[0] for r in results])
        dp,dw,dip,diw = [[np.mean(parts,axis=0) for parts in zip(*(r[i] for r in results),strict=True)]
                         for i in range(1,5)]
        for i in range(depth):
            dp[i],dw[i] = group_gradient(dp[i],dw[i])
        dw = [p*(g-np.sum(p*g,axis=0)) for p,g in zip(q,dw,strict=True)]
        dshort,dlong,dsw = [],[],[]
        for i,branch in enumerate(branches):
            sw = branch.short_weights
            ps,ws,pl,wl = [np.mean(parts,axis=0) for parts in zip(*(r[5][i] for r in results),strict=True)]
            dshort.append(ps)
            dlong.append(pl)
            dsw.append(sw*(ws-np.sum(sw*ws,axis=0)))
            diw.append(wl)
        diw = np.concatenate(diw)
        diw = stencil*(diw-np.sum(stencil*diw,axis=0))
        return value,np.concatenate([a.ravel() for a in (*dp,*dw,*dip,*dshort,*dlong,*dsw,diw)])

    return x0,bounds,decode,function


class RayLoss:
    def __init__(self, recipe, radius, target, strength=.08):
        self.strength = strength
        self.profiles = []
        side = 2*radius+1
        pixels = side*side
        for lobe in recipe.lobes or ():
            a = math.radians(lobe.angle)
            d = np.arange(8,min(radius-8,lobe.length*recipe.size*1.4))
            if not len(d) or not lobe.gain:
                continue
            bands = (
                ((-5.,-1.,0.,1.,5.),(-.5,1/3,1/3,1/3,-.5),.7),
                ((-2.,0.,2.),(-.5,1.,-.5),.3),
            )
            for offsets,coefficients,importance in bands:
                rows,columns,values = [],[],[]
                for across,band in zip(offsets,coefficients,strict=True):
                    u = across*recipe.size
                    x = radius+math.cos(a)*d-math.sin(a)*u
                    y = radius+math.sin(a)*d+math.cos(a)*u
                    ix,iy = np.floor(x).astype(int),np.floor(y).astype(int)
                    fx,fy = x-ix,y-iy
                    for i,wx in enumerate((1-fx,fx)):
                        for j,wy in enumerate((1-fy,fy)):
                            xx,yy = ix+i,iy+j
                            valid = (xx>=0)&(xx<side)&(yy>=0)&(yy<side)
                            rows.extend(np.flatnonzero(valid))
                            columns.extend((yy[valid]*side+xx[valid]).tolist())
                            values.extend((band*wx[valid]*wy[valid]).tolist())
                matrix = csr_matrix((np.asarray(values,np.float32),(rows,columns)),shape=(len(d),pixels))
                desired = matrix @ target.reshape(3,-1).T
                peak = np.maximum(desired.max(axis=0),1e-12)
                visible = desired >= .025*peak
                scale = np.maximum(np.abs(desired),.08*peak)
                weight = np.where(visible,1./scale**2,0.).astype(np.float32)
                weight *= importance/max(1,np.count_nonzero(visible))
                adjacent = .5*(weight[1:]+weight[:-1])
                self.profiles.append((matrix,matrix.T.tocsr(),desired,weight,adjacent))

    def intensities(self,image,regions):
        if not self.profiles:
            return np.empty(0,dtype=np.float32)
        flats = [image[region].reshape(3,-1).T for region in regions]
        result = []
        for matrix,_,desired,weight,_ in self.profiles[1::2]:
            visible = weight>0
            expected = np.sum(desired[visible],dtype=np.float64)
            if expected<=1.e-12:
                continue
            actual = sum(np.sum(np.maximum(matrix @ flat,0.)[visible],dtype=np.float64) for flat in flats)
            result.append(actual/(len(flats)*expected))
        return np.asarray(result)

    def apply(self, image, regions, delta=None):
        if not self.profiles:
            return 0.
        strength = 2*self.strength/(len(regions)*len(self.profiles))
        loss = 0.
        for r in regions:
            fine = image[r]
            derivative = None if delta is None else delta[r]
            flat = fine.reshape(3,-1).T
            for matrix,adjoint,desired,weight,adjacent in self.profiles:
                residual = matrix @ flat - desired
                ripple = np.diff(residual,axis=0)
                missing = np.minimum(residual+.10*np.maximum(desired,0.),0.)
                loss += strength*float(np.sum(weight*(.5*residual**2+.25*missing**2),dtype=np.float64)
                                       +.125*np.sum(adjacent*ripple**2,dtype=np.float64))
                if derivative is not None:
                    profile_gradient = weight*(residual+.5*missing)
                    ripple_grad = .25*adjacent*ripple
                    profile_gradient[:-1] -= ripple_grad
                    profile_gradient[1:] += ripple_grad
                    derivative += strength*(adjoint @ profile_gradient).T.reshape(fine.shape)
        return loss


class AngularRayLoss(RayLoss):
    def __init__(self, recipe, radius, target, strength=.08):
        super().__init__(recipe, radius, target, strength)
        self.contrast = []
        lobes = sorted((lobe for lobe in recipe.lobes or () if lobe.gain), key=lambda lobe: lobe.angle)
        side = 2*radius+1
        flat = target.reshape(3,-1).T
        for index,lobe in enumerate(lobes):
            theta = np.deg2rad(lobe.angle)
            reach = lobe.length*recipe.size
            distance = np.arange(max(6,.22*reach),min(radius-1,1.15*reach))
            if len(distance)<2:
                continue
            rows = np.arange(len(distance))
            x = radius+distance*np.cos(theta)
            y = radius+distance*np.sin(theta)
            gaps = ((lobes[index-1].angle-lobe.angle)%360-360,
                    (lobes[(index+1)%len(lobes)].angle-lobe.angle)%360 or 360)
            profiles = []
            for gap in gaps:
                # Use the valley between nearby rays and a local sideband for isolated rays.
                width = np.minimum(4*lobe.width*recipe.size,2*distance*np.sin(np.deg2rad(min(abs(gap),90)/4)))
                sideband = theta+np.deg2rad(np.clip(gap,-90,90))/4
                width = np.copysign(width,gap)
                a,b,c = [],[],[]
                for sign,xx,yy in ((1,x,y),(-1,x-width*np.sin(sideband),y+width*np.cos(sideband))):
                    ix,iy = np.floor(xx).astype(int),np.floor(yy).astype(int)
                    fx,fy = xx-ix,yy-iy
                    for i,wx in enumerate((1-fx,fx)):
                        for j,wy in enumerate((1-fy,fy)):
                            u,v = ix+i,iy+j
                            valid = (u>=0)&(u<side)&(v>=0)&(v<side)
                            a.extend(rows[valid])
                            b.extend(v[valid]*side+u[valid])
                            c.extend(sign*(wx*wy)[valid])
                matrix = csr_matrix((np.asarray(c,np.float32),(a,b)),shape=(len(rows),side**2))
                desired = matrix @ flat
                peak = desired.max(axis=0)
                visible = (desired>1.e-12)&(desired>=.12*peak)
                scale = np.maximum(desired,.20*peak).clip(min=1.e-12)
                weight = np.where(visible,1/scale**2,0)/max(1,visible.sum())
                profiles.append((matrix,matrix.T.tocsr(),desired,weight,visible,scale))
            self.contrast.append(profiles)

    def intensities(self,image,regions):
        flats = [image[region].reshape(3,-1).T for region in regions]
        result = []
        for profiles in self.contrast:
            local = []
            for flat in flats:
                values = [(matrix @ flat)[visible]/scale[visible]
                          for matrix,_,_,_,visible,scale in profiles if visible.any()]
                if values:
                    local.append(np.quantile(np.concatenate(values),.25))
            if local:
                result.append(min(local))
        return np.asarray(result)

    def roughness(self,image,regions):
        flats = [image[region].reshape(3,-1).T for region in regions]
        roughness = []
        for profiles in self.contrast:
            for matrix,_,desired,_,visible,scale in profiles:
                adjacent = visible[1:]&visible[:-1]
                if adjacent.any():
                    for flat in flats:
                        residual = (matrix @ flat-desired)/scale
                        roughness.append(np.mean(abs(np.diff(residual,axis=0)[adjacent])))
        return float(np.mean(roughness)) if roughness else 0.

    def apply(self,image,regions,delta=None):
        loss = super().apply(image,regions,delta)
        if not self.contrast:
            return loss
        strength = .30/(len(regions)*len(self.contrast))
        for region in regions:
            fine = image[region]
            flat = fine.reshape(3,-1).T
            for profiles in self.contrast:
                for matrix,adjoint,desired,weight,_,_ in profiles:
                    actual = matrix @ flat
                    residual = actual-desired
                    missing = np.minimum(actual-.65*desired,0)
                    loss += strength*float(np.sum(weight*(.25*residual**2+.75*missing**2)))
                    if delta is not None:
                        gradient = adjoint @ (weight*(.5*residual+1.5*missing))
                        delta[region] += strength*gradient.T.reshape(fine.shape)
        return loss
