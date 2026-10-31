"""Independent probabilities, layout audit and numerical SG reference for the 20261031 experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import erf, i0e
from light_sampling_geometry import cube_pdf

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'builds/light-sampling/20261031-weights'
MODELS = ['original', 'halfspace-a', 'cosine-b', 'near-far', 'moment-sphere', 'diffuse-sg', 'halfspace-near-far']


def sg_cosine(k, c):
    k, c = np.broadcast_arrays(np.asarray(k, float), np.asarray(c, float))
    small = k < .05
    safe = np.maximum(k, .05)
    e = np.exp(-safe)
    front = np.where(small, .5 + k*(-1/6+k*(1/24-k/120)), (safe-1+e)/safe**2)
    back = np.where(small, .5+k*(-5/6+k*(17/24-k*49/120)), e*(1-(1+safe)*e)/safe**2)
    t = k*np.sqrt(.5*(k*k+2.7360833*k+17.021297)/(k**3+4.0100827*k*k+15.219156*k+76.087896))
    v = t*c
    q = lambda x: x*(1+erf(x))+np.exp(-x*x)/np.sqrt(np.pi)
    u = np.where(t < .01, .5*(c+1)+t*(c*c-1)/(2*np.sqrt(np.pi)),
                 (q(v)-q(-t))/(2*np.maximum(t, 1e-30)))
    return np.maximum(back+(front-back)*np.clip(u, 0, 1), 1e-6/(1+k))


def sg_cosine_float32(k, c):
    """Float32 shader expression, compared with independent quadrature below."""
    k, c = np.float32(k), np.float32(c)
    if k < .05:
        front = .5 + k*(-1/6 + k*(1/24 - k/120))
        back = .5 + k*(-5/6 + k*(17/24 - k*49/120))
    else:
        e = np.exp(-k)
        front = (k-1+e)/(k*k)
        back = e*(1-(1+k)*e)/(k*k)
    t = k*np.sqrt(.5*(k*k+2.7360833*k+17.021297)/(k*k*k+4.0100827*k*k+15.219156*k+76.087896))
    if t < .01:
        u = .5*(c+1)+t*(c*c-1)*.2820947918
    else:
        v = np.array([abs(t*c), t], dtype='f4')
        r = 1/(1+.3275911*v)
        erfc = (((((1.061405429*r-1.453152027)*r)+1.421413741)*r-.284496736)*r+.254829592)*r*np.exp(-v*v)
        q = np.exp(-v*v)*.5641895835-v*erfc
        u = np.clip((2*max(t*c, 0)+q[0]-q[1])/(2*t), 0, 1)
    return max(back+(front-back)*u, 1e-6/(1+k))


def probabilities(nodes, model, origin, normal):
    mass = nodes[:, 3]
    half = (nodes[:, 4:7]-nodes[:, :3])*.5
    delta = nodes[:, :3]+half-origin
    radius2 = np.sum(half*half, axis=1)
    if model in (4, 5):
        delta = nodes[:, 8:11]-origin
        radius2 = 2*nodes[:, 11]
    d2 = np.sum(delta*delta, axis=1)
    h = delta@normal
    cosine = np.clip((h+np.sqrt(radius2))/np.sqrt(np.maximum(d2, .25)), .001, 1)
    if model == 5:
        variance = np.where(mass > 0, nodes[:, 11], 1)
        weight = mass/variance*sg_cosine(d2/variance, np.clip(h/np.sqrt(np.maximum(d2, 1e-30)), -1, 1))
    elif model in (3, 6):
        amplitude = mass*cosine
        if model == 6:
            extent = np.abs(normal)@half.T
            excluded = h+extent < -1e-6*(np.sum(np.abs(delta*normal),axis=1)+extent+1)
            amplitude = np.where(excluded, 0, amplitude)
        nearest = np.sum(np.maximum(np.abs(delta)-half, 0)**2, axis=1)
        farthest = np.sum((np.abs(delta)+half)**2, axis=1)
        zero = (nearest == 0)&(amplitude > 0)
        near = np.where(zero, amplitude, 0) if zero.any() else amplitude/np.maximum(nearest, 1e-300)
        far = np.divide(amplitude, farthest, out=np.zeros_like(amplitude), where=farthest > 0)
        if far.sum() == 0:
            return np.zeros_like(amplitude)
        return .5*(near/near.sum()+far/far.sum())
    else:
        if model == 2:
            extent = np.abs(normal)@half.T
            height = np.where(h >= extent, h, np.maximum(h+extent, 0)**2/(4*np.maximum(extent, 1e-30)))
            cosine = np.clip(height/np.sqrt(np.maximum(d2, .25)), .001, 1)
        weight = mass*cosine/np.maximum.reduce([d2, radius2, np.full_like(d2, .25)])
        if model in (1, 2):
            extent = np.abs(normal)@half.T
            weight = np.where(h+extent < -1e-6*(np.sum(np.abs(delta*normal),axis=1)+extent+1), 0, weight)
    return weight/weight.sum() if weight.sum() > 0 else np.zeros_like(weight)


def load_tree(path, model):
    words = np.fromfile(path, '<u4')
    header = words[:8]
    stride = 12 if model in (4, 5) else 8
    nodes = words[8:].reshape(-1, stride)
    return header, nodes.view('<f4').astype(float), nodes[:, 7]


def distribution(nodes, links, origin, normal, model):
    leaves, zero = {}, 0.
    pending = [(1, 1.)]
    while pending:
        index, p = pending.pop()
        if nodes[index, 3] == 0:
            zero += p
        elif links[index]&0x80000000:
            leaves[index] = p
        else:
            link = int(links[index]); base, count = link&0x0fffffff, ((link>>28)&7)+1
            probs = probabilities(nodes[base:base+count], model, origin, normal)
            if probs.sum() == 0:
                zero += p
            else:
                pending.extend((base+j, p*w) for j,w in enumerate(probs) if w > 0)
    assert abs(sum(leaves.values())+zero-1) < 1e-9
    return leaves, zero


def audit_capture(directory, model, arity=4):
    directory = Path(directory)
    header, nodes, links = load_tree(directory/'tree.bin', model)
    assert header[1] > 0 and header[2] == header[3] == 0
    pending, seen, leaves = [1], set(), {}
    while pending:
        i = pending.pop(); assert i not in seen; seen.add(i)
        node = nodes[i]; assert np.isfinite(node[:7]).all()
        if node[3] == 0: continue
        if links[i]&0x80000000:
            assert np.all(node[4:7]-node[:3] == 1)
            leaves[tuple(node[:3])] = i
            if model in (4, 5):
                assert np.allclose(node[8:11], node[:3]+.5) and np.isclose(node[11], 5/12)
            continue
        link = int(links[i]); base, count = link&0x0fffffff, ((link>>28)&7)+1
        assert 1 <= count <= arity and base > 0 and base+count <= len(nodes)
        child = nodes[base:base+count]; assert np.isclose(node[3], child[:, 3].sum(), rtol=2e-6)
        active = child[child[:, 3] > 0]
        assert np.all(active[:, :3] >= node[:3]) and np.all(active[:, 4:7] <= node[4:7])
        if model in (4, 5):
            mean = np.average(active[:, 8:11], axis=0, weights=active[:, 3])
            variance = np.average(active[:, 11]+np.sum((active[:, 8:11]-mean)**2,axis=1), weights=active[:, 3])
            assert np.allclose(mean, node[8:11], atol=.003)
            assert np.isclose(variance, node[11], rtol=.002, atol=.003)
        pending.extend(range(base, base+count))
    assert len(leaves) == header[1]
    data = np.fromfile(directory/'audit.bin', '<f4')[:1024].reshape(64,4,4).astype(float)
    reasons = np.fromfile(directory/'audit.bin', '<u4')[1024:1040]
    assert int(sum(map(int, reasons[:10]))) == header[6]
    pdf_errors, cube_errors, zero_probs = [], [], []
    for record in data:
        if record[3,3] == 0: continue
        q, zero = distribution(nodes, links, record[0,:3], record[1,:3], model)
        zero_probs.append(zero)
        if record[3,3] < 0:
            assert zero > 0, 'GPU returned a zero selection outside the reconstructed zero-result support'
        if record[3,3] > 0:
            i = leaves[tuple(record[3,:3])]
            pdf_errors.append(abs(record[1,3]/q[i]-1))
            if record[2,3] > 0:
                expected = cube_pdf(record[3,:3],record[0,:3],record[1,:3],record[2,:3])
                cube_errors.append(abs(record[2,3]/expected-1))
    assert zero_probs, "No receiving audit points"
    assert not pdf_errors or max(pdf_errors) < .01,pdf_errors
    assert not cube_errors or max(cube_errors) < .005,cube_errors
    leaf_set = sorted((p,int(links[i]&0x7fffffff),float(nodes[i,3])) for p,i in leaves.items())
    return dict(emitters=len(leaves),leaf_set_sha256=hashlib.sha256(repr(leaf_set).encode()).hexdigest(),
                node_bytes=48 if model in (4, 5) else 32,buffer_bytes=(directory/'tree.bin').stat().st_size,
                max_pdf_relative_error=max(pdf_errors,default=0),max_cube_relative_error=max(cube_errors,default=0),
                zero_probabilities=zero_probs,zero_records=int(np.sum(data[:,3,3]<0)),cube_sample_count=len(cube_errors),reasons=list(map(int,reasons)),sample_count=len(pdf_errors))


def validate():
    checks=[]
    for k in [0,1e-5,.01,.049,.05,.1,1,10,100,1000]:
        for c in np.linspace(-1,1,21):
            transverse=np.sqrt(max(1-c*c,0))
            # Independent azimuth-integrated hemisphere quadrature using the Bessel function.
            def integrand(z):
                a=k*transverse*np.sqrt(max(1-z*z,0))
                return z*np.exp(k*(c*z-1)+a)*i0e(a)
            reference,error=quad(integrand,0,1,epsabs=1e-13,points=[float(np.clip(c,0,1))],limit=300)
            actual=float(sg_cosine(k,c))
            normalization=k/(-np.expm1(-2*k)) if k>0 else .5
            checks.append(dict(k=k,c=float(c),absolute_normalized_error=abs(actual-reference)*normalization,
                               float32_expression_normalized_error=float(abs(sg_cosine_float32(k,c)-reference)*normalization),quadrature_error=error))
    assert max(x['absolute_normalized_error'] for x in checks)<.006
    assert max(x['float32_expression_normalized_error'] for x in checks)<.006
    addressing=[]
    for w,h in [(1280,720),(1920,1080)]:
        points=[((2*x+1)*w//16,(2*y+1)*h//16,y*8+x) for y in range(8) for x in range(8)]
        assert len(set((x,y) for x,y,_ in points))==64
        assert all(((x*8//w)*2+1)*w//16==x and ((y*8//h)*2+1)*h//16==y for x,y,_ in points)
        addressing.append(dict(width=w,height=h,writers=len(points),max_slot=max(s for _,_,s in points)))
    low=np.array([[0,0,0],[0,0,-1],[0,0,1],[0,0,0]],float)
    node=np.zeros((4,12));node[:,:3]=low;node[:,4:7]=low+1;node[:,3]=[1,1e-6,1000,1];node[:,8:11]=low+.5;node[:,11]=5/12
    for origin in [[.5,.5,.5],[0,0,0],[.5,.5,1+1e-6],[0,0,1e5]]:
        for model in range(6):
            q=probabilities(node,model,np.array(origin),np.array([0,0,1]))
            assert np.isfinite(q).all() and np.all(q>=0) and (abs(q.sum()-1)<1e-12 or q.sum()==0)
    normal=np.array([-.005037308670580387,-.7657549977302551,-.6431126594543457], dtype='f4')
    delta=np.array([52.23280334472656,93.40628051757812,-110.5288314819336], dtype='f4')
    h=np.sum(normal*delta,dtype='f4'); extent=np.sum(np.abs(normal)*np.float32(.5),dtype='f4')
    actual=float(np.sum(normal.astype(float)*delta.astype(float))+np.sum(np.abs(normal.astype(float))*.5))
    old_margin=np.float32(1e-6)*(abs(h)+extent+1)
    new_margin=np.float32(1e-6)*(np.sum(np.abs(normal*delta),dtype='f4')+extent+1)
    assert actual>0 and h+extent < -old_margin and not h+extent < -new_margin
    result=dict(sg_quadrature=checks,audit_addressing=addressing,support_edge_cases='passed',
                cancellation_regression=dict(exact_front_extent=actual,float_front_extent=float(h+extent),
                                             old_margin=float(old_margin),new_margin=float(new_margin)))
    RUN.mkdir(parents=True,exist_ok=True);(RUN/'host-reference.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(sg_max_normalized_error=max(x['absolute_normalized_error'] for x in checks),sg_checks=len(checks),audit_addressing=addressing)))


if __name__=='__main__':
    validate()
