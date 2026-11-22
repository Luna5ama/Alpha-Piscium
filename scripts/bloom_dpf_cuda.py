"""CUDA execution for the CPU model's exact separable bilinear matrices."""
import ctypes as C
import os
from pathlib import Path

import numpy as np

import bloom_dpf_model as model


def library():
    dll_path = Path(os.environ["BLOOM_DPF_CUDA_LIBRARY"]).resolve(strict=True)
    toolkit = os.environ.get("CUDA_PATH")
    if toolkit:
        with os.add_dll_directory(str(Path(toolkit) / "bin")):
            dll = C.CDLL(str(dll_path))
    else:
        dll = C.CDLL(str(dll_path))
    p = C.c_void_p
    i = C.c_int
    dll.dpf_filter.argtypes = [p,p,p,*([i]*8),p,p,i,p,i,p,p,p,p]
    dll.dpf_filter.restype = i
    dll.dpf_error.argtypes = [i]
    dll.dpf_error.restype = C.c_char_p
    return dll


_LIB = None


def pack(matrices):
    plan, indices, coeffs = [], [], []
    i, v = 0, 0
    for matrix in matrices:
        rows, columns, data = matrix.indptr,matrix.indices,matrix.data
        plan.extend((i,i+len(rows),v))
        indices.extend((rows,columns))
        coeffs.append(data)
        i += len(rows)+len(columns)
        v += len(data)
    return (np.asarray(plan,np.int32),np.concatenate(indices).astype(np.int32,copy=False),
            np.concatenate(coeffs).astype(np.float32,copy=False))


def pointer(array):
    return None if array is None else array.ctypes.data


class Filter(model.Filter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.forward = pack(m for x,y in zip(self.x,self.y,strict=True) for m in (x[0],y[0]))

    def run(self, image, weights, derivative=None, input_gradient=False):
        global _LIB
        if _LIB is None:
            _LIB = library()
        image = np.ascontiguousarray(image,dtype=np.float32)
        phases,source_channels,sh,sw = image.shape
        dh,dw = self.destination
        weights = np.ascontiguousarray(weights,dtype=np.float32)
        taps,channels = weights.shape
        if source_channels not in (1,channels):
            raise ValueError("The source channel count must match the tap weights or broadcast from one")
        if derivative is None:
            plans,indices,coeff = self.forward
            out = np.empty((phases,channels,dh,dw),np.float32)
            dx = dwgt = dpos = None
        else:
            source,offsets,down,start,pool,src_start,full_source = self.geometry
            matrices = []
            for (ox,oy),x,y in zip(offsets,self.x,self.y,strict=True):
                xd = model.axis_matrix(sw,dw,float(ox),down,start[1],pool,True,src_start[1],full_source[1])[0]
                yd = model.axis_matrix(sh,dh,float(oy),down,start[0],pool,True,src_start[0],full_source[0])[0]
                matrices.extend((x[0],y[0],x[1],y[1],xd,yd))
            plans,indices,coeff = pack(matrices)
            derivative = np.ascontiguousarray(derivative,dtype=np.float32)
            if derivative.shape != (phases,channels,dh,dw):
                raise ValueError("The derivative must match the filter output shape")
            out = None
            dx = np.empty((phases,channels,sh,sw),np.float32) if input_gradient else None
            dwgt = np.empty((taps,channels),np.float64)
            dpos = np.empty((taps,2),np.float64)
        code = _LIB.dpf_filter(pointer(image),pointer(derivative),pointer(weights),
            phases,source_channels,channels,sh,sw,dh,dw,taps,
            pointer(plans),pointer(indices),len(indices),pointer(coeff),len(coeff),
            pointer(out),pointer(dx),pointer(dwgt),pointer(dpos))
        if code:
            raise RuntimeError(f"CUDA filter error {code}: {_LIB.dpf_error(code).decode()}")
        return out if derivative is None else (dx,dwgt,dpos)

    def apply(self, image, weights):
        return self.run(image,weights)

    def backward_positions(self, image, derivative, weights, *, input_gradient=True):
        return self.run(image,weights,derivative,input_gradient)
