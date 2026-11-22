"""Local Bloom target preview and spatial DPF fit. Set BLOOM_DPF_CUDA_LIBRARY to the compiled bloom_dpf_filter library for CUDA fitting."""
import argparse
import base64
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit
import zipfile

from bloom_dpf_fit import coarse_target, shader_coefficients
from bloom_dpf_policy import fit_spatial_recipe
from bloom_dpf_target import PRESETS, Recipe, reference_tail, resize_mass, target_kernel
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


PAGE = Path(__file__).with_name("bloom_dpf.html")
PREVIEW_RADIUS = 160
DISPLAY_FLOOR = 2e-6
DISPLAY_WHITE = .012
VIEWS = (160, 320, 768, 1536)


def preview_radius(recipe):
    return math.ceil(PREVIEW_RADIUS * max(1, recipe.size))


def idle_worker():
    if sys.platform == "win32":
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentThread.restype = ctypes.c_void_p
        kernel.SetThreadPriority.argtypes = (ctypes.c_void_p, ctypes.c_int)
        if not kernel.SetThreadPriority(kernel.GetCurrentThread(), -15):
            raise OSError("Could not set the fit thread to Idle priority")


def png_uri(rgb, ev, logarithmic):
    value = np.maximum(rgb, 0) * (2**ev)
    if logarithmic:
        value = np.log1p(value / DISPLAY_FLOOR) / np.log1p(DISPLAY_WHITE / DISPLAY_FLOOR)
    else:
        value = value / DISPLAY_WHITE
    pixels = np.rint(np.clip(value, 0, 1).transpose(1, 2, 0)*255).astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getbuffer()).decode("ascii")


def preview_payload(kernel, ev, radius, logarithmic, *, hdr=False):
    if not -8 <= ev <= 8:
        raise ValueError("Preview EV must be in [-8, 8]")
    p = kernel.shape[-1] // 2
    # Same kernel and exposure for white, warm, and cyan emitters.
    r = min(PREVIEW_RADIUS, 62)
    crop = kernel[:,p-r:p+r+1,p-r:p+r+1]
    lights = np.array([[1,1,1],[1,.44,.11],[.12,.74,1]],dtype=np.float32)
    strip = np.concatenate([crop*rgb[:,None,None] for rgb in lights],axis=2)
    payload = {"image":png_uri(kernel,ev,logarithmic),"lights":png_uri(strip,ev,logarithmic),"radius":radius}
    if hdr:
        point = resize_mass(kernel,PREVIEW_RADIUS//2,.5) * .25
        strip = np.concatenate([point,point*4,point*16],axis=2)
        payload["hdr"] = png_uri(strip,ev,logarithmic)
    return payload


def view_fitted(response, phase, radius):
    step = radius / PREVIEW_RADIUS
    coordinate = np.arange(-PREVIEW_RADIUS,PREVIEW_RADIUS+1) * step
    fine, pooled = response["fine"], response["pooled"]
    indices = range(len(fine)) if phase == "mean" else (phase,)
    fine_limit = fine.shape[-1] // 2 + .5 - step/2
    interior = np.maximum(np.abs(coordinate[:,None]),np.abs(coordinate)) <= fine_limit
    images = []
    for i in indices:
        x,y = response["phases"][i]
        pool = response["pool"]
        origin = response["side"]//2 - (pool-1)*.5
        xx,yy = np.meshgrid((coordinate+origin+x)/pool,(coordinate+origin+y)/pool)
        coarse = np.stack([map_coordinates(channel,[yy,xx],order=1,mode="constant",cval=0.) for channel in pooled[i]])
        detailed = resize_mass(fine[i],PREVIEW_RADIUS,1/step) / np.float32(step**2)
        coarse[:,interior] = detailed[:,interior]
        images.append(coarse)
    return np.mean(images,axis=0)


@lru_cache(maxsize=2)
def profile_target(recipe, fine_radius, outer_radius, side, phases, pool):
    complete = target_kernel(recipe, outer_radius)
    pooled, _ = coarse_target(complete, side, phases, pool, fine_radius)
    fine = target_kernel(recipe, fine_radius)
    return dict(fine=np.repeat(fine[None], len(phases), axis=0), pooled=pooled,
                side=side, phases=phases, pool=pool)


def profile_views(recipe, completed, phase, radius):
    if completed is None:
        return target_kernel(recipe, PREVIEW_RADIUS, radius/PREVIEW_RADIUS), None, None
    result, response = completed
    if phase != "mean" and (type(phase) is not int or not 0 <= phase < len(response["fine"])):
        raise ValueError("Unknown validation phase")
    target = profile_target(recipe, response["fine"].shape[-1]//2,
                            result["outer_radius"], response["side"],
                            tuple(map(tuple,response["phases"])), response["pool"])
    return view_fitted(target,phase,radius), view_fitted(response,phase,radius), response


def line_profile(recipe, completed, phase, radius, endpoints):
    ends = np.asarray(endpoints,dtype=np.float64)
    if ends.shape != (2,2) or not np.isfinite(ends).all() or np.any(np.abs(ends)>radius):
        raise ValueError("Expected two endpoints within the preview in full-resolution pixels")
    target, fitted, response = profile_views(recipe,completed,phase,radius)
    step = radius/PREVIEW_RADIUS
    length = float(np.linalg.norm(ends[1]-ends[0]))
    distance = np.linspace(0,length,max(2,math.ceil(length/step)+1))
    points = ends[0]+np.linspace(0,1,len(distance))[:,None]*(ends[1]-ends[0])
    coordinates = (points/step+PREVIEW_RADIUS).T[::-1]
    def samples(image):
        rgb = np.stack([map_coordinates(channel,coordinates,order=1,mode="nearest") for channel in image],axis=1)
        return dict(rgb=rgb.tolist(),luminance=(rgb @ np.array([.2126,.7152,.0722])).tolist())
    footprint = np.full(len(distance),step)
    if response is not None:
        fine_limit = response["fine"].shape[-1]//2+.5-step/2
        footprint[np.ceil(np.max(np.abs(points),axis=1)/step)*step>fine_limit] = response["pool"]
    return dict(recipe_sha256=recipe.digest(),endpoints=ends.tolist(),phase=phase,
                distance=distance.tolist(),target=samples(target),
                fitted=None if fitted is None else samples(fitted),
                pixel_spacing=step,footprint=footprint.tolist(),pool=None if response is None else response["pool"])


class FitSlot:
    def __init__(self, filter_class=None):
        if filter_class is None and os.environ.get("BLOOM_DPF_CUDA_LIBRARY"):
            from bloom_dpf_cuda import Filter
            filter_class = Filter
        self.filter_class = filter_class
        self.lock = threading.Lock()
        self.state = {"status":"empty"}
        self.result = None
        self.cancelled = None

    def start(self, recipe):
        with self.lock:
            if self.state["status"] == "running":
                raise ValueError("A fit is already active")
            cancel = threading.Event()
            self.cancelled = cancel
            self.result = None
            self.state = {"status":"running","recipe_sha256":recipe.digest(),"iteration":0}
        threading.Thread(target=self.fit,args=(recipe,cancel),daemon=True).start()

    def fit(self, recipe, cancel):
        def progress(iteration, loss):
            with self.lock:
                self.state.update(iteration=iteration,loss=loss)
        try:
            idle_worker()
            result, images = fit_spatial_recipe(recipe,radius=preview_radius(recipe),
                                        outer_radius=math.ceil(768*max(1,recipe.size)),
                                        progress=progress,cancelled=cancel.is_set,filter_class=self.filter_class)
        except InterruptedError:
            with self.lock:
                self.state["status"] = "cancelled"
        except Exception as error:
            with self.lock:
                self.state.update(status="failed",error=str(error))
        else:
            with self.lock:
                if cancel.is_set():
                    self.state["status"] = "cancelled"
                else:
                    self.result = (result, images)
                    self.state.update(status="finished",phases=result["holdout_phases"],
                                      holdout=result["holdout"],solver=result["solver"],loss=result["loss"],
                                      converged=result["converged"],iteration_limit=result["iteration_limit"],
                                      budget=result["weighted_bilinear_fetches_per_full_res_pixel"])

    def cancel(self):
        with self.lock:
            if self.state["status"] == "running":
                self.cancelled.set()

    def status(self):
        with self.lock:
            return dict(self.state)

    def completed(self, recipe):
        result = self.matching(recipe)
        if result is None:
            raise ValueError("The completed fit belongs to a different recipe")
        return result

    def matching(self, recipe):
        with self.lock:
            if not self.result or self.state["recipe_sha256"] != recipe.digest():
                return None
            return self.result


class Handler(BaseHTTPRequestHandler):
    fit = FitSlot()

    def answer(self, content, content_type, *, status=200, filename=None):
        self.send_response(status)
        self.send_header("Content-Type",content_type)
        self.send_header("Content-Length",str(len(content)))
        self.send_header("Cache-Control","no-store")
        if filename:
            self.send_header("Content-Disposition",f'attachment; filename="{filename}"')
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def json(self, value, status=200):
        self.answer(json.dumps(value,allow_nan=False).encode(),"application/json",status=status)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            self.answer(PAGE.read_bytes(),"text/html; charset=utf-8")
        elif path == "/api/presets":
            self.json({name:Recipe(name).resolved() for name in PRESETS})
        elif path == "/api/state":
            self.json(self.fit.status())
        else:
            self.json({"error":"Unknown endpoint"},404)

    def do_POST(self):
        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Expected JSON input")
            length = int(self.headers.get("Content-Length","0"))
            if not 0 < length <= 65536:
                raise ValueError("Expected a small recipe request")
            body = json.loads(self.rfile.read(length))
            path = urlsplit(self.path).path
            if path == "/api/cancel":
                self.fit.cancel()
                self.json(self.fit.status())
                return
            recipe = Recipe(**body["recipe"])
            if path in ("/api/target","/api/fitted","/api/profile"):
                radius = body.get("radius",160)
                if radius not in VIEWS:
                    raise ValueError("Unknown preview range")
                logarithmic = body.get("display","log") == "log"
                phase = body.get("phase","mean")
                if path == "/api/profile":
                    self.json(line_profile(recipe,self.fit.matching(recipe),phase,radius,body["endpoints"]))
                    return
                if path == "/api/target":
                    kernel,_,_ = profile_views(recipe,self.fit.matching(recipe),phase,radius)
                else:
                    _, response = self.fit.completed(recipe)
                    phase = body["phase"]
                    if phase != "mean" and (type(phase) is not int or not 0 <= phase < len(response["fine"])):
                        raise ValueError("Unknown validation phase")
                    kernel = view_fitted(response,phase,radius)
                self.json({"recipe_sha256":recipe.digest(),
                           **preview_payload(kernel,float(body["ev"]),radius,logarithmic,
                                             hdr=path == "/api/target" and recipe.lobes is not None)})
            elif path == "/api/fit":
                self.fit.start(recipe)
                self.json(self.fit.status(),202)
            elif path == "/api/export":
                result, _ = self.fit.completed(recipe)
                artifact = io.BytesIO()
                with zipfile.ZipFile(artifact,"w",compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr("recipe.json",json.dumps(recipe.resolved(),indent=2) + "\n")
                    archive.writestr("fit.json",json.dumps(result,indent=2,allow_nan=False) + "\n")
                    archive.writestr("shaders/techniques/bloom/BloomDpfCustom.glsl",shader_coefficients(recipe,result))
                self.answer(artifact.getvalue(),"application/zip",filename="bloom-dpf.zip")
            else:
                self.json({"error":"Unknown endpoint"},404)
        except (TypeError, ValueError, KeyError) as error:
            self.json({"error":str(error)},400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,default=8765)
    args = parser.parse_args()
    reference_tail()
    with ThreadingHTTPServer(("127.0.0.1",args.port),Handler) as server:
        print(f"Bloom DPF preview: http://127.0.0.1:{server.server_port}/",flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            Handler.fit.cancel()


if __name__ == "__main__":
    main()
