#!/usr/bin/env python3
import os
import sys
import cv2
import torch
import argparse
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from skimage import morphology
from torchvision.transforms import Compose
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from numpy.linalg import lstsq
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Add MiDaS to import path
sys.path.append("C:/Users/alexf/software-projects/clippd/midas")
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR     = Path("../data")
FRAME_DIR    = DATA_DIR / "frames"
DEPTH_DIR    = DATA_DIR / "depth"
EDGE_DIR     = DATA_DIR / "edges"
EMBOSS_DIR   = DATA_DIR / "emboss"
LAYERED_DIR  = DATA_DIR / "layered"
MULTI_DIR    = DATA_DIR / "multiscale"
TRACKED_DIR  = DATA_DIR / "tracked"
TERRAIN_DIR  = DATA_DIR / "terrain"
CAMERA_DIR   = DATA_DIR / "camera"
EXPORT_DIR   = DATA_DIR / "export"

VIDEO_PATH   = DATA_DIR / "input_clip.mp4"
MODEL_PATH   = Path("../weights/dpt_large_384.pt")
SAM_CKPT     = Path("../weights/sam2.1_hiera_large.pt")
SAM_CFG      = Path("C:/Users/alexf/software-projects/clippd/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LAYERS   = 5
BATCH_SIZE   = 4
MULTI_SCALES = [0.5, 1.0, 1.5]
FRAME_LIMIT  = 200   # cap for quick testing

# ─── STEP 1: Extract frames ───────────────────────────────────────────────
def extract_frames():
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    idx = 0
    while idx < FRAME_LIMIT:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(FRAME_DIR / f"frame_{idx:04d}.png"), frame)
        idx += 1
    cap.release()
    print(f"[frames] extracted {idx} frames (capped at {FRAME_LIMIT}).")

# ─── STEP 2: Depth estimation ─────────────────────────────────────────────
def load_midas():
    model = DPTDepthModel(backbone="vitl16_384", non_negative=True)
    ckpt = torch.load(str(MODEL_PATH), map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval().to(DEVICE)
    tf = Compose([
        Resize(384,384),
        NormalizeImage(mean=[0.5]*3, std=[0.5]*3),
        PrepareForNet()
    ])
    return model, tf

def estimate_depth():
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    midas, tf = load_midas()
    frames = sorted(FRAME_DIR.glob("*.png"))[:FRAME_LIMIT]
    for i in range(0, len(frames), BATCH_SIZE):
        batch = frames[i:i+BATCH_SIZE]
        imgs, shapes = [], []
        for p in batch:
            im = cv2.imread(str(p))[...,::-1]/255.0
            imgs.append(tf({"image":im, "mask":np.ones(im.shape[:2])})["image"])
            shapes.append(im.shape[:2])
        inp = torch.from_numpy(np.stack(imgs)).to(DEVICE)
        with torch.no_grad():
            preds = midas(inp)
        for pred, pth, shp in zip(preds, batch, shapes):
            disp = torch.nn.functional.interpolate(
                pred.unsqueeze(0).unsqueeze(0),
                size=shp, mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()
            plt.imsave(str(DEPTH_DIR / pth.name), disp, cmap="plasma")
    print(f"[depth] done for {len(frames)} frames.")

# ─── STEP 3: Edge detection ───────────────────────────────────────────────
def detect_edges():
    EDGE_DIR.mkdir(parents=True, exist_ok=True)
    for p in sorted(FRAME_DIR.glob("*.png"))[:FRAME_LIMIT]:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(im, 100, 200)
        cv2.imwrite(str(EDGE_DIR / p.name), edges)
    print(f"[edges] done for {FRAME_LIMIT} frames.")

# ─── STEP 4: Temporal emboss ──────────────────────────────────────────────
def emboss_frames():
    EMBOSS_DIR.mkdir(parents=True, exist_ok=True)
    kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
    for p in sorted(FRAME_DIR.glob("*.png"))[:FRAME_LIMIT]:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        emb = cv2.filter2D(im, -1, kernel) + 128
        cv2.imwrite(str(EMBOSS_DIR / p.name), emb)
    print(f"[emboss] done for {FRAME_LIMIT} frames.")

# ─── UTILS: SAM2 + MaskRCNN ──────────────────────────────────────────────
def load_sam_model():
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(SAM_CFG.parent), version_base=None)
    compose(config_name=SAM_CFG.name)
    sam = build_sam2(config_file=str(SAM_CFG), ckpt_path=str(SAM_CKPT), device=DEVICE)
    return SAM2ImagePredictor(sam)

def load_inst_model():
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.eval().to(DEVICE)
    return model

def segment_by_depth(dmap):
    norm = (dmap - dmap.min())/(dmap.max()-dmap.min()+1e-8)
    masks, layers = [], np.zeros_like(dmap, dtype=np.uint8)
    thr = np.linspace(0,1,NUM_LAYERS+1)
    for i in range(NUM_LAYERS):
        m = ((norm>=thr[i])&(norm<thr[i+1])).astype(np.uint8)
        if m.sum()>0:
            m = morphology.binary_closing(m, morphology.disk(3))
        masks.append(m)
        layers[m>0] = i+1
    return masks, layers

# ─── STEP 5: Layered segmentation ─────────────────────────────────────────
def layered_segmentation():
    LAYERED_DIR.mkdir(parents=True, exist_ok=True)
    inst = load_inst_model()
    sam  = load_sam_model()
    for p in sorted(FRAME_DIR.glob("*.png"))[:FRAME_LIMIT]:
        # Use .copy() after slice reversal to ensure positive strides
        rgb = cv2.imread(str(p))[...,::-1].copy()
        dmap = plt.imread(str(DEPTH_DIR/p.name))
        dmap = dmap[...,0] if dmap.ndim>2 else dmap
        masks,_ = segment_by_depth(dmap)

        vis = rgb.copy()
        for lm in masks:
            # prepare input for MaskRCNN as float32
            t = (rgb / 255.0).astype(np.float32)
            t = torch.from_numpy(t).permute(2,0,1).to(DEVICE)
            with torch.no_grad():
                pred = inst([t])[0]

            inst_ms = [
                (m[0].cpu().numpy()>0.5).astype(np.uint8)
                for m,s in zip(pred["masks"], pred["scores"]) if s>0.7
            ]
            chosen = [m for m in inst_ms if (m&lm).sum()/m.sum()>0.5]

            # fallback to SAM if too few
            if len(chosen)<2:
                # Fix: Set the image before predicting
                # Use .copy() to ensure there are no negative strides
                rgb_copy = rgb.copy()
                sam.set_image(rgb_copy)
                sms,_,_ = sam.predict(multimask_output=True)
                for m in sms:
                    bm = (m>0.5).astype(np.uint8)
                    if bm.sum()>100 and ((bm&lm).sum()/bm.sum())>0.1:
                        chosen.append(bm)

            for m in chosen:
                cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, (255,0,0), 1)

        cv2.imwrite(str(LAYERED_DIR / p.name), vis[...,::-1])

        # Save masks for tracking
        np.save(str(LAYERED_DIR / f"{p.stem}_masks.npy"), chosen)

    print(f"[layered] done for {FRAME_LIMIT} frames.")

# ─── STEP 6: Multi-scale segmentation ────────────────────────────────────
def multiscale_segmentation():
    MULTI_DIR.mkdir(parents=True, exist_ok=True)
    inst = load_inst_model()
    sam  = load_sam_model()
    for p in sorted(FRAME_DIR.glob("*.png"))[:FRAME_LIMIT]:
        # Use .copy() after slice reversal to ensure positive strides
        rgb0 = cv2.imread(str(p))[...,::-1].copy()
        d0   = plt.imread(str(DEPTH_DIR/p.name))
        d0   = d0[...,0] if d0.ndim>2 else d0
        H,W  = rgb0.shape[:2]
        fused = rgb0.copy()

        for scale in MULTI_SCALES:
            h,w = int(H*scale), int(W*scale)
            lmasks,_ = segment_by_depth(cv2.resize(d0,(w,h)))

            # upsample and segment at full res
            for lm in lmasks:
                lm_up = cv2.resize(lm,(W,H),interpolation=cv2.INTER_NEAREST)

                t = (rgb0/255.0).astype(np.float32)
                t = torch.from_numpy(t).permute(2,0,1).to(DEVICE)
                with torch.no_grad():
                    pred = inst([t])[0]

                inst_ms = [
                    (m[0].cpu().numpy()>0.5).astype(np.uint8)
                    for m,s in zip(pred["masks"],pred["scores"]) if s>0.7
                ]
                chosen = [m for m in inst_ms if (m&lm_up).sum()/m.sum()>0.5]

                if len(chosen)<2:
                    # Fix: Set the image before predicting
                    # Use .copy() to ensure there are no negative strides
                    rgb0_copy = rgb0.copy()
                    sam.set_image(rgb0_copy)
                    sms,_,_ = sam.predict(multimask_output=True)
                    for m in sms:
                        bm = (m>0.5).astype(np.uint8)
                        if bm.sum()>100 and ((bm&lm_up).sum()/bm.sum())>0.1:
                            chosen.append(bm)

                for m in chosen:
                    cnts,_ = cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(fused, cnts, -1, (0,255,0),1)

        cv2.imwrite(str(MULTI_DIR / p.name), fused[...,::-1])
    print(f"[multiscale] done for {FRAME_LIMIT} frames.")

# ─── STEP 7: Simple IoU tracking ─────────────────────────────────────────
def track_objects():
    TRACKED_DIR.mkdir(parents=True, exist_ok=True)
    tracker,next_id = {}, 1

    def iou(a,b):
        xi = (a&b).sum(); xu = (a|b).sum()
        return xi/xu if xu else 0

    for p in sorted(LAYERED_DIR.glob("*.png"))[:FRAME_LIMIT]:
        # Check if masks file exists
        mask_file = LAYERED_DIR/f"{p.stem}_masks.npy"
        if not mask_file.exists():
            print(f"Warning: Mask file {mask_file} not found. Skipping frame.")
            continue

        masks = np.load(str(mask_file), allow_pickle=True)
        updated=[]
        for m in masks:
            assigned=False
            for oid,prev in tracker.items():
                if iou(m,prev)>0.3:
                    tracker[oid]=m; updated.append((m,oid)); assigned=True; break
            if not assigned:
                tracker[next_id]=m; updated.append((m,next_id)); next_id+=1

        rgb = cv2.imread(str(p))
        for m,oid in updated:
            cnts,_ = cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb,cnts,-1,(0,255,0),1)
            y,x = np.where(m)
            if len(y) > 0 and len(x) > 0:
                cv2.putText(rgb,str(oid),(int(x.mean()),int(y.mean())),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

        cv2.imwrite(str(TRACKED_DIR/p.name), rgb)
    print(f"[tracked] done for {FRAME_LIMIT} frames.")

# ─── STEP 8: Terrain overlay ──────────────────────────────────────────────
def terrain_overlay():
    TERRAIN_DIR.mkdir(parents=True, exist_ok=True)
    for p in sorted(FRAME_DIR.glob("*.png"))[:FRAME_LIMIT]:
        rgb = cv2.imread(str(p))
        dmap= plt.imread(str(DEPTH_DIR/p.name))
        dmap= dmap[...,0] if dmap.ndim>2 else dmap
        out = rgb.copy()
        h,w = dmap.shape; mn,mx = dmap.min(), dmap.max()
        norm=lambda z:(z-mn)/(mx-mn+1e-8)

        for y in range(0,h,20):
            c=tuple(int(255*v) for v in plt.cm.terrain(norm(dmap[y,:].mean()))[:3])
            cv2.line(out,(0,y),(w,y),c,1)
        for x in range(0,w,20):
            c=tuple(int(255*v) for v in plt.cm.terrain(norm(dmap[:,x].mean()))[:3])
            cv2.line(out,(x,0),(x,h),c,1)

        Y,X = np.mgrid[:h,:w]
        A = np.stack([X.flatten(),Y.flatten(),np.ones(h*w)],1)
        b = dmap.flatten()
        a,b0,c0 = lstsq(A,b,rcond=None)[0]
        Zp=(a*X + b0*Y + c0)
        mask = np.abs(Zp-dmap)<np.percentile(np.abs(Zp-dmap),10)
        out[mask]=(0,255,0)

        cv2.imwrite(str(TERRAIN_DIR/p.name), out)
    print(f"[terrain] done for {FRAME_LIMIT} frames.")

# ─── STEP 9: Camera intrinsics overlay ────────────────────────────────────
def camera_overlay():
    CAMERA_DIR.mkdir(parents=True, exist_ok=True)
    for p in sorted(FRAME_DIR.glob("*.png"))[:FRAME_LIMIT]:
        im = cv2.imread(str(p))
        h,w = im.shape[:2]; cx,cy=w//2,h//2; fx=fy=w
        cv2.line(im,(cx,0),(cx,h),(0,255,255),1)
        cv2.line(im,(0,cy),(w,cy),(0,255,255),1)
        cv2.putText(im,f"fx={fx},fy={fy},cx={cx},cy={cy}",
                    (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
        cv2.imwrite(str(CAMERA_DIR/p.name), im)
    print(f"[camera] done for {FRAME_LIMIT} frames.")

# ─── STEP 10: SLAM reconstruction ─────────────────────────────────────────
def slam_reconstruct():
    orbb = cv2.ORB_create(5000)
    bf   = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    frames = sorted(FRAME_DIR.glob("*.png"))[:FRAME_LIMIT]
    projs, pts3d, prev_kp, prev_des = [], [], None, None

    for i,p in enumerate(frames):
        img = cv2.imread(str(p)); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kp,des = orbb.detectAndCompute(gray,None)
        if i==0:
            projs.append(np.hstack((np.eye(3),np.zeros((3,1)))))
            prev_kp,prev_des=kp,des
            continue

        m=bf.match(prev_des,des)
        m=sorted(m,key=lambda x:x.distance)[:200]
        if len(m)<8: continue

        pts0=np.float32([prev_kp[x.queryIdx].pt for x in m]).reshape(-1,2)
        pts1=np.float32([kp[x.trainIdx].pt   for x in m]).reshape(-1,2)
        E,_=cv2.findEssentialMat(pts1,pts0,focal=1,pp=(0,0),
                                 method=cv2.RANSAC,prob=0.999,threshold=1.0)
        if E is None: continue

        _,R,t,_=cv2.recoverPose(E,pts1,pts0)
        projs.append(np.hstack((R,t)))
        P0,P1=projs[-2],projs[-1]

        pts4d = cv2.triangulatePoints(P0,P1,pts0.T,pts1.T)
        pts3d.append((pts4d[:3]/pts4d[3]).T)
        prev_kp,prev_des=kp,des

    if pts3d:
        pts = np.vstack(pts3d)
        fig=plt.figure(); ax=fig.add_subplot(111,projection='3d')
        ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=1,c='blue')
        TRACKED_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(TRACKED_DIR/"slam_point_cloud.png"))
    print(f"[slam] done.")

# ─── STEP 11: Export ─────────────────────────────────────────────────────
def export_videos():
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found in PATH—skipping export.")
        return

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    mapping = {
        "depth":     DEPTH_DIR,
        "edges":     EDGE_DIR,
        "emboss":    EMBOSS_DIR,
        "layered":   LAYERED_DIR,
        "multiscale":MULTI_DIR,
        "terrain":   TERRAIN_DIR,
        "camera":    CAMERA_DIR,
    }

    for name, dpath in mapping.items():
        inp  = str(dpath / "frame_%04d.png")

        # Check if directory exists and has any frames before trying to export
        if not dpath.exists() or not list(dpath.glob("frame_*.png")):
            print(f"[export] Skipping {name} as directory is empty or doesn't exist")
            continue

        outp = str(EXPORT_DIR / f"{name}.mp4")
        print(f"[export] {name} → {outp}")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-framerate", "30",
                "-i", inp,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                outp
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[export] Error exporting {name}: {e}")

    print("[export] done.")

# ─── MAIN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cinematic pipeline")
    parser.add_argument(
        "--step", required=True,
        choices=[
            "frames","depth","edges","emboss","layered","multiscale",
            "tracked","terrain","camera","slam","export"
        ]
    )
    args = parser.parse_args()
    {
        "frames":    extract_frames,
        "depth":     estimate_depth,
        "edges":     detect_edges,
        "emboss":    emboss_frames,
        "layered":   layered_segmentation,
        "multiscale":multiscale_segmentation,
        "tracked":   track_objects,
        "terrain":   terrain_overlay,
        "camera":    camera_overlay,
        "slam":      slam_reconstruct,
        "export":    export_videos
    }[args.step]()
