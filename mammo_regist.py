import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import cv2
from tqdm import tqdm
import SimpleITK as sitk
from scipy.optimize import minimize

def ecc_score(fixed, moving, sx, sy, tx, ty):
    M = np.array([[sx, 0,  tx],
                  [0,  sy, ty]], dtype=np.float32)
    warped = cv2.warpAffine(moving, M, (fixed.shape[1], fixed.shape[0]),
                            flags=cv2.INTER_LINEAR)
    # Normalize
    f = (fixed - fixed.mean()); m = (warped - warped.mean())
    denom = np.linalg.norm(f) * np.linalg.norm(m) + 1e-12
    return (f*m).sum() / denom  # ECC to maximize

def register_scale_translation(fixed, moving, init=(1.0,1.0,0.0,0.0)):
    def neg_ecc(p): return -ecc_score(fixed, moving, *p)
    res = minimize(neg_ecc, init, method="L-BFGS-B",
                   bounds=[(0.5,2.0),(0.5,2.0),(-200,200),(-200,200)])
    sx, sy, tx, ty = res.x
    M = np.array([[sx,0,tx],[0,sy,ty]], np.float32)
    out = cv2.warpAffine(moving, M, (fixed.shape[1], fixed.shape[0]),
                         flags=cv2.INTER_LINEAR)
    return M, out

def pil_to_sitk(img_pil, force_grayscale=True, pixel_type=sitk.sitkFloat32):
    if force_grayscale and img_pil.mode not in ("L",):
        img_pil = img_pil.convert("L")       # drop color/alpha for rigid mono
    arr = np.array(img_pil)                  # shape [H, W] or [H, W, C]
    if arr.ndim == 3:                        # if you kept color, make it a vector image
        img_sitk = sitk.GetImageFromArray(arr, isVector=True)
    else:
        img_sitk = sitk.GetImageFromArray(arr)
    return sitk.Cast(img_sitk, pixel_type)

def sitk_to_pil(img_sitk, to_uint8=True):
    arr = sitk.GetArrayFromImage(img_sitk)       # shape [H, W] for 2D
    if to_uint8:
        # Rescale to [0,255] for display (avoid clipping scientific ranges)
        arr = arr.astype(np.float32)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (255.0 * (arr - mn) / (mx - mn)).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
        return Image.fromarray(arr, mode="L")
    else:
        # If you want to keep float, just skip scaling and handle elsewhere
        return Image.fromarray(arr)

def sitk_rigid_registration(fixed_img_sitk, moving_img_sitk):
    if isinstance(fixed_img_sitk, Image.Image):
        fixed_sitk = pil_to_sitk(fixed_img_sitk)
    if isinstance(moving_img_sitk, Image.Image):
        moving_sitk = pil_to_sitk(moving_img_sitk)
    
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(32)          # good for mono or multi-modal
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-6, numberOfIterations=200)
    # Initialize around image centers (or use MOMENTS)
    init_tx = sitk.CenteredTransformInitializer(
        fixed_sitk, moving_sitk, sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R.SetInitialTransform(init_tx, inPlace=False)

    # --- 3) Execute + resample moving into fixed space ---
    tx = R.Execute(fixed_sitk, moving_sitk)
    resampled = sitk.Resample(
        moving_sitk, fixed_sitk, tx, sitk.sitkLinear, 0.0, moving_sitk.GetPixelID())
    
    return sitk_to_pil(resampled)