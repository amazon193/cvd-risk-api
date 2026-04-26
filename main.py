import gdown
import os

# Google Drive file IDs (paste yours here)
UNET_ID       = "https://drive.google.com/file/d/1pEEdGBJBsC9IrTENSO1li75sdhPkpZHQ/view?usp=drive_link"
CLF_ID     = "https://drive.google.com/file/d/1Ffxs3HVyHpigYYo9cXd5nt8HTaVMz_3V/view?usp=drive_link"
SCALER_ID        = "https://drive.google.com/file/d/10BvriDIde-6M0_DgPDHX5MMc8KW6U-9z/view?usp=drive_link"

# Download models if not already present
if not os.path.exists('unet_model.h5'):
    gdown.download(
        f"https://drive.google.com/uc?id={UNET_ID}",
        'unet_model.h5', quiet=False)

if not os.path.exists('scaler.pkl'):
    gdown.download(
        f"https://drive.google.com/uc?id={SCALER_ID}",
        'scaler.pkl', quiet=False)

if not os.path.exists('classifier.pkl'):
    gdown.download(
        f"https://drive.google.com/uc?id={CLF_ID}",
        'classifier.pkl', quiet=False)
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
import base64
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import skeletonize
from skimage import measure
from scipy import ndimage
import tensorflow as tf

app = FastAPI()

# Load models at startup
unet   = tf.keras.models.load_model('unet_model.h5')
scaler = joblib.load('scaler.pkl')
clf    = joblib.load('classifier.pkl')

FEAT_COLS = [
    'avr','artery_width','vein_width',
    'avr_deviation','tortuosity_mean',
    'tortuosity_max','tortuosity_std',
    'fractal_dim','angle_mean','angle_std',
    'angle_min','angle_max','acute_ratio',
    'branch_count','vessel_density',
    'vessel_length','vessel_width'
]

@app.get("/")
def home():
    return {"status": "CVD Risk API running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    result = predict_cvd_risk(img_bytes)
    if 'result_image' in result:
        result['result_image'] = base64.b64encode(
            result['result_image']).decode('utf-8')
    return JSONResponse(content=result)

def predict_cvd_risk(img_bytes):

    # Decode image
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return {'error': 'Invalid image'}

    img_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess
    img_bgr  = cv2.resize(img_bgr, (256, 256))
    green    = img_bgr[:,:,1]
    g3ch     = cv2.merge([green, green, green])
    lab      = cv2.cvtColor(g3ch, cv2.COLOR_BGR2LAB)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img_pre  = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    inv_g    = 1.0/1.5
    table    = np.array([
        ((i/255.0)**inv_g)*255 for i in range(256)
    ]).astype('uint8')
    img_pre  = cv2.LUT(img_pre, table)
    img_pre  = img_pre.astype('float32')
    img_pre  = (img_pre - img_pre.min()) / (img_pre.max() - img_pre.min() + 1e-6)
    mean     = img_pre.mean()
    std      = img_pre.std() + 1e-6
    img_pre  = (img_pre - mean) / std
    img_pre  = img_pre.astype('float32')

    # UNet segmentation
    prob_map    = unet.predict(np.expand_dims(img_pre, 0), verbose=0)
    vessel_mask = (prob_map[0,:,:,0] > 0.5).astype(np.float32)
    mask_bool   = vessel_mask.astype(bool)

    # A/V separation
    img_u8  = np.clip(img_pre * 255, 0, 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB)
    R = img_rgb[:,:,0].astype(np.float32)
    G = img_rgb[:,:,1].astype(np.float32)
    B = img_rgb[:,:,2].astype(np.float32)
    rg    = R / (G + 1e-6)
    vote1 = (rg > np.median(rg[mask_bool])).astype(np.uint8)
    V     = (R + G + B) / 3.0
    vote2 = (V > np.median(V[mask_bool])).astype(np.uint8)
    r_norm = R / (R + G + B + 1e-6)
    vote3  = (r_norm > np.median(r_norm[mask_bool])).astype(np.uint8)
    vs     = vote1 + vote2 + vote3
    artery = np.zeros(vessel_mask.shape, dtype=np.float32)
    vein   = np.zeros(vessel_mask.shape, dtype=np.float32)
    artery[mask_bool & (vs >= 2)] = 1
    vein[mask_bool   & (vs <  2)] = 1

    # Feature extraction
    sk_a   = skeletonize(artery.astype(bool))
    sk_v   = skeletonize(vein.astype(bool))
    sk_all = skeletonize(mask_bool)

    a_area = float(np.sum(artery > 0))
    v_area = float(np.sum(vein > 0))
    a_len  = max(float(np.sum(sk_a)), 1.0)
    v_len  = max(float(np.sum(sk_v)), 1.0)
    aw     = a_area / a_len
    vw     = v_area / v_len
    avr    = aw / (vw + 1e-6)

    density = float(np.sum(mask_bool)) / vessel_mask.size
    length  = float(np.sum(sk_all))
    area    = float(np.sum(mask_bool))

    labeled   = measure.label(sk_all)
    props     = measure.regionprops(labeled)
    tort_vals = []
    for prop in props:
        if prop.area < 10: continue
        coords = prop.coords
        if len(coords) < 3: continue
        arc   = float(prop.area)
        s, e  = coords[0], coords[-1]
        chord = float(np.sqrt((e[0]-s[0])**2 + (e[1]-s[1])**2))
        if chord < 1.0: continue
        tort_vals.append(arc / chord)
    tort = float(np.mean(tort_vals)) if tort_vals else 0.0

    binary = (vessel_mask > 0).astype(np.uint8)
    if np.sum(binary) > 0:
        size   = max(binary.shape)
        n_pow  = int(np.ceil(np.log2(size)))
        padded = np.zeros((2**n_pow, 2**n_pow), dtype=np.uint8)
        padded[:binary.shape[0], :binary.shape[1]] = binary
        box_sizes = [2**i for i in range(1, n_pow)]
        counts    = []
        for bs in box_sizes:
            cnt = 0
            for r in range(0, padded.shape[0], bs):
                for c in range(0, padded.shape[1], bs):
                    if padded[r:r+bs, c:c+bs].any():
                        cnt += 1
            counts.append(cnt)
        valid = [(s, c) for s, c in zip(box_sizes, counts) if c > 0]
        if len(valid) >= 2:
            ls = np.log([1/s for s, c in valid])
            lc = np.log([c   for s, c in valid])
            fd = abs(np.polyfit(ls, lc, 1)[0])
        else:
            fd = 0.0
    else:
        fd = 0.0

    skel_int   = sk_all.astype(np.uint8)
    kernel     = np.ones((3,3), dtype=np.uint8)
    nbr_sum    = ndimage.convolve(skel_int, kernel)
    branch_pts = np.argwhere((nbr_sum >= 4) & sk_all)
    angles     = []
    TRACE      = 10
    for (by, bx) in branch_pts:
        neighbors = [
            (by+dy, bx+dx)
            for dy in [-1,0,1]
            for dx in [-1,0,1]
            if not (dy==0 and dx==0)
            and 0 <= by+dy < sk_all.shape[0]
            and 0 <= bx+dx < sk_all.shape[1]
            and sk_all[by+dy, bx+dx]
        ]
        if len(neighbors) < 2: continue
        vecs = []
        for (ny, nx) in neighbors[:3]:
            ey = np.clip(by+(ny-by)*TRACE, 0, sk_all.shape[0]-1)
            ex = np.clip(bx+(nx-bx)*TRACE, 0, sk_all.shape[1]-1)
            v  = np.array([ey-by, ex-bx], dtype=np.float32)
            nv = np.linalg.norm(v)
            if nv > 0: vecs.append(v/nv)
        for ii in range(len(vecs)):
            for jj in range(ii+1, len(vecs)):
                dot = np.clip(np.dot(vecs[ii], vecs[jj]), -1.0, 1.0)
                angles.append(np.degrees(np.arccos(dot)))

    angle = float(np.mean(angles)) if angles else 90.0
    bc    = len(branch_pts)

    # Build feature vector
    feat_dict = {
        'avr'             : avr,
        'artery_width'    : aw,
        'vein_width'      : vw,
        'avr_deviation'   : abs(avr - 0.71),
        'tortuosity_mean' : tort,
        'tortuosity_max'  : max(tort_vals) if tort_vals else 0.0,
        'tortuosity_std'  : float(np.std(tort_vals)) if tort_vals else 0.0,
        'fractal_dim'     : fd,
        'angle_mean'      : angle,
        'angle_std'       : float(np.std(angles)) if angles else 0.0,
        'angle_min'       : float(np.min(angles)) if angles else 0.0,
        'angle_max'       : float(np.max(angles)) if angles else 0.0,
        'acute_ratio'     : float(np.sum(np.array(angles)<45)/max(len(angles),1)) if angles else 0.0,
        'branch_count'    : float(bc),
        'vessel_density'  : density,
        'vessel_length'   : length,
        'vessel_width'    : area / max(length, 1.0),
    }

    feat_vec   = np.array([feat_dict.get(c, 0.0) for c in FEAT_COLS]).reshape(1,-1)
    feat_sc    = scaler.transform(feat_vec)
    pred_label = int(clf.predict(feat_sc)[0])
    pred_probs = clf.predict_proba(feat_sc)[0]

    full_probs = np.zeros(3, dtype=float)
    for idx, cls in enumerate(clf.classes_):
        if cls < 3:
            full_probs[cls] = pred_probs[idx]

    # Disease probabilities
    htn = 0.0
    if   avr < 0.67: htn += 0.40
    elif avr < 0.70: htn += 0.20
    elif avr > 0.75: htn += 0.15
    if tort > 1.2:   htn += 0.30
    htn_prob = min(htn, 1.0) * 100

    dm = 0.0
    if fd       < 1.5: dm += 0.35
    if density  < 0.05: dm += 0.30
    if bc       < 20:  dm += 0.25
    if angle < 50 or angle > 115: dm += 0.10
    dm_prob  = min(dm, 1.0) * 100

    cvd_prob = float(full_probs[2]) * 100
    n_risks  = sum([htn_prob>50, dm_prob>50, cvd_prob>40])
    if n_risks >= 2:
        cvd_prob = min(cvd_prob + 15, 100)

    risk_score = (full_probs[0]*10 + full_probs[1]*50 + full_probs[2]*100)
    conf       = full_probs[pred_label] * 100

    # Determine status
    if   avr < 0.67: avr_s = 'Narrowing'
    elif avr > 0.75: avr_s = 'Dilation'
    else:            avr_s = 'Normal'

    avr_ok  = 0.67 <= avr <= 0.75
    tort_ok = tort < 1.2
    fd_ok   = 1.5 <= fd <= 1.9
    ang_ok  = 45 <= angle <= 120
    den_ok  = density > 0.03
    bc_ok   = bc > 10
    all_ok  = all([avr_ok, tort_ok, fd_ok, ang_ok, den_ok, bc_ok, pred_label==0])

    if all_ok:
        status_label = 'HEALTHY PERSON'
        status_color = '#1a8f3c'
        status_line2 = 'All retinal vessel parameters are within normal range.'
        advice       = 'No CVD risk detected. Continue routine check-ups and healthy lifestyle.'
        is_healthy   = True
    elif pred_label == 0:
        status_label = 'LOW CVD RISK'
        status_color = '#27ae60'
        status_line2 = 'Mild vessel changes. Low probability of CVD.'
        advice       = 'Regular monitoring advised. Maintain healthy lifestyle.'
        is_healthy   = False
    elif pred_label == 1:
        status_label = 'MEDIUM CVD RISK'
        status_color = '#e67e22'
        status_line2 = 'Moderate vessel abnormalities.'
        advice       = 'Consult a physician. Regular BP and glucose monitoring recommended.'
        is_healthy   = False
    else:
        status_label = 'HIGH CVD RISK'
        status_color = '#c0392b'
        status_line2 = 'Significant vessel abnormalities detected.'
        advice       = 'Immediate cardiologist consultation required.'
        is_healthy   = False

    # Generate result image
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    img_d         = np.clip(img_pre, 0, 1)
    av_map        = np.zeros((*vessel_mask.shape, 3), dtype=np.float32)
    av_map[:,:,0] = artery
    av_map[:,:,2] = vein

    panels = [
        (img_orig,    'Original'),
        (img_d,       'Preprocessed'),
        (vessel_mask, 'Vessel Mask'),
        (av_map,      'A/V Map'),
        (artery,      'Arteries'),
        (vein,        'Veins'),
    ]
    cmaps = [None, None, 'gray', None, 'Reds', 'Blues']

    for idx, (panel, title, cmap_) in enumerate(
            zip([p for p,_ in panels], [t for _,t in panels], cmaps)):
        r  = idx // 4
        c_ = idx % 4
        ax = axes[r][c_]
        ax.imshow(panel, cmap=cmap_)
        ax.set_title(title, fontweight='bold', fontsize=9)
        ax.axis('off')

    ax_risk = axes[1][2]
    ax_risk.barh(
        ['Low','Med','High'],
        [full_probs[0]*100, full_probs[1]*100, full_probs[2]*100],
        color=['#27ae60','#e67e22','#e74c3c'],
        edgecolor='black', alpha=0.8)
    ax_risk.set_xlim(0, 110)
    ax_risk.set_title('Risk Breakdown', fontweight='bold', fontsize=9)
    ax_risk.grid(True, alpha=0.3, axis='x')

    ax_sum = axes[1][3]
    ax_sum.axis('off')
    ax_sum.set_facecolor('#d5f5e3' if is_healthy else '#fadbd8')
    summary = (
        "Status: {}\nRisk Score: {:.1f}/100\nConf: {:.1f}%\n\n"
        "HTN: {:.1f}%\nDM : {:.1f}%\nCVD: {:.1f}%"
    ).format(status_label, risk_score, conf, htn_prob, dm_prob, cvd_prob)
    ax_sum.text(0.1, 0.9, summary,
        transform=ax_sum.transAxes, fontsize=9, va='top',
        fontfamily='monospace', color=status_color, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                  edgecolor=status_color, linewidth=2))

    plt.suptitle("Result: {} | Score: {:.1f}/100".format(status_label, risk_score),
                 fontsize=12, fontweight='bold', color=status_color)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_bytes_out = buf.read()

    return {
        'status_label' : status_label,
        'status_color' : status_color,
        'status_line2' : status_line2,
        'advice'       : advice,
        'is_healthy'   : is_healthy,
        'risk_score'   : round(risk_score, 1),
        'confidence'   : round(conf,       1),
        'htn_prob'     : round(htn_prob,   1),
        'dm_prob'      : round(dm_prob,    1),
        'cvd_prob'     : round(cvd_prob,   1),
        'low_prob'     : round(full_probs[0]*100, 1),
        'med_prob'     : round(full_probs[1]*100, 1),
        'high_prob'    : round(full_probs[2]*100, 1),
        'features'     : {
            'avr'        : round(avr,     4),
            'avr_status' : avr_s,
            'tortuosity' : round(tort,    4),
            'fractal_dim': round(fd,      4),
            'angle'      : round(angle,   2),
            'density'    : round(density, 6),
            'branches'   : int(bc),
        },
        'result_image' : img_bytes_out
    }
