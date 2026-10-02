
r"""
Synthetic CASME II dataset generator.
Creates realistic fake face patch clips and a matching Excel label file.
This lets you test the FULL HFA training pipeline without the real dataset.
Replace with real CASME II data once you get academic access.

Run this ONCE:
  cd C:\Users\shrin\OneDrive\Desktop
  python -m mmbi.models.generate_synthetic_data
"""

import os
import numpy as np
import cv2
import pandas as pd

# ── Output path — mirrors real CASME II structure ───────────────────────────
OUTPUT_ROOT  = r"C:\datasets\casme2"
CROPPED_DIR  = os.path.join(OUTPUT_ROOT, "Cropped")
LABEL_PATH   = os.path.join(OUTPUT_ROOT, "CASME2-coding-20140508.xlsx")

# Dataset parameters
N_SUBJECTS   = 10      # fake subjects
N_CLIPS_EACH = 8       # clips per subject
FRAME_W      = 640
FRAME_H      = 480
PATCH_SIZE   = 96

EMOTIONS = ['happiness', 'disgust', 'repression', 'surprise', 'others']
EMOTION_WEIGHTS = [0.20, 0.20, 0.15, 0.20, 0.25]   # realistic CASME II distribution


def draw_face_patch(emotion: str, frame_idx: int,
                    n_frames: int, noise_seed: int) -> np.ndarray:
    """
    Draws a synthetic face patch with emotion-specific landmark movements.
    Not photo-realistic — just structured enough to test the pipeline.
    Emotion signal grows from onset (frame 0) → apex (frame n//2) → offset (frame n).
    """
    rng    = np.random.default_rng(noise_seed + frame_idx)
    img    = np.ones((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8) * 210

    # Skin tone base
    img[:] = (200, 175, 155)

    # Phase: 0 at onset, 1.0 at apex, 0 at offset
    phase  = frame_idx / max(n_frames - 1, 1)
    apex   = 1.0 - abs(2 * phase - 1.0)   # triangle wave peaking at midpoint

    cx, cy = PATCH_SIZE // 2, PATCH_SIZE // 2

    # Draw basic face oval
    cv2.ellipse(img, (cx, cy), (34, 42), 0, 0, 360, (180, 155, 135), -1)

    # Eyes
    eye_open = 8 + int(apex * 4) if emotion in ['surprise', 'fear'] else 6
    cv2.ellipse(img, (cx - 12, cy - 8), (8, eye_open), 0, 0, 360, (60, 40, 30), -1)
    cv2.ellipse(img, (cx + 12, cy - 8), (8, eye_open), 0, 0, 360, (60, 40, 30), -1)

    # Brows (move based on emotion)
    brow_y_offset = 0
    if emotion == 'surprise':
        brow_y_offset = -int(apex * 6)   # brows raise
    elif emotion in ['disgust', 'repression']:
        brow_y_offset = int(apex * 4)    # brows lower/furrow

    cv2.line(img,
             (cx - 20, cy - 18 + brow_y_offset),
             (cx - 5,  cy - 16 + brow_y_offset),
             (80, 55, 40), 2)
    cv2.line(img,
             (cx + 5,  cy - 16 + brow_y_offset),
             (cx + 20, cy - 18 + brow_y_offset),
             (80, 55, 40), 2)

    # Nose
    cv2.circle(img, (cx, cy + 4), 3, (160, 130, 110), -1)

    # Mouth (varies by emotion)
    mouth_curve = 0
    mouth_open  = 0
    if emotion == 'happiness':
        mouth_curve = int(apex * 5)      # corners up
    elif emotion == 'disgust':
        mouth_curve = -int(apex * 4)     # corners down
    elif emotion == 'surprise':
        mouth_open  = int(apex * 8)      # jaw drop

    # Mouth line
    cv2.ellipse(img,
                (cx, cy + 16 + mouth_open // 2),
                (12, 4 + mouth_open),
                0, 0, 180,
                (120, 80, 70), 2)

    # Wrinkle texture (high-frequency signal — what HFA detects)
    if emotion in ['disgust', 'repression'] and apex > 0.3:
        intensity = int(apex * 60)
        for i in range(3):
            x_offset = cx - 6 + i * 6
            cv2.line(img,
                     (x_offset, cy - 20),
                     (x_offset, cy - 10),
                     (160 - intensity, 135 - intensity, 115 - intensity), 1)

    # Gaussian noise (realistic skin texture)
    noise = rng.integers(-12, 12, img.shape, dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img   # BGR (cv2 default)


def generate():
    os.makedirs(CROPPED_DIR, exist_ok=True)
    print(f"Generating synthetic CASME II dataset at: {OUTPUT_ROOT}")

    records = []
    rng     = np.random.default_rng(42)

    for sub in range(1, N_SUBJECTS + 1):
        sub_str  = f"sub{str(sub).zfill(2)}"
        sub_dir  = os.path.join(CROPPED_DIR, sub_str)
        os.makedirs(sub_dir, exist_ok=True)

        for clip_idx in range(N_CLIPS_EACH):
            # Pick emotion
            emotion   = rng.choice(EMOTIONS, p=EMOTION_WEIGHTS)
            clip_name = f"EP{str(clip_idx+1).zfill(2)}_0{sub}f"
            clip_dir  = os.path.join(sub_dir, clip_name)
            os.makedirs(clip_dir, exist_ok=True)

            # Number of frames: 15–40 (realistic CASME II range)
            n_frames  = int(rng.integers(15, 41))
            onset     = 0
            apex      = n_frames // 2
            offset    = n_frames - 1
            noise_seed = sub * 1000 + clip_idx * 100

            # Generate and save each frame
            for f_idx in range(n_frames):
                patch = draw_face_patch(emotion, f_idx, n_frames, noise_seed)
                fname = f"reg_img{str(f_idx+1).zfill(3)}.jpg"
                fpath = os.path.join(clip_dir, fname)
                cv2.imwrite(fpath, patch)

            records.append({
                'Subject':     sub,
                'Filename':    clip_name,
                'OnsetFrame':  onset,
                'ApexFrame':   apex,
                'OffsetFrame': offset,
                'Action':      'synthetic',
                'Emotion':     emotion,
                'Notes':       'synthetic_data'
            })

            print(f"  {sub_str}/{clip_name}  emotion={emotion}  frames={n_frames}")

    # Save label Excel file
    df = pd.DataFrame(records)
    df.to_excel(LABEL_PATH, index=False)

    print(f"\nDone. Generated {len(records)} clips across {N_SUBJECTS} subjects.")
    print(f"Label file saved to: {LABEL_PATH}")
    print(f"\nEmotion distribution:")
    print(df['Emotion'].value_counts().to_string())
    print(f"\nNow run:")
    print(f"  python -m mmbi.models.hfa_trainer")


if __name__ == '__main__':
    generate()