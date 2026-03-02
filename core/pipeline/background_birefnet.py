import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BREF = os.path.join(ROOT, "third_party", "BiRefNet")
if BREF not in sys.path:
    sys.path.insert(0, BREF)

import numpy as np
import cv2


# ------------------------------------------------------------
# Padding Helpers
# ------------------------------------------------------------

def _pad_to_target(bgr: np.ndarray, target_h: int, target_w: int) -> tuple[np.ndarray, tuple[int, int]]:
    """
    입력 이미지를 target_h x target_w 로 오른쪽/아래쪽 패딩.
    return: padded_bgr, (orig_h, orig_w)
    """
    h, w = bgr.shape[:2]

    pad_bottom = max(0, target_h - h)
    pad_right = max(0, target_w - w)

    padded = cv2.copyMakeBorder(
        bgr,
        top=0, bottom=pad_bottom,
        left=0, right=pad_right,
        borderType=cv2.BORDER_REPLICATE
    )
    return padded, (h, w)


def _crop_back(arr: np.ndarray, orig_hw: tuple[int, int]) -> np.ndarray:
    """패딩된 결과를 원래 크기로 복원"""
    oh, ow = orig_hw
    return arr[:oh, :ow]


# ------------------------------------------------------------
# Alpha Blending
# ------------------------------------------------------------

def _alpha_blend_white(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    bgr: (H,W,3) uint8
    alpha: (H,W) float32 in [0,1]
    return: white background composed bgr uint8
    """
    h, w = bgr.shape[:2]

    white = np.ones((h, w, 3), dtype=np.float32) * 255.0
    fg = bgr.astype(np.float32)

    a3 = np.repeat(alpha[..., None], 3, axis=2)

    out = fg * a3 + white * (1.0 - a3)
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------------
# BiRefNet Wrapper
# ------------------------------------------------------------

class BiRefNetMatting:
    """
    BiRefNet 기반 매팅 모델 래퍼
    """

    def __init__(self, weight_path: str, device: str | None = None):
        import torch
        from models.birefnet import BiRefNet
        from utils import check_state_dict
        from safetensors.torch import load_file

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.torch = torch

        model = BiRefNet(bb_pretrained=False)

        sd = load_file(weight_path)
        sd = check_state_dict(sd)

        model.load_state_dict(sd, strict=False)
        model.eval().to(device)

        self.model = model

    def predict_alpha(self, bgr: np.ndarray) -> np.ndarray:
        """
        입력: 600x800 BGR
        내부적으로 640x800으로 패딩 후 추론
        출력은 다시 600x800으로 복원
        """
        torch = self.torch

        # 🔥 핵심: 600 → 640 패딩 (32 배수 맞춤)
        bgr_pad, orig_hw = _pad_to_target(
            bgr,
            target_h=800,
            target_w=640
        )

        rgb = cv2.cvtColor(bgr_pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            y = self.model(x)

            if isinstance(y, (list, tuple)):
                y = y[0]
            if isinstance(y, dict):
                y = y.get("pred", next(iter(y.values())))

            if y.dim() == 4 and y.size(1) != 1:
                y = y[:, :1, :, :]

            alpha = torch.sigmoid(y)[0, 0].cpu().numpy()

        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

        # 🔥 원래 600x800으로 복원
        alpha = _crop_back(alpha, orig_hw)

        return alpha


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def remove_bg_and_compose_white(
    bgr: np.ndarray,
    matting: BiRefNetMatting
) -> tuple[np.ndarray, np.ndarray]:
    """
    returns:
      bgra uint8 (H,W,4)
      white_bgr uint8 (H,W,3)
    """

    alpha = matting.predict_alpha(bgr)

    alpha_u8 = (alpha * 255.0).astype(np.uint8)

    # 🔥 OpenCV 기준으로 BGRA 그대로 생성
    bgra = np.dstack([bgr, alpha_u8])

    white_bgr = _alpha_blend_white(bgr, alpha)

    return bgra, white_bgr