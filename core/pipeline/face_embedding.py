import os
import json
from typing import List, Dict, Tuple

import cv2
import numpy as np
import mediapipe as mp


# ArcFace 112x112 5-point template (표준)
_ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # left mouth
        [70.7299, 92.2041],  # right mouth
    ],
    dtype=np.float32,
)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def _list_retouch_images(job_path: str, report: dict) -> list[str]:
    outputs = report.get("retouch", {}).get("outputs", [])
    if not outputs:
        return []

    files = []
    for rel in outputs:
        abs_path = os.path.join(job_path, rel)
        if os.path.exists(abs_path):
            files.append(rel)

    return files


class FaceMeshAligner:
    """
    MediaPipe FaceMesh로 5개 포인트(눈/코/입꼬리)를 뽑아서 ArcFace 112x112로 정렬한다.
    """
    def __init__(self):
        self.fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        # MediaPipe FaceMesh indices
        self.IDX_LEFT_EYE = 33
        self.IDX_RIGHT_EYE = 263
        self.IDX_NOSE = 1
        self.IDX_MOUTH_L = 61
        self.IDX_MOUTH_R = 291

    def align_112(self, bgr: np.ndarray) -> Tuple[np.ndarray | None, Dict | None]:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.fm.process(rgb)
        if not res.multi_face_landmarks:
            return None, None

        lm = res.multi_face_landmarks[0].landmark

        def pt(i: int) -> np.ndarray:
            return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

        src = np.stack(
            [
                pt(self.IDX_LEFT_EYE),
                pt(self.IDX_RIGHT_EYE),
                pt(self.IDX_NOSE),
                pt(self.IDX_MOUTH_L),
                pt(self.IDX_MOUTH_R),
            ],
            axis=0,
        )

        M, _ = cv2.estimateAffinePartial2D(src, _ARCFACE_DST, method=cv2.LMEDS)
        if M is None:
            return None, None

        aligned = cv2.warpAffine(
            bgr,
            M,
            (112, 112),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        meta = {"src5": src.tolist()}
        return aligned, meta


class ArcFaceONNXEmbedder:
    """
    ArcFace ONNX (arcfaceresnet100-8.onnx) 임베딩 추출기.
    - 입력: aligned 112x112 BGR
    - 출력: 512-d embedding (L2 normalized)
    """
    def __init__(self, model_path: str, prefer_gpu: bool = True):
        import onnxruntime as ort

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ArcFace ONNX not found: {model_path}")

        providers = ort.get_available_providers()
        use_cuda = prefer_gpu and ("CUDAExecutionProvider" in providers)

        self.providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_cuda
            else ["CPUExecutionProvider"]
        )

        self.sess = ort.InferenceSession(model_path, providers=self.providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def embed_aligned(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        # BGR -> RGB
        rgb = cv2.cvtColor(aligned_bgr_112, cv2.COLOR_BGR2RGB).astype(np.float32)

        # ArcFace 관행 전처리: (img - 127.5) / 128.0
        rgb = (rgb - 127.5) / 128.0

        x = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)  # (1,3,112,112)

        y = self.sess.run([self.out_name], {self.in_name: x})[0]  # (1,512) or similar
        emb = y[0].astype(np.float32)

        # L2 normalize
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb


def extract_identity_embeddings(
    job_path: str,
    report: dict,
    *,
    device: str = "cuda",
    sim_threshold: float = 0.38,
) -> Dict:
    """
    (InsightFace 대신) MediaPipe 정렬 + ArcFace ONNX로 임베딩을 뽑고, 동일인 체크(outlier 제거) 후 저장한다.

    저장:
      data/jobs/{job_id}/embeddings/
        - face_embeds.npy (K,512)
        - identity_embedding.npy (512,)
        - meta.json
        - identity_ref.jpg
    """
    rel_paths = _list_retouch_images(job_path, report)
    if not rel_paths:
        raise RuntimeError("No retouch images found. Run /retouch first.")

    prefer_gpu = (device.lower() == "cuda")
    arcface_path = os.path.join("third_party", "BiRefNet", "weights", "arcfaceresnet100-8.onnx")

    aligner = FaceMeshAligner()
    embedder = ArcFaceONNXEmbedder(arcface_path, prefer_gpu=prefer_gpu)

    items = []
    for rel in rel_paths:
        abs_path = os.path.join(job_path, rel)
        img = cv2.imread(abs_path)
        if img is None:
            items.append({"src": rel, "ok": False, "reason": "Failed to read image"})
            continue

        aligned, a_meta = aligner.align_112(img)
        if aligned is None:
            items.append({"src": rel, "ok": False, "reason": "Face alignment failed (FaceMesh/affine)"})
            continue

        try:
            emb = embedder.embed_aligned(aligned)
        except Exception as e:
            items.append({"src": rel, "ok": False, "reason": f"ONNX infer error: {type(e).__name__}: {e}"})
            continue

        items.append({
            "src": rel,
            "ok": True,
            "embedding": emb.astype(np.float32),
            "align": a_meta,
        })

    ok_items = [x for x in items if x.get("ok")]
    if len(ok_items) < 3:
        raise RuntimeError(f"Too few embeddings extracted: {len(ok_items)} (need >= 3)")

    # reference: 첫 ok를 ref로
    ref_emb = ok_items[0]["embedding"]

    kept, dropped = [], []
    for x in ok_items:
        sim = _cosine_sim(x["embedding"], ref_emb)
        x["sim_to_ref"] = sim
        if sim >= sim_threshold:
            kept.append(x)
        else:
            dropped.append(x)

    # fallback: 평균 임베딩 기준으로 한 번 더
    if len(kept) < 3:
        mean_emb = np.mean([x["embedding"] for x in ok_items], axis=0).astype(np.float32)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        kept, dropped = [], []
        for x in ok_items:
            sim = _cosine_sim(x["embedding"], mean_emb)
            x["sim_to_ref"] = sim
            if sim >= sim_threshold:
                kept.append(x)
            else:
                dropped.append(x)

    if len(kept) < 3:
        raise RuntimeError(f"Identity check failed: kept={len(kept)} (<3). Try lowering sim_threshold.")

    embeds = np.stack([x["embedding"] for x in kept], axis=0).astype(np.float32)  # (K,512)
    rep_emb = np.mean(embeds, axis=0).astype(np.float32)
    rep_emb = rep_emb / (np.linalg.norm(rep_emb) + 1e-8)

    emb_dir = os.path.join(job_path, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    np.save(os.path.join(emb_dir, "face_embeds.npy"), embeds)
    np.save(os.path.join(emb_dir, "identity_embedding.npy"), rep_emb)

    kept_set = set(x["src"] for x in kept)
    meta = []
    for x in items:
        row = {k: v for k, v in x.items() if k not in ("embedding",)}
        row["kept"] = (x.get("src") in kept_set)
        meta.append(row)

    with open(os.path.join(emb_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 대표 이미지 저장(kept 첫 번째)
    ref_rel = kept[0]["src"]
    ref_img = cv2.imread(os.path.join(job_path, ref_rel))
    if ref_img is not None:
        cv2.imwrite(os.path.join(emb_dir, "identity_ref.jpg"), ref_img)

    return {
        "method": "mediapipe_facemesh + arcface_onnx(arcfaceresnet100-8)",
        "device": device,
        "providers": getattr(embedder, "providers", []),
        "sim_threshold": sim_threshold,
        "inputs": rel_paths,
        "kept": [x["src"] for x in kept],
        "dropped": [x["src"] for x in dropped],
        "saved": {
            "embeddings_npy": "embeddings/face_embeds.npy",
            "identity_embedding_npy": "embeddings/identity_embedding.npy",
            "meta_json": "embeddings/meta.json",
            "identity_ref": "embeddings/identity_ref.jpg",
        },
    }