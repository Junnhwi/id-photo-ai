import os
import json
import shutil
from typing import Dict, List


def _retouch_rel_to_white_rel(retouch_rel: str) -> str:
    """
    예:
      retouch/retouch_white_idphoto_01_xxx.jpg
      -> background/white_idphoto_01_xxx.jpg
    """
    name = os.path.basename(retouch_rel)

    if not name.startswith("retouch_"):
        raise ValueError(f"Unexpected retouch filename format: {retouch_rel}")

    white_name = name[len("retouch_"):]  # retouch_ 제거
    return f"background/{white_name}"


def _make_caption(trigger_token: str) -> str:
    """
    V1은 공통 캡션 하나로 단순하게 시작.
    """
    return f"{trigger_token}, korean male, portrait, front-facing, white background, ID photo"


def build_training_dataset(
    job_path: str,
    report: dict,
    *,
    trigger_token: str = "jhwface",
) -> Dict:
    """
    embedding 단계에서 kept 된 이미지들을 기준으로
    대응되는 background/white_*.jpg를 train_dataset/으로 복사하고,
    같은 이름의 txt 캡션 파일을 생성한다.
    """

    identity = report.get("identity", {})
    kept_retouch = identity.get("kept", [])

    if not kept_retouch:
        raise RuntimeError("No kept identity images found. Run /embedding first.")

    dataset_dir = os.path.join(job_path, "train_dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    caption_text = _make_caption(trigger_token)

    selected_files: List[Dict] = []
    missing_files: List[Dict] = []

    idx = 1
    for retouch_rel in kept_retouch:
        try:
            white_rel = _retouch_rel_to_white_rel(retouch_rel)
        except Exception as e:
            missing_files.append({
                "src_retouch": retouch_rel,
                "reason": f"name mapping failed: {type(e).__name__}: {e}"
            })
            continue

        src_abs = os.path.join(job_path, white_rel)
        if not os.path.exists(src_abs):
            missing_files.append({
                "src_retouch": retouch_rel,
                "expected_white": white_rel,
                "reason": "mapped white image not found"
            })
            continue

        stem = f"{idx:04d}"
        dst_img_name = f"{stem}.jpg"
        dst_txt_name = f"{stem}.txt"

        dst_img_abs = os.path.join(dataset_dir, dst_img_name)
        dst_txt_abs = os.path.join(dataset_dir, dst_txt_name)

        shutil.copy2(src_abs, dst_img_abs)

        with open(dst_txt_abs, "w", encoding="utf-8") as f:
            f.write(caption_text)

        selected_files.append({
            "index": idx,
            "src_retouch": retouch_rel,
            "src_white": white_rel,
            "dst_image": f"train_dataset/{dst_img_name}",
            "dst_caption": f"train_dataset/{dst_txt_name}",
        })

        idx += 1

    if len(selected_files) == 0:
        raise RuntimeError("Dataset build failed: no valid white background images were copied.")

    meta = {
        "trigger_token": trigger_token,
        "source_type": "background_white",
        "caption_template": caption_text,
        "count": len(selected_files),
        "files": selected_files,
        "missing": missing_files,
    }

    meta_path = os.path.join(dataset_dir, "dataset_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "trigger_token": trigger_token,
        "source_type": "background_white",
        "count": len(selected_files),
        "dir": "train_dataset",
        "meta_json": "train_dataset/dataset_meta.json",
        "files": selected_files,
        "missing": missing_files,
    }