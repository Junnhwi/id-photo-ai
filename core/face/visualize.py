import os
import cv2


def save_face_preview(
    image_bgr,
    faces,
    main_face_bbox,
    *,
    save_dir: str,
    filename: str,
) -> str:
    """
    얼굴 박스가 그려진 preview 이미지를 work/ 폴더에 저장한다.

    faces:
      - MediaPipe 형식: [{"bbox": (x,y,w,h), "score": float}, ...]  또는
      - 박스 리스트: [(x,y,w,h), ...]
    main_face_bbox:
      - 대표 얼굴 박스 (x,y,w,h) 또는 None
    """
    os.makedirs(save_dir, exist_ok=True)
    preview = image_bgr.copy()

    # faces가 dict 리스트인지, tuple 리스트인지 구분해서 박스만 뽑기
    boxes = []
    for item in faces:
        if isinstance(item, dict) and "bbox" in item:
            boxes.append(item["bbox"])
        else:
            boxes.append(item)

    # 1) 모든 박스(초록)
    for (x, y, w, h) in boxes:
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 2) 대표 얼굴(빨강, 굵게)
    if main_face_bbox is not None:
        x, y, w, h = main_face_bbox
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # 3) 텍스트
    cv2.putText(
        preview,
        f"faces={len(boxes)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    preview_name = f"preview__{filename}"
    preview_path = os.path.join(save_dir, preview_name)
    cv2.imwrite(preview_path, preview)
    return preview_path