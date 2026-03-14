import cv2


def tone_normalize(bgr):
    """
    밝기/대비 안정화
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise(bgr):
    """
    경계를 유지하면서 노이즈 제거
    """
    return cv2.bilateralFilter(bgr, d=3, sigmaColor=15, sigmaSpace=15)


def sharpen(bgr):
    """
    샤프닝
    """
    blur = cv2.GaussianBlur(bgr, (0, 0), 0.6)
    return cv2.addWeighted(bgr, 1.05, blur, -0.05, 0)


def retouch_image(bgr):
    """
    Deprecated compatibility helper.
    Current pipeline does not require retouch stage,
    but this function remains to ease merge/conflict resolution with older branches.
    """
    img = tone_normalize(bgr)
    img = denoise(img)
    img = sharpen(img)
    return img
