import cv2
import numpy as np
import glob
import os

# ──────────────────────────────────────────
# 設定
# ──────────────────────────────────────────
IMAGE_FOLDER = "/Users/rainhuang/Desktop/Data"
OUTPUT_PATH  = "/Users/rainhuang/Desktop/Data/output_stitched.jpg"
DIRECTION    = "horizontal"
AUTO_DETECT  = False
# ──────────────────────────────────────────


def load_images(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                   glob.glob(os.path.join(folder, "*.png")))
    if not paths:
        raise FileNotFoundError(f"搵唔到圖片：{folder}")
    print(f"找到 {len(paths)} 張圖片：{[os.path.basename(p) for p in paths]}")
    return [cv2.imread(p) for p in paths], paths


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def auto_detect_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:5]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return order_points(approx.reshape(4, 2))
    h, w = img.shape[:2]
    print("  ⚠️  自動偵測失敗，使用整張圖邊界")
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]])


def manual_select_corners(img, filename):
    points = []
    display = img.copy()
    window = f"點選4個角點：{filename}  [ESC 取消]"

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(display, str(len(points)), (x+10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window, display)

    cv2.imshow(window, display)
    cv2.setMouseCallback(window, click)
    while len(points) < 4:
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    return order_points(np.array(points, dtype="float32"))


def correct_perspective(img, corners):
    tl, tr, br, bl = corners
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (w, h))


def enhance_document(img):
    """輕微抗鋸齒，唔做其他處理"""
    return cv2.GaussianBlur(img, (3, 3), 0.5)


def stitch_images(images, direction="horizontal"):
    if direction == "horizontal":
        target_h = min(img.shape[0] for img in images)
        resized = [cv2.resize(img, (int(img.shape[1] * target_h / img.shape[0]), target_h))
                   for img in images]
        return np.hstack(resized)
    else:
        target_w = min(img.shape[1] for img in images)
        resized = [cv2.resize(img, (target_w, int(img.shape[0] * target_w / img.shape[1])))
                   for img in images]
        return np.vstack(resized)


def show_and_wait(window_name, img):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    print("按任意鍵關閉視窗...")
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key != 255:
            break
        try:
            vis = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
            if vis < 1:
                break
        except:
            break
    cv2.destroyAllWindows()
    for _ in range(5):      # macOS 需要多幾個 tick 先真正銷毀
        cv2.waitKey(1)


def main():
    images, paths = load_images(IMAGE_FOLDER)
    corrected = []

    for i, (img, path) in enumerate(zip(images, paths)):
        fname = os.path.basename(path)
        print(f"處理：{fname}")

        if AUTO_DETECT:
            corners = auto_detect_corners(img)
        else:
            corners = manual_select_corners(img, fname)

        warped = correct_perspective(img, corners)
        enhanced = enhance_document(warped)
        corrected.append(enhanced)

        print(f"  ✓ 完成，尺寸：{enhanced.shape[1]}x{enhanced.shape[0]}")

    print(f"\n拼接 {len(corrected)} 張圖片（{DIRECTION}）...")
    final = stitch_images(corrected, DIRECTION)
    print(f"最終尺寸：{final.shape[1]}x{final.shape[0]}")

    success = cv2.imwrite(OUTPUT_PATH, final)
    if not success:
        fallback = OUTPUT_PATH.rsplit(".", 1)[0] + ".png"
        cv2.imwrite(fallback, final)
        print(f"✅ 儲存至（PNG）：{fallback}")
    else:
        print(f"✅ 儲存至：{OUTPUT_PATH}")

def show_and_wait(window_name, img):
    import tempfile, subprocess, os
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, img)
    subprocess.run(["open", tmp.name])
    print(f"已用系統預覽開啟")


if __name__ == "__main__":
    main()