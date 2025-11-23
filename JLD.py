import cv2
import numpy as np

cam = cv2.VideoCapture("car1.mp4")
sapma = 100
kernel = np.ones((3, 3), dtype=np.uint8)


def crop_matris(img):
    x, y = img.shape[:2]
    value = np.array([
        [(sapma, x - sapma), (int((y * 3.2) / 8), int(x * 0.6)),
         (int((y * 5) / 8), int(x * 0.6)), (y, x - sapma)]], np.int32)
    return value


def crop_image(img, matris):
    x, y = img.shape[:2]
    mask = np.zeros(shape=(x, y), dtype=np.uint8)
    mask = cv2.fillPoly(mask, matris, 255)
    mask = cv2.bitwise_and(img, img, mask=mask)
    return mask


def filt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.inRange(img, 150, 255)
    img = cv2.erode(img, kernel)
    img = cv2.dilate(img, kernel)
    img = cv2.medianBlur(img, 9)
    img = cv2.Canny(img, 40, 200)
    return img


def line_mean(lines):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:  # Dikey çizgileri atla
            continue
        slope = (y2 - y1) / (x2 - x1)
        if slope < -0.2:
            right.append((x1, y1, x2, y2))
        elif slope > 0.2:
            left.append((x1, y1, x2, y2))

    # Ortalama hesaplama - güvenli versiyon
    right_mean = np.mean(right, axis=0) if right else None
    left_mean = np.mean(left, axis=0) if left else None

    return right_mean, left_mean


def draw_line(img, line):
    if line is not None:
        line = np.int32(np.around(line))
        x1, y1, x2, y2 = line
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return img


def draw_polylines(img, matris):
    dst = np.array([[matris[0][1, 0], matris[0][1, 1]],
                    [matris[0][0, 0], matris[0][0, 1]],
                    [matris[0][3, 0], matris[0][3, 1]],
                    [matris[0][2, 0], matris[0][2, 1]]], np.int32)
    cv2.polylines(img, [dst], True, (0, 255, 255), 2)
    return img


def pers(img, matris, resize_x=300, resize_y=200):
    x, y = img.shape[:2]
    src = np.float32([
        [matris[0][1, 0], matris[0][1, 1]],
        [matris[0][2, 0], matris[0][2, 1]],
        [matris[0][0, 0], matris[0][0, 1]],
        [matris[0][3, 0], matris[0][3, 1]]])
    dst = np.float32([
        [0, 0],
        [resize_x - 1, 0],
        [0, resize_y - 1],
        [resize_x - 1, resize_y - 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    img_output = cv2.warpPerspective(img, M, (resize_x, resize_y))
    return img_output


def create_enhanced_preview(filtered_img, matris, resize_x=300, resize_y=200):
    """Gelişmiş filtrelenmiş görüntü önizlemesi"""
    # Filtrelenmiş görüntüyü renklendir
    filtered_colored = cv2.applyColorMap(filtered_img, cv2.COLORMAP_HOT)

    # Perspective transform uygula
    enhanced_preview = pers(filtered_colored, matris, resize_x, resize_y)

    # Başlık ekle
    cv2.putText(enhanced_preview, "Filtered View", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return enhanced_preview


def create_bird_eye_view(img, matris, resize_x=300, resize_y=200):
    """Gelişmiş kuş bakışı görünüm"""
    bird_view = pers(img, matris, resize_x, resize_y)

    # Grid çizgileri ekle
    grid_color = (100, 100, 100)
    for i in range(0, resize_x, 50):
        cv2.line(bird_view, (i, 0), (i, resize_y), grid_color, 1)
    for i in range(0, resize_y, 40):
        cv2.line(bird_view, (0, i), (resize_x, i), grid_color, 1)

    # Başlık ekle
    cv2.putText(bird_view, "Bird's Eye View", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return bird_view


def add_info_panel(img, frame_count, lines_detected):
    """Bilgi paneli ekle"""
    info_y = img.shape[0] - 80
    cv2.rectangle(img, (10, info_y), (400, img.shape[0] - 10), (0, 0, 0), -1)
    cv2.rectangle(img, (10, info_y), (400, img.shape[0] - 10), (255, 255, 255), 2)

    cv2.putText(img, f"Frame: {frame_count}", (20, info_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"Lines Detected: {lines_detected}", (20, info_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img


frame_count = 0

while cam.isOpened():

    ret, image = cam.read()
    if not ret:
        print("Video bitti")
        break

    frame_count += 1
    img_org = image.copy()

    matris = crop_matris(image)
    img_cropped = crop_image(image, matris)
    img_filtered = filt(img_cropped)

    # Gelişmiş mini ekranlar
    # Sol üst: Kuş bakışı görünüm (grid'li)
    bird_eye = create_bird_eye_view(img_org, matris)
    img_org[:200, :300] = bird_eye

    # Sağ üst: Filtrelenmiş görünüm (renkli)
    filtered_preview = create_enhanced_preview(img_filtered, matris)
    img_org[:200, 300:600] = filtered_preview

    # Çizgi tespiti
    lines = cv2.HoughLinesP(img_filtered, 1, np.pi / 180, 20,
                            minLineLength=5, maxLineGap=200)

    lines_count = len(lines) if lines is not None else 0

    # Polylines çiz
    image = draw_polylines(img_org, matris)

    # Şerit çizgileri çiz
    if lines is not None:
        right_line, left_line = line_mean(lines)
        if right_line is not None:
            image = draw_line(image, right_line)
        if left_line is not None:
            image = draw_line(image, left_line)

    # Bilgi paneli ekle
    image = add_info_panel(image, frame_count, lines_count)

    # Mini ekranların çerçevelerini belirginleştir
    cv2.rectangle(image, (0, 0), (300, 200), (255, 255, 255), 2)  # Sol üst
    cv2.rectangle(image, (300, 0), (600, 200), (255, 255, 255), 2)  # Sağ üst

    cv2.imshow("Enhanced Lane Detection", image)

    key = cv2.waitKey(16) & 0xFF
    if key == ord("q"):
        print("Program kapatıldı")
        break

cam.release()
cv2.destroyAllWindows()