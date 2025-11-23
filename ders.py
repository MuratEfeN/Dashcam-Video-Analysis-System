import cv2
import numpy as np
import time

# Video ve çekirdek
cam = cv2.VideoCapture("videos/car1.mp4")
kernel = np.ones((3, 3), dtype=np.uint8)

# YOLO modeli yükleme
net = cv2.dnn.readNet("models/yolov3.weights", "models/yolov3.cfg")
with open("models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Ekrandaki anlık nesne sayısı için sayaçlar
current_objects = {
    "car": 0,
    "person": 0,
    "traffic light": 0
}

# Nesne takibi için gerekli yapılar
detected_objects = {}  # ID'ye göre nesneleri takip eden sözlük
next_id = 0  # Benzersiz ID için sayaç
detection_timeout = 20  # Saniye cinsinden nesne takip süresi
min_detection_distance = 30  # Piksel cinsinden minimum mesafe

# Trafik ışığı analizi için renk aralıkları
RED_MIN = np.array([0, 50, 50])
RED_MAX = np.array([10, 255, 255])
RED2_MIN = np.array([170, 50, 50])
RED2_MAX = np.array([180, 255, 255])
YELLOW_MIN = np.array([15, 50, 50])
YELLOW_MAX = np.array([35, 255, 255])
GREEN_MIN = np.array([35, 50, 50])
GREEN_MAX = np.array([90, 255, 255])

# Şerit durumu
lane_status = "Bilinmiyor"
traffic_light_status = "Not Detected"  # Başlangıç değeri


def detect_objects_yolo(frame):
    global current_objects, detected_objects, traffic_light_status, next_id

    current_time = time.time()
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []
    traffic_light_boxes = []

    # Mevcut frame için sayaçları sıfırla
    current_objects = {"car": 0, "person": 0, "traffic light": 0}

    # Her frame'de trafik ışığı durumunu sıfırla
    traffic_light_status = "Not Detected"

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]
            if confidence > 0.5 and (classes[class_id] == "car" or classes[class_id] == "person" or classes[
                class_id] == "traffic light"):
                center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # Trafik ışığı kutuları ayrıca kaydet
                if classes[class_id] == "traffic light":
                    traffic_light_boxes.append([x, y, int(w), int(h)])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Mevcut algılanan nesneleri işaretle
    current_detections = {}

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        class_name = classes[class_ids[i]]
        confidence = confidences[i]
        center_x = x + w // 2
        center_y = y + h // 2

        # Ekrandaki anlık nesne sayısını artır
        current_objects[class_name] += 1

        # Nesne türüne göre renk belirle
        if class_name == "traffic light":
            color = (255, 0, 255)  # Mor (pembe) - Trafik ışıkları için
        elif class_name == "car":
            color = (0, 165, 255)  # Turuncu - Araçlar için
        elif class_name == "person":
            color = (0, 255, 255)  # Sarı - İnsanlar için
        else:
            color = (0, 255, 0)  # Varsayılan yeşil

        # Etiket oluştur
        label = f"{class_name}: {confidence:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Takip için nesneyi işaretle
        detected = False

        # Aynı sınıftan yakın mesafedeki önceki nesneleri kontrol et
        for obj_id, obj_info in list(detected_objects.items()):
            if obj_info["class"] == class_name:
                old_x, old_y = obj_info["center"]
                # İki merkez nokta arasındaki mesafeyi hesapla
                distance = np.sqrt((center_x - old_x) ** 2 + (center_y - old_y) ** 2)

                if distance < min_detection_distance:
                    # Aynı nesne, konum bilgisini güncelle
                    detected_objects[obj_id]["time"] = current_time
                    detected_objects[obj_id]["center"] = (center_x, center_y)
                    detected_objects[obj_id]["box"] = (x, y, w, h)
                    current_detections[obj_id] = True
                    detected = True
                    break

        # Yeni nesne algılandı
        if not detected:
            new_id = f"{next_id}"
            next_id += 1
            detected_objects[new_id] = {
                "class": class_name,
                "time": current_time,
                "center": (center_x, center_y),
                "box": (x, y, w, h)
            }
            current_detections[new_id] = True

    # Süresi geçmiş nesneleri temizle
    for obj_id in list(detected_objects.keys()):
        if current_time - detected_objects[obj_id]["time"] > detection_timeout:
            del detected_objects[obj_id]

    # Trafik ışıklarını analiz et - sadece trafik ışığı algılandıysa
    if traffic_light_boxes:
        traffic_light_status = analyze_traffic_lights(frame, traffic_light_boxes)
    else:
        traffic_light_status = "Not Detected"

    return frame


def analyze_traffic_lights(frame, traffic_light_boxes):
    """Trafik ışıklarını analiz eder ve renklerini belirler"""
    detected_lights = []  # Tespit edilen ışıklar

    for box in traffic_light_boxes:
        x, y, w, h = box

        # Kutunun içindeki bölgeyi al
        roi = frame[max(0, y):min(y + h, frame.shape[0]), max(0, x):min(x + w, frame.shape[1])]
        if roi.size == 0:  # Boş bölge kontrolü
            continue

        # HSV'ye dönüştür
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Renk maskesi oluştur
        red_mask1 = cv2.inRange(hsv, RED_MIN, RED_MAX)
        red_mask2 = cv2.inRange(hsv, RED2_MIN, RED2_MAX)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        yellow_mask = cv2.inRange(hsv, YELLOW_MIN, YELLOW_MAX)
        green_mask = cv2.inRange(hsv, GREEN_MIN, GREEN_MAX)

        # Renk piksel sayıları
        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)

        # Renk belirleme ve dikdörtgen çizme
        if red_pixels > yellow_pixels and red_pixels > green_pixels and red_pixels > 10:
            detected_lights.append(("Red", (x, y), (x + w, y + h), (0, 0, 255)))  # Kırmızı
        elif yellow_pixels > red_pixels and yellow_pixels > green_pixels and yellow_pixels > 10:
            detected_lights.append(("Yellow", (x, y), (x + w, y + h), (0, 255, 255)))  # Sarı
        elif green_pixels > red_pixels and green_pixels > yellow_pixels and green_pixels > 10:
            detected_lights.append(("Green", (x, y), (x + w, y + h), (0, 255, 0)))  # Yeşil

    # Tespit edilen ışıkları ekranda göster
    for label, start, end, color in detected_lights:
        cv2.rectangle(frame, start, end, color, 3)  # Çerçeve çiz
        cv2.putText(frame, label, (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Eğer hiç ışık rengi tespit edilmediyse "Not Detected" döndür
    return detected_lights[0][0] if detected_lights else "Not Detected"


def crop_matris(img):
    x, y = img.shape[:2]
    sapma = 100
    value = np.array([
        [(sapma, x - sapma), (int((y * 3.2) / 8), int(x * 0.6)),
         (int((y * 5) / 8), int(x * 0.6)), (y, x - sapma)]
    ], np.int32)
    return value


def crop_image(img, matris):
    x, y = img.shape[:2]
    mask = np.zeros((x, y), dtype=np.uint8)
    mask = cv2.fillPoly(mask, matris, 255)
    return cv2.bitwise_and(img, img, mask=mask)


def filt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.inRange(img, 150, 255)
    img = cv2.erode(img, kernel)
    img = cv2.dilate(img, kernel)
    img = cv2.medianBlur(img, 9)
    img = cv2.Canny(img, 40, 200)
    return img


def line_mean(lines):
    left, right = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if slope < -0.2:
            right.append((x1, y1, x2, y2))
        elif slope > 0.2:
            left.append((x1, y1, x2, y2))
    right_mean = np.mean(right, axis=0) if right else None
    left_mean = np.mean(left, axis=0) if left else None
    return right_mean, left_mean


def draw_line(img, line):
    line = np.int32(np.around(line))
    x1, y1, x2, y2 = line
    return cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 10)


def draw_polylines(img, matris):
    dst = np.array([
        [matris[0][1, 0], matris[0][1, 1]],
        [matris[0][0, 0], matris[0][0, 1]],
        [matris[0][3, 0], matris[0][3, 1]],
        [matris[0][2, 0], matris[0][2, 1]]
    ], np.int32)
    return cv2.polylines(img, [dst], True, (0, 255, 255), 2)


def pers(img, matris, resize_x=300, resize_y=200):
    x, y = img.shape[:2]
    src = np.float32([
        [matris[0][1, 0], matris[0][1, 1]],
        [matris[0][2, 0], matris[0][2, 1]],
        [matris[0][0, 0], matris[0][0, 1]],
        [matris[0][3, 0], matris[0][3, 1]]
    ])
    dst = np.float32([
        [0, 0], [y - 1, 0], [0, x - 1], [y - 1, x - 1]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.resize(cv2.warpPerspective(img, M, (y, x)), (resize_x, resize_y))


def draw_info_panel(img, current_objects, lane_status, traffic_light_status):
    """Bilgi paneli çizme"""
    height, width = img.shape[:2]

    # Panel arka planı
    panel_height = 220  # Yüksekliği artırdık
    panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
    panel[:, :] = (50, 50, 50)  # Koyu gri arkaplan

    # Başlık - daha kalın ve net
    cv2.putText(panel, "DRIVING ASSISTANT INFO PANEL", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)

    # Ekrandaki Anlık Nesne Sayımı
    cv2.putText(panel, f"Vehicle Count on Screen: {current_objects['car']}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 205, 255), 2)
    cv2.putText(panel, f"Person Count on Screen: {current_objects['person']}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 205, 255), 2)
    cv2.putText(panel, f"Traffic Light Count on Screen: {current_objects['traffic light']}", (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 205, 255), 2)

    # Uyarı Simgeleri ve Durumları
    warning_y = 75
    warning_x_start = width - 250

    # Yaya uyarısı
    if current_objects['person'] > 0:
        # Uyarı simgesi (üçgen)
        triangle_points = np.array([[warning_x_start, warning_y],
                                    [warning_x_start - 15, warning_y + 20],
                                    [warning_x_start + 15, warning_y + 20]], np.int32)
        cv2.fillPoly(panel, [triangle_points], (0, 255, 255))  # Sarı
        cv2.putText(panel, "!", (warning_x_start - 4, warning_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(panel, "PEDESTRIAN ALERT", (warning_x_start + 25, warning_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        warning_y += 35

    # Araç yoğunluğu uyarısı
    if current_objects['car'] > 15:
        # Uyarı simgesi (daire)
        cv2.circle(panel, (warning_x_start, warning_y + 10), 12, (0, 165, 255), -1)  # Turuncu
        cv2.putText(panel, "!", (warning_x_start - 4, warning_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(panel, "HIGH TRAFFIC DENSITY", (warning_x_start + 25, warning_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        warning_y += 35

    # Trafik ışığı kırmızı uyarısı
    if traffic_light_status == "Red":
        # Dur simgesi (dikdörtgen)
        cv2.rectangle(panel, (warning_x_start - 15, warning_y),
                      (warning_x_start + 15, warning_y + 20), (0, 0, 255), -1)  # Kırmızı
        cv2.putText(panel, "STOP", (warning_x_start - 12, warning_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(panel, "RED LIGHT - STOP!", (warning_x_start + 25, warning_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Şerit Durumu çevirisi
    if lane_status == "Iki serit algilandi":
        lane_status_eng = "Both lanes detected"
    elif lane_status == "Sag serit algilandi":
        lane_status_eng = "Right lane detected"
    elif lane_status == "Sol serit algilandi":
        lane_status_eng = "Left lane detected"
    elif lane_status == "Algilanmadi":
        lane_status_eng = "Not detected"
    else:
        lane_status_eng = "Unknown"

    cv2.putText(panel, f"Lane Status: {lane_status_eng}", (width // 2, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)

    # Trafik Işığı Durumu
    light_color = (255, 255, 255)  # Varsayılan beyaz

    if traffic_light_status == "Red":
        light_color = (0, 0, 255)  # Kırmızı
    elif traffic_light_status == "Yellow":
        light_color = (0, 255, 255)  # Sarı
    elif traffic_light_status == "Green":
        light_color = (0, 255, 0)  # Yeşil
    elif traffic_light_status == "Not Detected":
        light_color = (128, 128, 128)  # Gri

    cv2.putText(panel, f"Traffic Light: {traffic_light_status}", (width // 2, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, light_color, 2)

    # Zamanı ve tarih ekle
    current_time = time.strftime("%H:%M:%S")
    current_date = time.strftime("%d/%m/%Y")
    cv2.putText(panel, f"Time: {current_time} - Date: {current_date}", (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

    # Panel ile resmi birleştir
    result = np.vstack((panel, img))
    return result


# Ana döngü
frame_count = 0
fps_start_time = time.time()
fps = 0

# Tam ekran ayarları
window_name = "Driving Assistant - Full Screen"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cam.isOpened():
    ret, image = cam.read()
    if not ret:
        print("Video bitti.")
        break

    # FPS hesaplama
    frame_count += 1
    if frame_count >= 10:  # Her 10 framede bir FPS güncelle
        fps = frame_count / (time.time() - fps_start_time)
        fps_start_time = time.time()
        frame_count = 0

    # Şerit takip işlemleri
    matris = crop_matris(image)
    img_crop = crop_image(image, matris)
    img_filtered = filt(img_crop)
    lines = cv2.HoughLinesP(img_filtered, 1, np.pi / 180, 20, minLineLength=5, maxLineGap=200)

    image = draw_polylines(image, matris)

    # Şerit durumunu belirle
    lane_status = "Algilanmadi"
    if lines is not None:
        right_line, left_line = line_mean(lines)

        # Şerit durumunu belirle
        if right_line is not None and left_line is not None:
            lane_status = "Iki serit algilandi"
            image = draw_line(image, right_line)
            image = draw_line(image, left_line)
        elif right_line is not None:
            lane_status = "Sag serit algilandi"
            image = draw_line(image, right_line)
        elif left_line is not None:
            lane_status = "Sol serit algilandi"
            image = draw_line(image, left_line)

    # YOLO ile araç tespiti
    image = detect_objects_yolo(image)

    # FPS gösterimi
    cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Bilgi paneli ekle
    final_image = draw_info_panel(image, current_objects, lane_status, traffic_light_status)

    # Göster
    cv2.imshow(window_name, final_image)

    # ESC tuşu ile çıkış (tam ekranda q tuşu çalışmayabilir)
    key = cv2.waitKey(16) & 0xFF
    if key == ord("q") or key == 27:  # q veya ESC tuşu
        break

cam.release()
cv2.destroyAllWindows()