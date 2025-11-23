import cv2
import numpy as np
import matplotlib.pyplot as plt

# Video yükle
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


def filt_steps(img):
    """Filtreleme adımlarını ayrı ayrı döndür"""
    # 1. Gri tonlama
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Threshold (beyaz çizgi tespiti)
    thresh = cv2.inRange(gray, 150, 255)

    # 3. Erosion (gürültü azaltma)
    eroded = cv2.erode(thresh, kernel)

    # 4. Dilation (çizgileri kalınlaştırma)
    dilated = cv2.dilate(eroded, kernel)

    # 5. Median Blur (pürüzsüzleştirme)
    blurred = cv2.medianBlur(dilated, 9)

    # 6. Canny (kenar tespiti)
    edges = cv2.Canny(blurred, 40, 200)

    return gray, thresh, eroded, dilated, blurred, edges


def create_comparison_image(original_crop, gray_crop):
    """Orijinal ve gri tonlama karşılaştırması"""
    # Yan yana birleştir
    h, w = original_crop.shape[:2]
    comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)

    # Sol taraf: Orijinal (renkli)
    comparison[:, :w] = original_crop

    # Sağ taraf: Gri tonlama (3 kanala çevir)
    comparison[:, w:] = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)

    # Ortada ayırıcı çizgi
    cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)

    # Yazılar ekle
    cv2.putText(comparison, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(comparison, "GRAYSCALE", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return comparison


def draw_polylines(img, matris):
    img_copy = img.copy()
    dst = np.array([[matris[0][1, 0], matris[0][1, 1]],
                    [matris[0][0, 0], matris[0][0, 1]],
                    [matris[0][3, 0], matris[0][3, 1]],
                    [matris[0][2, 0], matris[0][2, 1]]], np.int32)
    cv2.polylines(img_copy, [dst], True, (0, 255, 255), 3)
    return img_copy


def visualize_hough_lines(img, lines):
    """Hough çizgilerini renkli olarak göster"""
    img_hough = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        # Her çizgiyi farklı renkle çiz
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            color = colors[i % len(colors)]
            cv2.line(img_hough, (x1, y1), (x2, y2), color, 2)
            # Çizgi numarasını yaz
            cv2.putText(img_hough, f"{i + 1}", ((x1 + x2) // 2, (y1 + y2) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img_hough


# İlk frame'i al
ret, frame = cam.read()
if ret:
    print("Frame alındı, işleniyor...")

    # 1. Orijinal frame
    original = frame.copy()

    # 2. ROI (Region of Interest) belirleme
    matris = crop_matris(frame)
    roi_marked = draw_polylines(original, matris)

    # 3. Crop edilmiş görüntü
    cropped = crop_image(frame, matris)

    # 4. Filtreleme adımları
    gray, thresh, eroded, dilated, blurred, edges = filt_steps(cropped)

    # 4.5. Gri tonlama karşılaştırması oluştur
    gray_comparison = create_comparison_image(cropped, gray)

    # 5. Hough Line Detection
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20,
                            minLineLength=5, maxLineGap=200)

    # 6. Hough çizgileri görselleştirme
    hough_visual = visualize_hough_lines(edges, lines)

    # Matplotlib ile görselleştirme - Daha büyük figür ve spacing
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # 1. Orijinal Frame
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('1. Orijinal Frame', fontsize=12, pad=10)
    axes[0, 0].axis('off')

    # 2. ROI İşaretli
    axes[0, 1].imshow(cv2.cvtColor(roi_marked, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('2. ROI İşaretli', fontsize=12, pad=10)
    axes[0, 1].axis('off')

    # 3. Crop Edilmiş
    axes[0, 2].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('3. Crop Edilmiş', fontsize=12, pad=10)
    axes[0, 2].axis('off')

    # 4. Gri Tonlama Karşılaştırması
    axes[1, 0].imshow(cv2.cvtColor(gray_comparison, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('4. Renkli vs Gri Tonlama', fontsize=12, pad=10)
    axes[1, 0].axis('off')

    # 5. Threshold
    axes[1, 1].imshow(thresh, cmap='gray')
    axes[1, 1].set_title('5. Threshold\n(150-255)', fontsize=12, pad=10)
    axes[1, 1].axis('off')

    # 6. Erosion
    axes[1, 2].imshow(eroded, cmap='gray')
    axes[1, 2].set_title('6. Erosion\n(Gürültü Azaltma)', fontsize=12, pad=10)
    axes[1, 2].axis('off')

    # 7. Dilation
    axes[2, 0].imshow(dilated, cmap='gray')
    axes[2, 0].set_title('7. Dilation\n(Kalınlaştırma)', fontsize=12, pad=10)
    axes[2, 0].axis('off')

    # 8. Canny Edges
    axes[2, 1].imshow(edges, cmap='gray')
    axes[2, 1].set_title('8. Canny Edge\nDetection', fontsize=12, pad=10)
    axes[2, 1].axis('off')

    # 9. Hough Lines
    axes[2, 2].imshow(cv2.cvtColor(hough_visual, cv2.COLOR_BGR2RGB))
    axes[2, 2].set_title(f'9. Hough Lines\n({len(lines) if lines is not None else 0} adet)', fontsize=12, pad=10)
    axes[2, 2].axis('off')

    # Spacing ayarlamaları
    plt.subplots_adjust(
        left=0.05,  # Sol kenar boşluğu
        right=0.95,  # Sağ kenar boşluğu
        top=0.95,  # Üst kenar boşluğu (ana başlık yok)
        bottom=0.05,  # Alt kenar boşluğu
        wspace=0.15,  # Yan yana subplot'lar arası boşluk
        hspace=0.35  # Alt-üst subplot'lar arası boşluk
    )

    plt.show()

    # İstatistikler yazdır
    print(f"\n=== FRAME ANALİZ SONUÇLARI ===")
    print(f"Orijinal boyut: {original.shape}")
    print(f"Crop sonrası sıfır olmayan piksel: {np.count_nonzero(cropped)}")
    print(f"Threshold sonrası beyaz piksel: {np.count_nonzero(thresh)}")
    print(f"Canny sonrası kenar piksel: {np.count_nonzero(edges)}")
    print(f"Tespit edilen çizgi sayısı: {len(lines) if lines is not None else 0}")

    if lines is not None:
        print(f"\n=== ÇIZGI DETAYLARI ===")
        for i, line in enumerate(lines[:5]):  # İlk 5 çizgiyi göster
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
                print(f"Çizgi {i + 1}: ({x1},{y1})->({x2},{y2}) | Uzunluk: {length:.1f} | Eğim: {slope:.2f}")

    # Ayrı pencerede de göster (isteğe bağlı)
    show_opencv = input("\nOpenCV pencereleri ile de görmek ister misiniz? (y/n): ")
    if show_opencv.lower() == 'y':
        cv2.imshow("1-Original", original)
        cv2.imshow("2-ROI", roi_marked)
        cv2.imshow("3-Cropped", cropped)
        cv2.imshow("4-Gray", gray)
        cv2.imshow("5-Threshold", thresh)
        cv2.imshow("6-Edges", edges)
        cv2.imshow("7-Hough Lines", hough_visual)

        print("Penceleleri kapatmak için herhangi bir tuşa basın...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

else:
    print("Video okunamadı!")

cam.release()