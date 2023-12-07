import cv2
import numpy as np

def filter_color_hsv(image, color_lower1, color_upper1, color_lower2, color_upper2):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, color_lower1, color_upper1)
    mask2 = cv2.inRange(hsv, color_lower2, color_upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    return cv2.bitwise_and(image, image, mask=mask)

def dilate_img(img, soft=False):
    kernel_dilatacion = np.ones((9, 9), np.uint8)
    kernel_dilatacion_soft = np.ones((3, 3), np.uint8)

    kernel = kernel_dilatacion_soft if soft else kernel_dilatacion
    return cv2.dilate(img, kernel)

video_path = "tirada_1.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, dsize=(int(frame.shape[1] / 3), int(frame.shape[0] / 3)))

    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([5, 255, 255])
    lower_red2 = np.array([170, 100, 50])
    upper_red2 = np.array([179, 255, 255])
    red_mask = filter_color_hsv(frame, lower_red1, upper_red1, lower_red2, upper_red2)
    red_mask_gray = cv2.cvtColor(red_mask, cv2.COLOR_BGR2GRAY)

    # Filtrar por área y obtener centroides
    labels, _, stats, centroids = cv2.connectedComponentsWithStats(red_mask_gray)

    # Crear la máscara una sola vez fuera del bucle
    dice_mask = np.zeros_like(red_mask_gray)

    # Visualizar objetos encontrados
    for stat in stats[1:]:  # Excluir el fondo (índice 0)
        x, y, w, h, area = map(int, stat)

        # Calcular el aspect ratio del rectángulo
        aspect_ratio = w / float(h)

        # Filtrar por área y aspect ratio
        if 300 < area < 500 and 0.8 < aspect_ratio < 1.2:
            # Acumular la región en la máscara
            dice_mask[y:y+h, x:x+w] += red_mask_gray[y:y+h, x:x+w]

    # Aplicar cierre morfológico sin dilatación
    kernel_cierre = np.ones((1, 1), np.uint8)  # Ajusta el tamaño del kernel
    dice_mask = cv2.morphologyEx(dice_mask, cv2.MORPH_CLOSE, kernel_cierre, iterations=1)  # Ajusta el número de iteraciones
    
    # Umbralizar la máscara
    _, binary_mask = cv2.threshold(dice_mask, 1, 255, cv2.THRESH_BINARY)
    
    # Encontrar contornos en la máscara binaria
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos por área
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 50]

    # Visualizar y contar puntos en los dados
    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        dice_region = frame[y:y+h, x:x+w]

        # Convertir la región del dado a escala de grises
        dice_gray = cv2.cvtColor(dice_region, cv2.COLOR_BGR2GRAY)

        # Umbralizar la región del dado
        _, binary_dice = cv2.threshold(dice_gray, 195, 255, cv2.THRESH_BINARY)
        #cv2.imshow("",binary_dice)
        # Encontrar contornos en la región binarizada
        contours_inside_dice, _ = cv2.findContours(binary_dice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Contar el número de puntos basándonos en el número de contornos
        #points_inside_dice = len(contours_inside_dice)
        points_inside_dice = max(1, min(len(contours_inside_dice), 6))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'{points_inside_dice}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
