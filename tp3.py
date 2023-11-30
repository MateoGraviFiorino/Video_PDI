import cv2
import numpy as np

# Función para filtrar por color (en este caso, rojo)
def filter_color(image, color_lower, color_upper):
    mask = cv2.inRange(image, color_lower, color_upper)
    return cv2.bitwise_and(image, image, mask=mask)

# Función para detectar dados en una imagen
def detect_dice(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dice_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        if 0.8 < aspect_ratio < 1.2 and 300 < cv2.contourArea(contour) < 500:
            dice_regions.append((x, y, w, h))

    return dice_regions

# Función para reconocer números en dados
def recognize_numbers(image, dice_regions):
    dice_numbers = []

    for (x, y, w, h) in dice_regions:
        # Obtener la región de interés (ROI) de cada dado
        roi = image[y:y+h, x:x+w]

        # Aquí puedes implementar tu lógica para reconocer el número en la ROI
        # Por ejemplo, podrías utilizar técnicas de OCR, aprendizaje automático, etc.

        # En este ejemplo, asigno un número aleatorio como demostración
        dice_number = np.random.randint(1, 7)
        dice_numbers.append(dice_number)

    return dice_numbers

def dibujar(imagen,region,numero,lower_red,upper_red):
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = filter_color(imagen,lower_red,upper_red)
    #filtrar por area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 300 :
            x, y, w, h = cv2.boundingRect(contour)
            if w/h> 0.92 and w/h < 1.1:
                cv2.rectangle(imagen, (x, y), (x+w,y+h), (100, 255, 0), 2)
                #print(area, contador)
                cv2.putText(imagen, "Area: " + str(area), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                #print("dado dibujado")
                #print(area)
def mascara(red_mask):
    # Convertir la máscara a escala de grises
    red_mask_gray = cv2.cvtColor(red_mask, cv2.COLOR_BGR2GRAY)
    return red_mask_gray

# Leer un video
video_path = "tirada_1.mp4"
cap = cv2.VideoCapture(video_path)

# Inicializar variables
umbral_area = 300  # Umbral para el área de los dados
umbral_frames_quietos = 65  # Umbral para considerar que el dado está quieto
frames_quietos = 0
prev_frame = None
prev_dice_regions = None

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, dsize=(int(frame.shape[1] / 3), int(frame.shape[0] / 3)))
    # Filtrar dados por su color rojo
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lower_red = np.array([80, 0, 0])
    upper_red = np.array([255, 70, 70])
    red_mask = filter_color(frame, lower_red, upper_red)
    red_mask_gray = mascara(red_mask)
    
    # Filtrar por área y obtener centroides
    _, _, _, stats = cv2.connectedComponentsWithStats(red_mask_gray)
    centroids = stats[1:, :2]  # Excluir el fondo y tomar las columnas x, y

    
    # Comparar con el frame anterior
    if prev_dice_regions is not None and len(centroids) == len(prev_dice_regions):
        centroid_diff = np.linalg.norm(stats - np.array(prev_dice_regions)[:, :2], axis=1)
        if np.all(centroid_diff > umbral_area) and np.all(centroid_diff < 360):
            frames_quietos += 1
        else:
            frames_quietos = 0

        if (frames_quietos > umbral_frames_quietos)  :
            #print("El dado se ha dejado de mover (por centroides)")
            #cv2.imshow("Frame",frame)
            # Dibujar dados basados en la detección de movimiento
            numeros_dados = recognize_numbers(frame, prev_dice_regions)
            dibujar(frame, prev_dice_regions, numeros_dados,lower_red,upper_red)
            
    prev_dice_regions = centroids

    # Detectar dados y dibujar en el frame
    dice_regions = detect_dice(frame)

    # Mostrar el resultado
    cv2.imshow('Frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
