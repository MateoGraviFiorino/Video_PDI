import cv2
import numpy as np

# --- Leer un video ------------------------------------------------
cap = cv2.VideoCapture('tirada_1.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while (cap.isOpened()): 
    ret, frame = cap.read()

    if ret==True:
        frame = cv2.resize(frame, dsize=(int(width/3), int(height/3)))

        # Comparar distancias de los centroides y centroides_anterior
        
        #filtrar dados por su color rojo
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lower_red = np.array([100, 0, 0])
        upper_red = np.array([255, 70, 70])
        mask = cv2.inRange(hsv, lower_red, upper_red
        )
        res = cv2.bitwise_and(frame, frame, mask=mask)

        #filtrar por area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w,
                y+h), (0, 255, 0), 2)
        
                cv2.putText(frame, "Area: " + str(area), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    #filtrar por realcion de aspecto
    #si se detectan la cantidad de dados deseadps, comparar con el frame anterior sus posiciosnes distancia
    #verificar que dicha distancia sea menor a un umbral
    #si todos cumplen con la condicion anterior aumentar un contador de frames quietos
    #si el contador de framesquietos syupera un umbral dadom, se detecta la situacion buscada: dados quietos.



    cv2.imshow('Frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    else:
        break

cap.release()
cv2.destroyAllWindows()