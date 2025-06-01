import cv2
import mediapipe as mp
import csv
import os

# Inicializar MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Etiqueta de emoción actual
current_emotion = "soprendido"

# Crear CSV si no existe
csv_filename = 'datos_emociones.csv'
file_exists = os.path.isfile(csv_filename)

csv_file = open(csv_filename, mode='a', newline='')
csv_writer = csv.writer(csv_file)

# Escribir encabezados si es nuevo
if not file_exists:
    header = ['emoción']
    for i in range(468):
        header += [f'x{i}', f'y{i}', f'z{i}']
    csv_writer.writerow(header)

# Captura de video
cap = cv2.VideoCapture(0)

print("Presiona 'g' para guardar la emoción actual:", current_emotion)
print("Presiona 'q' para salir.")

# Contador de muestras por emoción
contador_emociones = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    key = cv2.waitKey(1) & 0xFF

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Dibujar puntos faciales (blancos y pequeños)
            for lm in face_landmarks.landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)  # Blanco, pequeño

            if key == ord('g'):
                data = [current_emotion]
                for lm in face_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])
                csv_writer.writerow(data)

                # Incrementar contador
                if current_emotion not in contador_emociones:
                    contador_emociones[current_emotion] = 0
                contador_emociones[current_emotion] += 1

                print(f"✔ Emoción '{current_emotion}' guardada. Total: {contador_emociones[current_emotion]}")

    cv2.imshow('Recolector de emociones', frame)

    if key == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
