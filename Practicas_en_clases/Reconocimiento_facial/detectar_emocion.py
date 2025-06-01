import cv2
import mediapipe as mp
import joblib
import numpy as np
import time

# Cargar modelo y codificador de emociones
model = joblib.load("modelo_emociones.pkl")
label_encoder = joblib.load("emociones_encoder.pkl")

# Inicializar MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Captura de video
cap = cv2.VideoCapture(0)

# ======= PARPADEO CONFIG =======
def calcular_EAR(ojos):
    A = np.linalg.norm(np.array(ojos[1]) - np.array(ojos[5]))
    B = np.linalg.norm(np.array(ojos[2]) - np.array(ojos[4]))
    C = np.linalg.norm(np.array(ojos[0]) - np.array(ojos[3]))
    EAR = (A + B) / (2.0 * C)
    return EAR

ojo_derecho_idx = [33, 160, 158, 133, 153, 144]
EAR_UMBRAL = 0.23
frames_parpadeo = 0
parpadeos_detectados = 0
inicio_timer = time.time()
parpadeo_confirmado = False
mensaje_parpadeo = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Dibujar puntos faciales (blancos y pequeños)
        for lm in face_landmarks.landmark:
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

        # === DETECCIÓN DE PARPADEO ===
        ojo_derecho = []
        for idx in ojo_derecho_idx:
            lm = face_landmarks.landmark[idx]
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            ojo_derecho.append((x, y))

        if len(ojo_derecho) == 6:
            EAR = calcular_EAR(ojo_derecho)
            if EAR < EAR_UMBRAL:
                frames_parpadeo += 1
            else:
                if frames_parpadeo >= 2:
                    parpadeos_detectados += 1
                    parpadeo_confirmado = True
                frames_parpadeo = 0

        # Mostrar mensaje de parpadeo cada 5 segundos
        tiempo_actual = time.time()
        if tiempo_actual - inicio_timer >= 5:
            if parpadeo_confirmado:
                mensaje_parpadeo = "✅ Rostro real (parpadeo detectado)"
            else:
                mensaje_parpadeo = "⚠ Posible imagen o máscara (sin parpadeo)"
            inicio_timer = tiempo_actual
            parpadeo_confirmado = False

        # === PREDICCIÓN DE EMOCIÓN ===
        landmarks = []
        for lm in face_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        X = np.array(landmarks).reshape(1, -1)
        pred = model.predict(X)
        emocion = label_encoder.inverse_transform(pred)[0]

        # Mostrar resultados
        cv2.putText(frame, f'Emocion: {emocion}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, mensaje_parpadeo, (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Detector de Emociones + Parpadeo', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
