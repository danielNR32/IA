import pygame
import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier



velocidad_bala_vertical =22
velocidad_bala_horizontal_min = -12
velocidad_bala_horizontal_max = -12



# Esta función entrena una red neuronal para predecir cuándo saltar
def entrenar_modelo_master_chief(registro_saltos):
    if len(registro_saltos) < 15:
        print(f"[INFO] Pocos datos para entrenar el modelo de MasterChief. Datos actuales: {len(registro_saltos)}")
        return None
    datos = np.array(registro_saltos)
    X = datos[:, :6]
    y = datos[:, 6]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo_pred_salto = Sequential([
        Dense(32, input_dim=6, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    modelo_pred_salto.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    modelo_pred_salto.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
    loss, accuracy = modelo_pred_salto.evaluate(X_test, y_test, verbose=0)
    print(f"[INFO] Modelo de MasterChief entrenado con precisión: {accuracy:.4f} (loss: {loss:.4f}, datos de test: {len(y_test)})")
    return modelo_pred_salto

# Entrena un árbol de decisión para el salto de MasterChief
def entrenar_arbol_salto_master_chief(registro_saltos):
    if len(registro_saltos) < 15:
        print(f"[INFO] Pocos datos para entrenar el árbol de salto de MasterChief. Datos actuales: {len(registro_saltos)}")
        return None
    datos = np.array(registro_saltos)
    X = datos[:, :6]
    y = datos[:, 6]
    arbol_salto = DecisionTreeClassifier(max_depth=6)
    arbol_salto.fit(X, y)
    print(f"[INFO] Árbol de salto de MasterChief entrenado con {len(y)} datos.")
    return arbol_salto

# Entrena un modelo KNN para el salto de MasterChief
def entrenar_KNN_salto_master_chief(registro_saltos, n_neighbors=3):
    if len(registro_saltos) < n_neighbors:
        print(f"[INFO] Pocos datos para entrenar el KNN de MasterChief. Datos actuales: {len(registro_saltos)}")
        return None
    datos = np.array(registro_saltos)
    X = datos[:, :6]
    y = datos[:, 6]
    KNN_salto = KNeighborsClassifier(n_neighbors=n_neighbors)
    KNN_salto.fit(X, y)
    print(f"[INFO] KNN de MasterChief entrenado con {len(y)} datos.")
    return KNN_salto

# Esta función decide si MasterChief debe saltar según la predicción de la red
def decidir_salto_master_chief(jugador_halo, plasma, vel_bala_predicha, plasma_misil, plasma_misil_activa, modelo_pred_salto, en_salto, en_suelo):
    if modelo_pred_salto is None:
        print("[WARN] Modelo de MasterChief no entrenado. No se puede decidir salto.")
        return False, en_suelo
    distancia_terreno = abs(jugador_halo.x - plasma.x)
    delta_x_aerea = abs(jugador_halo.centerx - plasma_misil.centerx)
    delta_y_aerea = abs(jugador_halo.centery - plasma_misil.centery)
    hay_proyectil_aire = 1 if plasma_misil_activa else 0
    datos_entrada_pred = np.array([[vel_bala_predicha, distancia_terreno, delta_x_aerea, delta_y_aerea, hay_proyectil_aire, jugador_halo.x]])
    resultado_salto = modelo_pred_salto.predict(datos_entrada_pred, verbose=0)[0][0]
    print(f"[INFO] Decisión de salto: predicción={resultado_salto:.4f}, en_suelo={en_suelo}, salto_actual={en_salto}, entrada={datos_entrada_pred.tolist()}")
    if resultado_salto > 0.5 and en_suelo:
        en_salto = True
        en_suelo = False
        print(f"[ACTION] MasterChief salta (predicción={resultado_salto:.4f}, distancia_terreno={distancia_terreno}, delta_x_aerea={delta_x_aerea}, delta_y_aerea={delta_y_aerea})")
    return en_salto, en_suelo

# Decide si MasterChief debe saltar usando el árbol de decisión
def decidir_salto_master_chief_arbol(jugador_halo, plasma, vel_bala_predicha, plasma_misil, plasma_misil_activa, arbol_salto, en_salto, en_suelo):
    if arbol_salto is None:
        print("[WARN] Árbol de salto de MasterChief no entrenado. No se puede decidir salto.")
        return False, en_suelo
    distancia_terreno = abs(jugador_halo.x - plasma.x)
    delta_x_aerea = abs(jugador_halo.centerx - plasma_misil.centerx)
    delta_y_aerea = abs(jugador_halo.centery - plasma_misil.centery)
    hay_proyectil_aire = 1 if plasma_misil_activa else 0
    datos_entrada_pred = np.array([[vel_bala_predicha, distancia_terreno, delta_x_aerea, delta_y_aerea, hay_proyectil_aire, jugador_halo.x]])
    prediccion = arbol_salto.predict(datos_entrada_pred)[0]
    print(f"[INFO][ÁRBOL] Decisión de salto: predicción={prediccion}, en_suelo={en_suelo}, salto_actual={en_salto}, entrada={datos_entrada_pred.tolist()}")
    if prediccion == 1 and en_suelo:
        en_salto = True
        en_suelo = False
        print(f"[ACTION][ÁRBOL] MasterChief salta (distancia_terreno={distancia_terreno}, delta_x_aerea={delta_x_aerea}, delta_y_aerea={delta_y_aerea})")
    return en_salto, en_suelo

# Decide si MasterChief debe saltar usando el modelo KNN
def decidir_salto_master_chief_KNN(jugador_halo, plasma, vel_bala_predicha, plasma_misil, plasma_misil_activa, KNN_salto, en_salto, en_suelo):
    if KNN_salto is None:
        print("[WARN] KNN de salto de MasterChief no entrenado. No se puede decidir salto.")
        return False, en_suelo
    distancia_terreno = abs(jugador_halo.x - plasma.x)
    delta_x_aerea = abs(jugador_halo.centerx - plasma_misil.centerx)
    delta_y_aerea = abs(jugador_halo.centery - plasma_misil.centery)
    hay_proyectil_aire = 1 if plasma_misil_activa else 0
    datos_entrada_pred = np.array([[vel_bala_predicha, distancia_terreno, delta_x_aerea, delta_y_aerea, hay_proyectil_aire, jugador_halo.x]])
    prediccion = KNN_salto.predict(datos_entrada_pred)[0]
    print(f"[INFO][KNN] Decisión de salto: predicción={prediccion}, en_suelo={en_suelo}, salto_actual={en_salto}, entrada={datos_entrada_pred.tolist()}")
    if prediccion == 1 and en_suelo:
        en_salto = True
        en_suelo = False
        print(f"[ACTION][KNN] MasterChief salta (distancia_terreno={distancia_terreno}, delta_x_aerea={delta_x_aerea}, delta_y_aerea={delta_y_aerea})")
    return en_salto, en_suelo

# Esta función entrena una red neuronal para el movimiento lateral de MasterChief (binaria: 0=izquierda, 1=derecha)
def entrenar_movimiento_master_chief(registro_movimientos):
    if len(registro_movimientos) < 10:
        print(f"[INFO] No hay suficientes datos para entrenar el movimiento de MasterChief. Datos actuales: {len(registro_movimientos)}")
        return None
    datos = np.array(registro_movimientos)
    X = datos[:, :8].astype('float32')
    y = datos[:, 8].astype('int')
    mask = (y == 0) | (y == 2)
    X_bin = X[mask]
    y_bin = y[mask]
    y_bin = np.where(y_bin == 2, 1, 0)
    if len(y_bin) < 10:
        print(f"[INFO] No hay suficientes datos de izquierda/derecha para entrenar la red binaria. Datos actuales: {len(y_bin)}")
        return None
    X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42)
    modelo_pred_mov = Sequential([
        Dense(32, input_dim=8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    modelo_pred_mov.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    modelo_pred_mov.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)
    loss, accuracy = modelo_pred_mov.evaluate(X_test, y_test, verbose=0)
    print(f"[INFO] Precisión del modelo de movimiento binario de MasterChief: {accuracy:.4f} (loss: {loss:.4f}, datos de test: {len(y_test)})")
    return modelo_pred_mov

# Entrena un árbol de decisión para el movimiento lateral de MasterChief
def entrenar_arbol_movimiento_master_chief(registro_movimientos):
    if len(registro_movimientos) < 10:
        print(f"[INFO] No hay suficientes datos para entrenar el árbol de movimiento de MasterChief. Datos actuales: {len(registro_movimientos)}")
        return None
    datos = np.array(registro_movimientos)
    X = datos[:, :8].astype('float32')
    y = datos[:, 8].astype('int')
    arbol_movimiento = DecisionTreeClassifier(max_depth=6)
    arbol_movimiento.fit(X, y)
    print(f"[INFO] Árbol de movimiento de MasterChief entrenado con {len(y)} datos.")
    return arbol_movimiento

# Entrena un modelo KNN para el movimiento lateral de MasterChief
def entrenar_KNN_movimiento_master_chief(registro_movimientos, n_neighbors=3):
    if len(registro_movimientos) < n_neighbors:
        print(f"[INFO] No hay suficientes datos para entrenar el KNN de movimiento de MasterChief. Datos actuales: {len(registro_movimientos)}")
        return None
    datos = np.array(registro_movimientos)
    X = datos[:, :8].astype('float32')
    y = datos[:, 8].astype('int')
    KNN_movimiento = KNeighborsClassifier(n_neighbors=n_neighbors)
    KNN_movimiento.fit(X, y)
    print(f"[INFO] KNN de movimiento de MasterChief entrenado con {len(y)} datos.")
    return KNN_movimiento

# Esta función decide el movimiento lateral de MasterChief (izquierda, estatico, derecha) usando la red binaria
def decidir_movimiento_master_chief(jugador_halo, plasma_misil, modelo_pred_mov, en_salto, plasma):
    if modelo_pred_mov is None:
        print("[WARN] Modelo de movimiento de MasterChief no entrenado.")
        return jugador_halo.x, 1
    distancia_proyectil_suelo = abs(jugador_halo.x - plasma.x)
    entrada_movimiento = np.array([[
        jugador_halo.x,
        jugador_halo.y,
        plasma_misil.centerx,
        plasma_misil.centery,
        plasma.x,
        plasma.y,
        distancia_proyectil_suelo,
        1 if en_salto else 0
    ]], dtype='float32')
    prediccion = modelo_pred_mov.predict(entrada_movimiento, verbose=0)[0][0]
    # 0=izquierda, 1=derecha, quedarse quieto si la predicción está cerca de 0.5
    accion_master_chief = 1  # por default quieto
    if prediccion < 0.4 and jugador_halo.x > 0:
        jugador_halo.x -= 5
        accion_master_chief = 0
        print(f"[ACTION][NN] MasterChief se mueve a la izquierda (x={jugador_halo.x}) pred={prediccion:.3f}")
    elif prediccion > 0.6 and jugador_halo.x < 200 - jugador_halo.width:
        jugador_halo.x += 5
        accion_master_chief = 2
        print(f"[ACTION][NN] MasterChief se mueve a la derecha (x={jugador_halo.x}) pred={prediccion:.3f}")
    else:
        print(f"[ACTION][NN] MasterChief se queda quieto (x={jugador_halo.x}) pred={prediccion:.3f}")
    return jugador_halo.x, accion_master_chief

# Decide el movimiento lateral de MasterChief usando el árbol de decisión
def decidir_movimiento_master_chief_arbol(jugador_halo, plasma_misil, arbol_movimiento, en_salto, plasma):
    if arbol_movimiento is None:
        print("[WARN] Árbol de movimiento de MasterChief no entrenado.")
        return jugador_halo.x, 1
    distancia_proyectil_suelo = abs(jugador_halo.x - plasma.x)
    entrada_movimiento = np.array([[jugador_halo.x, jugador_halo.y, plasma_misil.centerx, plasma_misil.centery, plasma.x, plasma.y, distancia_proyectil_suelo, 1 if en_salto else 0]], dtype='float32')
    accion_master_chief = arbol_movimiento.predict(entrada_movimiento)[0]
    print(f"[INFO][ÁRBOL] Decisión movimiento: acción={accion_master_chief}, entrada={entrada_movimiento.tolist()}")
    if accion_master_chief == 0 and jugador_halo.x > 0:
        jugador_halo.x -= 5
        print(f"[ACTION][ÁRBOL] MasterChief se mueve a la izquierda (x={jugador_halo.x})")
    elif accion_master_chief == 2 and jugador_halo.x < 200 - jugador_halo.width:
        jugador_halo.x += 5
        print(f"[ACTION][ÁRBOL] MasterChief se mueve a la derecha (x={jugador_halo.x})")
    else:
        print(f"[ACTION][ÁRBOL] MasterChief se queda quieto (x={jugador_halo.x})")
    return jugador_halo.x, accion_master_chief

# Decide el movimiento lateral de MasterChief usando el modelo KNN
def decidir_movimiento_master_chief_KNN(jugador_halo, plasma_misil, KNN_movimiento, en_salto, plasma):
    if KNN_movimiento is None:
        print("[WARN] KNN de movimiento de MasterChief no entrenado.")
        return jugador_halo.x, 1
    distancia_proyectil_suelo = abs(jugador_halo.x - plasma.x)
    entrada_movimiento = np.array([[jugador_halo.x, jugador_halo.y, plasma_misil.centerx, plasma_misil.centery, plasma.x, plasma.y, distancia_proyectil_suelo, 1 if en_salto else 0]], dtype='float32')
    accion_master_chief = KNN_movimiento.predict(entrada_movimiento)[0]
    print(f"[INFO][KNN] Decisión movimiento: acción={accion_master_chief}, entrada={entrada_movimiento.tolist()}")
    if accion_master_chief == 0 and jugador_halo.x > 0:
        jugador_halo.x -= 5
        print(f"[ACTION][KNN] MasterChief se mueve a la izquierda (x={jugador_halo.x})")
    elif accion_master_chief == 2 and jugador_halo.x < 200 - jugador_halo.width:
        jugador_halo.x += 5
        print(f"[ACTION][KNN] MasterChief se mueve a la derecha (x={jugador_halo.x})")
    else:
        print(f"[ACTION][KNN] MasterChief se queda quieto (x={jugador_halo.x})")
    return jugador_halo.x, accion_master_chief

# Inicializa pygame y la ventana principal
pygame.init()
base_path = os.path.dirname(os.path.abspath(__file__))
w, h = 800, 400
pantalla_juego = pygame.display.set_mode((w, h))
pygame.display.set_caption("Juego de MasterChief: Esquivar proyectiles")
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
ROJO = (255, 0, 0)
AZUL = (0, 0, 255)
jugador_halo = None
plasma = None
plasma_misil = None
fondo_master_chief = None
enemigo_elite = None
menu_master_chief = None
en_salto = False
altura_salto_master_chief = 15
gravedad_master_chief = 1
pie_tierra = True
trayecto_ascendente = True
pausa_master_chief = False
tipografia = pygame.font.SysFont('Arial', 24)
menu_activo = True
modo_nn = False
modo_arbol = False
modo_KNN = False
registro_saltos = []
modelo_binario_salto = None
arbol_decision_salto = None
modelo_KNN_salto = None
registro_movimientos = []
modelo_binario_mov = []
arbol_decision_mov = None
modelo_KNN_mov = None
intervalo_decidir_salto_master_chief = 1
contador_salto_master_chief = 0
frame_actual_master_chief = 0
velocidad_animacion = 10
contador_frames = 0
vel_bala_inferior = -10

# Carga los sprites de MasterChief, proyectiles y enemigos
master_chief_frames = [
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/JEFE MAESTRO/1.png')), (48, 48)),
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/JEFE MAESTRO/1.png')), (48, 48)),
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/JEFE MAESTRO/2.png')), (48, 48)),
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/JEFE MAESTRO/3.png')), (48, 48)),
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/JEFE MAESTRO/4.png')), (48, 48)),
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/JEFE MAESTRO/5.png')), (48, 48)),
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/JEFE MAESTRO/6.png')), (48, 48)),
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/JEFE MAESTRO/7.png')), (48, 48)),
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/JEFE MAESTRO/8.png')), (48, 48)),
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/JEFE MAESTRO/9.png')), (48, 48)),
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/JEFE MAESTRO/10.png')), (48, 48)),
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/JEFE MAESTRO/11.png')), (48, 48)),
]
bala_de_plasma_frames = [
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/BALA/vertical.png')), (40, 40)),
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/BALA/izquierda.png')), (40, 40))
]
elite_frames = [
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/ELITE/1.png')), (80, 80)),
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/ELITE/2.png')), (80, 80)),
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/ELITE/3.png')), (80, 80)),
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/ELITE/4.png')), (80, 80)),
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/ELITE/5.png')), (80, 80)),
    pygame.transform.scale(pygame.image.load(os.path.join(base_path, 'assets/sprites/ELITE/6.png')), (80, 80))
]

fondo_scroll = pygame.image.load(os.path.join(base_path, 'assets/sprites/halo.png'))
fondo_scroll = pygame.transform.scale(fondo_scroll, (w, h))
jugador_halo = pygame.Rect(50, h - 100, 32, 48)
plasma = pygame.Rect(w - 50, h - 90, 16, 16)
plasma_misil = pygame.Rect(0, -50, 16, 16)
enemigo_elite = pygame.Rect(w - 100, h - 130, 64, 64)
bala_vertical = pygame.Rect(0, 0, 64, 64)
velocidad_proyectil_aire = [0, 5]
menu_rectangular = pygame.Rect(w // 2 - 135, h // 2 - 90, 270, 180)
plasma_activa = False
plasma_misil_activa = False
fondo_x1_master_chief = 0
fondo_x2_master_chief = w
ultimo_disparo_aire = 0
direccion_enemigo = 1
velocidad_enemigo = 5
cooldown_disparo = 0
intervalo_disparo = 60

# Controla el disparo vertical
def mover_bala_vertical():
    global bala_vertical, direccion_enemigo, cooldown_disparo
    bala_vertical.x += direccion_enemigo * velocidad_enemigo
    cooldown_disparo -= 1
    if bala_vertical.x <= 0 or bala_vertical.x >= 200 - bala_vertical.width:
        direccion_enemigo *= -1

# Dispara un proyectil aéreo desde una posición aleatoria
def disparar_bala_vertical():
    global plasma_misil, plasma_misil_activa, velocidad_proyectil_aire, ultimo_disparo_aire, cooldown_disparo
    if not plasma_misil_activa and cooldown_disparo <= 0:
        plasma_misil.x = jugador_halo.centerx - plasma_misil.width // 2
        plasma_misil.y = -plasma_misil.height  # inicia fuera de pantalla arriba
        velocidad_proyectil_aire[0] = 0
        velocidad_proyectil_aire[1] = velocidad_bala_vertical  # velocidad hacia abajo
        plasma_misil_activa = True
        cooldown_disparo = intervalo_disparo
        ultimo_disparo_aire = pygame.time.get_ticks()
        print(f"[DISPARO AÉREO] Bala lanzada desde x={plasma_misil.x}")

# Dispara un proyectil por el suelo desde la derecha de la pantalla
def disparar_proyectil_suelo():
    global plasma_activa, vel_bala_inferior
    if not plasma_activa:
        # Asegura que sea más rápida que la bala vertical (ej: si vertical = 5, horizontal será entre -9 y -6)
        vel_bala_inferior = random.randint(velocidad_bala_horizontal_min,velocidad_bala_horizontal_max)
        plasma_activa = True

# Controla el movimiento de MasterChief con las teclas y calcula la posición relativa al proyectil aéreo
def mover_master_chief_manual():
    global jugador_halo, pie_tierra, en_salto, accion_horizontal
    keys = pygame.key.get_pressed()
    accion_horizontal = 1
    if keys[pygame.K_LEFT] and jugador_halo.x > 0:
        jugador_halo.x -= 5
        accion_horizontal = 0
    if keys[pygame.K_RIGHT] and jugador_halo.x < 200 - jugador_halo.width:
        jugador_halo.x += 5
        accion_horizontal = 2
    if keys[pygame.K_UP] and pie_tierra:
        en_salto = True
        pie_tierra = False
    delta_horizontal = (jugador_halo.centerx - plasma_misil.centerx)
    delta_vertical = (jugador_halo.centery - plasma_misil.centery)
    distancia_directa = (delta_horizontal**2 + delta_vertical**2) ** 0.5
    print(f"[INFO] MasterChief(x={jugador_halo.x}, y={jugador_halo.y}) | plasma_proyectil(x={plasma_misil.centerx}, y={plasma_misil.centery}) | DistanciaX={delta_horizontal} | DistanciaY={delta_vertical} | Velocidadplasma_proyectil={velocidad_proyectil_aire} | Saltando={en_salto} | EnSuelo={pie_tierra}", end="\r")

# Guarda los datos de movimiento para entrenamiento y análisis
def mover_master_chief_automatico(modelo_pred_mov):
    global jugador_halo, accion_horizontal
    jugador_halo.x, accion_horizontal = decidir_movimiento_master_chief(jugador_halo, plasma_misil, modelo_pred_mov, en_salto)
    delta_horizontal = jugador_halo.centerx - plasma_misil.centerx
    delta_vertical = jugador_halo.centery - plasma_misil.centery
    registro_movimientos.append((delta_horizontal, delta_vertical, jugador_halo.x, plasma_misil.centerx, accion_horizontal))

# Reinicia el proyectil del suelo a la posición inicial
def reset_proyectil_suelo():
    global plasma, plasma_activa
    plasma.x = w - 50
    plasma_activa = False

# Reinicia el proyectil aéreo a la posición inicial
def reset_proyectil_aire():
    global plasma_misil, plasma_misil_activa
    plasma_misil.y = -50
    plasma_misil_activa = False

# Controla la física del salto de MasterChief
def manejar_salto_master_chief():
    global jugador_halo, en_salto, altura_salto_master_chief, gravedad_master_chief, pie_tierra, trayecto_ascendente
    if en_salto:
        if trayecto_ascendente:
            jugador_halo.y -= altura_salto_master_chief
            altura_salto_master_chief -= gravedad_master_chief
            if altura_salto_master_chief <= 0:
                trayecto_ascendente = False
        else:
            jugador_halo.y += altura_salto_master_chief
            altura_salto_master_chief += gravedad_master_chief
            if jugador_halo.y >= h - 100:
                jugador_halo.y = h - 100
                en_salto = False
                altura_salto_master_chief = 15
                trayecto_ascendente = True
                pie_tierra = True

# Actualiza la pantalla, mueve los elementos y detecta colisiones
def actualizar_juego_mc():
    global plasma, plasma_misil, frame_actual_master_chief, contador_frames, fondo_x1_master_chief, fondo_x2_master_chief
    mover_bala_vertical()
    fondo_x1_master_chief -= 3
    fondo_x2_master_chief -= 3

    if fondo_x1_master_chief <= -w:
        fondo_x1_master_chief = w
    if fondo_x2_master_chief <= -w:
        fondo_x2_master_chief = w

    pantalla_juego.blit(fondo_scroll, (fondo_x1_master_chief, 0))
    pantalla_juego.blit(fondo_scroll, (fondo_x2_master_chief, 0))

    # Dibuja a MasterChief (animación)
    if en_salto:
        if trayecto_ascendente:
            pantalla_juego.blit(master_chief_frames[0], (jugador_halo.x, jugador_halo.y))
        else:
            pantalla_juego.blit(master_chief_frames[1], (jugador_halo.x, jugador_halo.y))
    else:
        contador_frames += 10
        if contador_frames >= velocidad_animacion:
            frame_actual_master_chief = (frame_actual_master_chief + 1) % len(master_chief_frames)
            contador_frames = 0
        pantalla_juego.blit(master_chief_frames[frame_actual_master_chief], (jugador_halo.x, jugador_halo.y))

    # Dibuja al enemigo (elite)
    pantalla_juego.blit(elite_frames[frame_actual_master_chief % len(elite_frames)], (enemigo_elite.x, enemigo_elite.y))

    # Proyectil horizontal (suelo) con imagen fija
    if plasma_activa:
        plasma.x += vel_bala_inferior
        pantalla_juego.blit(bala_de_plasma_frames[1], (plasma.x, plasma.y))  # izquierda.png

    # Proyectil vertical (aire) con imagen fija
    if plasma_misil_activa:
        plasma_misil.x += velocidad_proyectil_aire[0]
        plasma_misil.y += velocidad_proyectil_aire[1]
        pantalla_juego.blit(bala_de_plasma_frames[0], (plasma_misil.x, plasma_misil.y))  # vertical.png

    # Reseteo si los proyectiles se salen de pantalla
    if plasma.x < 0:
        reset_proyectil_suelo()
    if plasma_misil.y > h:
        reset_proyectil_aire()
        disparar_bala_vertical()  # lanza una nueva hacia la nueva posición del jugador


    # Colisiones
    if jugador_halo.colliderect(plasma) or jugador_halo.colliderect(plasma_misil):
        print(f"[GAME OVER] MasterChief muerto Posición MasterChief: (x={jugador_halo.x}, y={jugador_halo.y}), Proyectil suelo: (x={plasma.x}, y={plasma.y}), Proyectil aire: (x={plasma_misil.x}, y={plasma_misil.y})")
        guardar_datos_csv()
        reiniciar_juego_mc()


# Guarda los datos de cada frame para entrenamiento y análisis
def guardar_datos_master_chief():
    global jugador_halo, plasma, vel_bala_inferior, en_salto
    distancia_terreno = abs(jugador_halo.x - plasma.x)
    salto_realizado = 1 if en_salto else 0
    delta_x_aerea = abs(jugador_halo.centerx - plasma_misil.centerx)
    delta_y_aerea = abs(jugador_halo.centery - plasma_misil.centery)
    hay_proyectil_aire = 1 if plasma_misil_activa else 0
    registro_saltos.append((
        vel_bala_inferior,
        distancia_terreno,
        delta_x_aerea,
        delta_y_aerea,
        hay_proyectil_aire,
        jugador_halo.x,
        salto_realizado,
        
    ))
    distancia_proyectil_suelo = abs(jugador_halo.x - plasma.x)
    registro_movimientos.append((
        jugador_halo.x,
        jugador_halo.y,
        plasma_misil.centerx,
        plasma_misil.centery,
        plasma.x,
        plasma.y,
        distancia_proyectil_suelo,
        1 if en_salto else 0,
        accion_horizontal
    ))

# Pausa el juego y muestra los datos recolectados
def alternar_pausa():
    global pausa_master_chief
    pausa_master_chief = not pausa_master_chief
    if pausa_master_chief:
        imprimir_datos_registro()
    else:
        print("Juego de MasterChief reanudado.")

# Dibuja un botón rectangular con texto centrado
def dibujar_boton_master_chief(rect, texto, color, color_texto=NEGRO):
    pygame.draw.rect(pantalla_juego, color, rect)
    pygame.draw.rect(pantalla_juego, NEGRO, rect, 2)
    texto_master_chief = tipografia.render(texto, True, color_texto)
    rect_texto = texto_master_chief.get_rect(center=rect.center)
    pantalla_juego.blit(texto_master_chief, rect_texto)

def mostrar_menu_halo():
    global menu_activo, modo_nn, modo_arbol, modo_KNN
    global registro_saltos, modelo_binario_salto, modelo_binario_mov
    global arbol_decision_salto, arbol_decision_mov
    global registro_movimientos, modelo_KNN_salto, modelo_KNN_mov

    fondo_menu = pygame.image.load(os.path.join(base_path, 'assets/sprites/menu_halo.png'))
    fondo_menu = pygame.transform.scale(fondo_menu, (w, h))
    pygame.mixer.music.load(os.path.join(base_path, 'assets/sprites/music/start_cancion.mp3'))
    pygame.mixer.music.set_volume(0.5)
    pygame.mixer.music.play(-1)

    try:
        fuente_halo = pygame.font.Font(os.path.join(base_path, 'assets/sprites/halo_font.ttf'), 24)
    except:
        fuente_halo = pygame.font.SysFont('Arial', 24)

    etiquetas = [
        {"label": "Modo Manual Halo", "submenu": [], "action": "manual"},
        {"label": "Opciones de Modelo", "submenu": [
            {"label": "Modo Automático Árbol", "action": "arbol"},
            {"label": "Modo Automático KNN", "action": "KNN"},
            {"label": "Modo Automático NN", "action": "nn"},
        ]},
        {"label": "Opciones de Datos", "submenu": [
            {"label": "Entrenar Modelos", "action": "train"},
            {"label": "Borrar Datos", "action": "clear"},
        ]}
    ]

    button_width, button_height = 220, 40
    spacing = 20
    start_x = (w - (len(etiquetas) * (button_width + spacing) - spacing)) // 2
    button_y = h - 60
    selected_index = -1

    while menu_activo:
        pantalla_juego.blit(fondo_menu, (0, 0))
        mouse_pos = pygame.mouse.get_pos()
        mouse_click = False

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                mouse_click = True

        for i, button in enumerate(etiquetas):
            rect = pygame.Rect(start_x + i * (button_width + spacing), button_y, button_width, button_height)
            hover = rect.collidepoint(mouse_pos)
            if hover:
                pygame.draw.rect(pantalla_juego, (30, 100, 255), rect)
            text = fuente_halo.render(button["label"], True, (255, 255, 255) if hover else (180, 210, 255))
            pantalla_juego.blit(text, text.get_rect(center=rect.center))

            if hover and mouse_click:
                selected_index = i if selected_index != i else -1
                if not button["submenu"]:  # Ejecutar acción directa
                    if button["action"] == "manual":
                        modo_nn = modo_arbol = modo_KNN = False
                        menu_activo = False

        if 0 <= selected_index < len(etiquetas):
            submenu = etiquetas[selected_index]["submenu"]
            for j, item in enumerate(submenu):
                submenu_rect = pygame.Rect(
                    start_x + selected_index * (button_width + spacing),
                    button_y - (j + 1) * (button_height + 5),
                    button_width, button_height
                )
                hover = submenu_rect.collidepoint(mouse_pos)
                if hover:
                    pygame.draw.rect(pantalla_juego, (30, 100, 255), submenu_rect)
                text = fuente_halo.render(item["label"], True, (255, 255, 255) if hover else (180, 210, 255))
                pantalla_juego.blit(text, text.get_rect(center=submenu_rect.center))

                if hover and mouse_click:
                    action = item["action"]
                    selected_index = -1
                    if action == "arbol":
                        modo_nn = False
                        modo_arbol = True
                        modo_KNN = False
                        menu_activo = False
                    elif action == "KNN":
                        modo_nn = False
                        modo_arbol = False
                        modo_KNN = True
                        menu_activo = False
                    elif action == "nn":
                        modo_nn = True
                        modo_arbol = modo_KNN = False
                        menu_activo = False
                    elif action == "train":
                        modelo_binario_salto = entrenar_modelo_master_chief(registro_saltos)
                        modelo_binario_mov = entrenar_movimiento_master_chief(registro_movimientos)
                        arbol_decision_salto = entrenar_arbol_salto_master_chief(registro_saltos)
                        arbol_decision_mov = entrenar_arbol_movimiento_master_chief(registro_movimientos)
                        modelo_KNN_salto = entrenar_KNN_salto_master_chief(registro_saltos)
                        modelo_KNN_mov = entrenar_KNN_movimiento_master_chief(registro_movimientos)
                    elif action == "clear":
                        registro_saltos.clear()
                        registro_movimientos.clear()
                        borrar_datos_csv()
                        print("[INFO]Datos eliminados.")

        pygame.display.flip()


# Borra los archivos CSV si existen
def borrar_datos_csv():
    ruta_guardado = os.path.join(os.path.dirname(__file__), "csv")
    salto_path = os.path.join(ruta_guardado, "datos_saltos.csv")
    mov_path = os.path.join(ruta_guardado, "datos_movimientos.csv")

    if os.path.exists(salto_path):
        os.remove(salto_path)
        print("[INFO] Archivo 'datos_saltos.csv' eliminado.")

    if os.path.exists(mov_path):
        os.remove(mov_path)
        print("[INFO] Archivo 'datos_movimientos.csv' eliminado.")

# Reinicia el estado del juego y muestra el menú
def reiniciar_juego_mc():
    global menu_activo, jugador_halo, plasma, plasma_misil, enemigo_elite, plasma_activa, plasma_misil_activa, en_salto, pie_tierra
    menu_activo = True
    jugador_halo.x, jugador_halo.y = 50, h - 100
    plasma.x = w - 50
    plasma_misil.y = -50
    enemigo_elite.x, enemigo_elite.y = w - 100, h - 100
    plasma_activa = False
    plasma_misil_activa = False
    en_salto = False
    pie_tierra = True
    imprimir_datos_registro()
    mostrar_menu_halo()

# Imprime los datos de movimiento recolectados
def imprimir_datos_registro():
    print("[DATA] Datos de movimiento recolectados:")
    for i, dato in enumerate(registro_movimientos):
        print(f"  [{i}] {dato}")

# Guarda los datos en archivos CSV
import pandas as pd
import os

def guardar_datos_csv():
    ruta_guardado = os.path.join(os.path.dirname(__file__), "csv")

    # Crear carpeta si no existe
    os.makedirs(ruta_guardado, exist_ok=True)

    # Guardar datos de salto
    salto_df = pd.DataFrame(registro_saltos, columns=[
        "vel_bala", "dist_terreno", "delta_x_aerea", "delta_y_aerea",
        "hay_proyectil_aire", "pos_x_jugador", "salto"
    ])
    salto_path = os.path.join(ruta_guardado, "datos_saltos.csv")
    salto_df.to_csv(salto_path, index=False)
    print(f"[INFO] Datos de salto guardados en '{salto_path}'")

    # Guardar datos de movimiento
    mov_df = pd.DataFrame(registro_movimientos, columns=[
        "x_jugador", "y_jugador", "x_bala_aire", "y_bala_aire",
        "x_bala_suelo", "y_bala_suelo", "dist_bala_suelo", "en_salto", "accion"
    ])
    mov_path = os.path.join(ruta_guardado, "datos_movimientos.csv")
    mov_df.to_csv(mov_path, index=False)
    print(f"[INFO] Datos de movimiento guardados en '{mov_path}'")

# Carga los datos desde CSV si existen
def cargar_datos_csv():
    global registro_saltos, registro_movimientos
    if os.path.exists("datos_saltos.csv"):
        salto_df = pd.read_csv("datos_saltos.csv")
        registro_saltos = salto_df.values.tolist()
        print(f"[INFO] Cargados {len(registro_saltos)} registros de salto desde 'datos_saltos.csv'")
    if os.path.exists("datos_movimientos.csv"):
        mov_df = pd.read_csv("datos_movimientos.csv")
        registro_movimientos = mov_df.values.tolist()
        print(f"[INFO] Cargados {len(registro_movimientos)} registros de movimiento desde 'datos_movimientos.csv'")

# Bucle principal del juego
def main_game_loop():
    global en_salto, pie_tierra, plasma_activa, plasma_misil_activa, contador_salto_master_chief, modo_arbol, modo_KNN
    reloj_master_chief = pygame.time.Clock()
    #carga los datos desde CSV si existen
    cargar_datos_csv()
    # Inicializa el menu principal
    mostrar_menu_halo()
    juego_activo = True
    while juego_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                guardar_datos_csv()
                juego_activo = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and pie_tierra and not pausa_master_chief:
                    en_salto = True
                    pie_tierra = False
                if evento.key == pygame.K_p:
                    alternar_pausa()
                if evento.key == pygame.K_q:
                    imprimir_datos_registro()
                    guardar_datos_csv()
                    pygame.quit()
                    exit()
                if evento.key == pygame.K_ESCAPE:  # Tecla ESC para volver al menú
                    guardar_datos_csv()
                    reiniciar_juego_mc()

        if not pausa_master_chief:
            if not modo_nn and not modo_arbol and not modo_KNN:
                mover_master_chief_manual()
                if en_salto:
                    manejar_salto_master_chief()
                guardar_datos_master_chief()
            if modo_nn:
                if contador_salto_master_chief >= intervalo_decidir_salto_master_chief:
                    en_salto, pie_tierra = decidir_salto_master_chief(
                        jugador_halo, plasma, vel_bala_inferior, plasma_misil,
                        plasma_misil_activa, modelo_binario_salto, en_salto, pie_tierra)
                    contador_salto_master_chief = 0
                else:
                    contador_salto_master_chief += 1
                if en_salto:
                    manejar_salto_master_chief()
                jugador_halo.x, accion_horizontal = decidir_movimiento_master_chief(
                    jugador_halo, plasma_misil, modelo_binario_mov, en_salto, plasma)
            if modo_arbol:
                if contador_salto_master_chief >= intervalo_decidir_salto_master_chief:
                    en_salto, pie_tierra = decidir_salto_master_chief_arbol(
                        jugador_halo, plasma, vel_bala_inferior, plasma_misil,
                        plasma_misil_activa, arbol_decision_salto, en_salto, pie_tierra)
                    contador_salto_master_chief = 0
                else:
                    contador_salto_master_chief += 1
                if en_salto:
                    manejar_salto_master_chief()
                jugador_halo.x, accion_horizontal = decidir_movimiento_master_chief_arbol(
                    jugador_halo, plasma_misil, arbol_decision_mov, en_salto, plasma)
            if modo_KNN:
                if contador_salto_master_chief >= intervalo_decidir_salto_master_chief:
                    en_salto, pie_tierra = decidir_salto_master_chief_KNN(
                        jugador_halo, plasma, vel_bala_inferior, plasma_misil,
                        plasma_misil_activa, modelo_KNN_salto, en_salto, pie_tierra)
                    contador_salto_master_chief = 0
                else:
                    contador_salto_master_chief += 1
                if en_salto:
                    manejar_salto_master_chief()
                jugador_halo.x, accion_horizontal = decidir_movimiento_master_chief_KNN(
                    jugador_halo, plasma_misil, modelo_KNN_mov, en_salto, plasma)
            if not plasma_activa:
                disparar_proyectil_suelo()
            disparar_bala_vertical()
            actualizar_juego_mc()

        pygame.display.flip()
        reloj_master_chief.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main_game_loop()