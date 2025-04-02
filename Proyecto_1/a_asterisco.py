import pygame
from queue import PriorityQueue
import time

# Configuraciones iniciales
ANCHO_VENTANA = 600
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización de Nodos")

# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)
AZUL = (0, 0, 255)

pygame.font.init() 
FUENTE = pygame.font.SysFont("Arial", 10,)  





class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.g = float("inf")  
        self.h = 0 
        self.f = float("inf")
        self.father = None  

    #Obtener la fila y columna del nodo actual
    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA
    
    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def buscar(self):
        self.color = AZUL

    def ruta_final(self):
        self.color = ROJO


    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)

    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    y, x = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col


def main(ventana, ancho):
    FILAS = 11
    grid = crear_grid(FILAS, ancho)
    inicio = None
    fin = None
    corriendo = True

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:  # Left click
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()

                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()

                elif nodo != fin and nodo != inicio:
                    nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:  # Click derecho
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                nodo.restablecer()     

                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

            elif pygame.key.get_pressed()[pygame.K_SPACE]:
                algoritmo(inicio, fin, grid, FILAS)
                

    pygame.quit()


def heuristica(nodo, fin):
    # Obtener las posiciones del nodo actual y del nodo final
    x1, y1 = nodo.get_pos()
    x2, y2 = fin.get_pos()
    distancia_x = abs(x2 - x1)
    distancia_y = abs(y2 - y1)
    distancia_total = distancia_x + distancia_y
    heuristica = distancia_total * 10

    return heuristica



def vecinos(nodo, grid):
    vecinos = []

    direcciones = [
        (1, 0), (-1, 0), (0, 1), (0, -1), 
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]
    
    for direccion in direcciones:
        nueva_fila = nodo.fila + direccion[0]
        nueva_col = nodo.col + direccion[1]
        
        if 0 <= nueva_fila < nodo.total_filas and 0 <= nueva_col < nodo.total_filas:
            vecino = grid[nueva_fila][nueva_col]
            if not vecino.es_pared():
                vecinos.append(vecino)

    return vecinos

def mostrar_ruta(current):
    while current.father:
        current = current.father
        current.ruta_final()


   

def algoritmo(inicio, fin, grid, filas):
    contador = 0
    lista_abierta = PriorityQueue()
    lista_abierta.put((0, contador, inicio))
    lista_cerrada = set()
    
    inicio.g = 0
    inicio.f = heuristica(inicio, fin)
    
    while not lista_abierta.empty():
        current = lista_abierta.get()[2]
        lista_cerrada.add(current)

        if current == fin:
            mostrar_ruta(fin)
            return True
        
        for vecino in vecinos(current, grid):
            if vecino in lista_cerrada:
                continue


               
    
            if (current.fila != vecino.fila):
                peso = 14
            else:
                peso = 10     
           
            temporal_g = current.g + peso

            if temporal_g < vecino.g:
                vecino.father = current
                vecino.g = temporal_g
                vecino.h = heuristica(vecino, fin)
                vecino.f = vecino.g + vecino.h

                if vecino not in [i[2] for i in lista_abierta.queue]:
                    contador += 1
                    lista_abierta.put((vecino.f, contador, vecino))
                    if vecino != fin:
                        vecino.buscar()
        
        if current != inicio:
            current.buscar()
        dibujar(VENTANA, grid, filas, ANCHO_VENTANA)
        time.sleep(0.1) 
    return False


main(VENTANA, ANCHO_VENTANA)
