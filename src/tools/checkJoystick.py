import pygame
import sys

# Inicializamos pygame
pygame.init()

# Inicializamos el joystick (volante)
pygame.joystick.init()

# Verificamos cuántos joysticks están conectados
joystick_count = pygame.joystick.get_count()

if joystick_count == 0:
    print("No se detectó ningún joystick.")
    sys.exit()

# Obtenemos el primer joystick (asumiendo que solo tienes uno)
joystick = pygame.joystick.Joystick(0)
joystick.init()

# Mostramos información básica del joystick
print(f"Nombre del joystick: {joystick.get_name()}")
print(f"Cantidad de ejes: {joystick.get_numaxes()}")
print(f"Cantidad de botones: {joystick.get_numbuttons()}")
print(f"Cantidad de hat switches: {joystick.get_numhats()}")

# Bucle principal
try:
    while True:
        for event in pygame.event.get():
            # Comprobamos si hay algún evento
            if event.type == pygame.JOYAXISMOTION:
                # Mostramos el valor de los ejes
                axis = event.axis
                value = joystick.get_axis(axis)
                print(f"Eje {axis} -> Valor: {value:.2f}")

            elif event.type == pygame.JOYBUTTONDOWN:
                # Comprobamos si un botón ha sido presionado
                button = event.button
                print(f"Botón {button} presionado")

            elif event.type == pygame.JOYBUTTONUP:
                # Comprobamos si un botón ha sido liberado
                button = event.button
                print(f"Botón {button} liberado")

        pygame.time.wait(100)  # Esperamos un poco para evitar saturar la salida
except KeyboardInterrupt:
    print("\nTerminando el programa.")
    pygame.quit()
    sys.exit()
