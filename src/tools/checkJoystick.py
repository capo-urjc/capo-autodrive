import pygame

pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"Joystick Name: {joystick.get_name()}")
print(f"Number of Axes: {joystick.get_numaxes()}")
print(f"Number of Buttons: {joystick.get_numbuttons()}")

for i in range(joystick.get_numaxes()):
    print(f"Axis {i}: {joystick.get_axis(i)}")

for i in range(joystick.get_numbuttons()):
    print(f"Button {i}: {joystick.get_button(i)}")

pygame.quit()