import slime
import settings
import pygame
import numpy
from numba import cuda
from math import pi
from random import randint


pygame.init()
window = pygame.display.set_mode((settings.WIDTH, settings.HEIGHT))
pygame.display.set_caption("Slime Mould Simulation")


def create_agents_square(num):
    pos = numpy.zeros((num, 2))

    for i in range(num):
        pos[i][0] = randint((settings.WIDTH // 2) - 150, (settings.WIDTH // 2) + 150)
        pos[i][1] = randint((settings.HEIGHT // 2) - 150, (settings.HEIGHT // 2) + 150)

    angles = numpy.random.random(num)
    angles *= pi * 2

    return pos, angles


def create_agents_circle(num):
    pos = numpy.zeros((num, 2))

    for i in range(num):
        valid_pos = False
        while not valid_pos:
            x = randint((settings.WIDTH // 2) - 150, (settings.WIDTH // 2) + 150)
            y = randint((settings.HEIGHT // 2) - 150, (settings.HEIGHT // 2) + 150)

            if (x - settings.WIDTH // 2)**2 + (y - settings.HEIGHT // 2)**2 <= 150**2:
                valid_pos = True

        pos[i][0] = x
        pos[i][1] = y

    angles = numpy.random.random(num)
    angles *= pi * 2

    return pos, angles


def create_agents_two_square(num):
    pos = numpy.zeros((num, 2))

    for i in range(num // 2):
        pos[i][0] = randint((settings.WIDTH // 2) - 200, (settings.WIDTH // 2) - 50)
        pos[i][1] = randint((settings.HEIGHT // 2) - 150, (settings.HEIGHT // 2) + 150)

    for i in range(num // 2, num):
        pos[i][0] = randint((settings.WIDTH // 2) + 50, (settings.WIDTH // 2) + 200)
        pos[i][1] = randint((settings.HEIGHT // 2) - 150, (settings.HEIGHT // 2) + 150)

    angles = numpy.random.random(num)
    angles *= pi * 2

    return pos, angles


@cuda.jit
def update_pixel_array(pixel_array):
    x, y = cuda.grid(2)

    if x < pixel_array.shape[0] and y < pixel_array.shape[1]:
        pixel_array[x][y] *= 0.995


def draw(screen_pixels):
    pygame.surfarray.blit_array(window, screen_pixels * 255)
    pygame.display.update()


def main():
    start_pattern = randint(0, 2)

    if start_pattern == 0:
        pos, angles = create_agents_circle(settings.NUM_AGENTS)
    elif start_pattern == 1:
        pos, angles = create_agents_square(settings.NUM_AGENTS)
    else:
        pos, angles = create_agents_two_square(settings.NUM_AGENTS)
    
    screen_pixels = numpy.zeros((settings.WIDTH, settings.HEIGHT))

    pos_d_data = cuda.to_device(pos)
    angle_d_data = cuda.to_device(angles)
    pixels_d_data = cuda.to_device(screen_pixels)

    threads_slime = 1
    blocks_slime = (threads_slime + angles.shape[0] - 1) // threads_slime

    threads_screen = (32, 32)
    blocks_screen = ((threads_screen[0] + screen_pixels.shape[0] - 1) // threads_screen[0], (threads_screen[1] + screen_pixels.shape[1] - 1) // threads_screen[1])

    while True:
        slime.update_agent_pos[blocks_slime, threads_slime](pos_d_data, angle_d_data, pixels_d_data)
        update_pixel_array[blocks_screen, threads_screen](pixels_d_data)

        screen_pixels = pixels_d_data.copy_to_host()
        draw(screen_pixels)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                quit()


main()