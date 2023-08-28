import cell
import pygame
import numpy
from numba import cuda
from random import uniform


CELLS_X = 750
CELLS_Y = 380

SCREEN_X = 1500
SCREEN_Y = 760


pygame.init()
window = pygame.display.set_mode((SCREEN_X, SCREEN_Y))


def draw_array(array):
    pygame.surfarray.blit_array(window, array)
    pygame.display.update()


def prepare_gpu_data(conv):
    cells = numpy.array([[uniform(0, 1) for _ in range(CELLS_Y)] for _ in range(CELLS_X)])
    updated_cells = numpy.array([[x for x in i] for i in cells])
    convolution = numpy.array(conv)
    screen_array = numpy.zeros((SCREEN_X, SCREEN_Y))

    cells_d = cuda.to_device(cells)
    cells_updated_d = cuda.to_device(updated_cells)
    conv_d = cuda.to_device(convolution)
    screen_d = cuda.to_device(screen_array)

    num_threads = (8, 8)
    num_blocks_x = (num_threads[0] + CELLS_X - 1) // num_threads[0]
    num_blocks_y = (num_threads[1] + CELLS_Y - 1) // num_threads[1]

    num_blocks = (num_blocks_x, num_blocks_y)

    return cells_d, cells_updated_d, conv_d, screen_d, num_threads, num_blocks


def main(convolution, act_func_inx):
    cells_d, cells_updated_d, conv_d, screen_d, num_threads, num_blocks = prepare_gpu_data([i for i in convolution])

    len_x = SCREEN_X // CELLS_X
    len_y = SCREEN_Y // CELLS_Y

    while True:
        cell.main[num_blocks, num_threads](cells_d, cells_updated_d, conv_d, screen_d, len_x, len_y, act_func_inx)

        cells = cells_updated_d.copy_to_host()
        updated_cells = numpy.array(cells)

        cells_d = cuda.to_device(cells)
        cells_updated_d = cuda.to_device(updated_cells)

        screen_array = screen_d.copy_to_host()
        draw_array(screen_array)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                quit()