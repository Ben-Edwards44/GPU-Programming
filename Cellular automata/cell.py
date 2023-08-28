from numba import cuda


@cuda.jit
def worms_act(x):
    return -1 / (2**(0.6 * x**2)) + 1


@cuda.jit
def waves_act(x):
    return abs(1.2 * x)


@cuda.jit
def paths_act(x):
    return 1 / (2**((x - 3.5)**2))


@cuda.jit
def mitosis_act(x):
    return -1 / (0.9 * x**2 + 1) + 1


@cuda.jit
def flickers_act(x):
    return -1 / (0.89 * x**2 + 1) + 1


@cuda.jit
def activation_func(x, func_inx):
    if func_inx == 0:
        return worms_act(x)
    elif func_inx == 1:
        return waves_act(x)
    elif func_inx == 2:
        return paths_act(x)
    elif func_inx == 3:
        return mitosis_act(x)
    else:
        return flickers_act(x)


@cuda.jit
def find_new_value(x, y, conv, cells, func_inx):
    conv_result = apply_convolution(x, y, cells, conv)
    activation_val = activation_func(conv_result, func_inx)

    if activation_val > 1:
        activation_val = 1
    elif activation_val < 0:
        activation_val = 0

    return activation_val


@cuda.jit
def apply_convolution(x, y, cells, conv):
    total = 0

    conv_inx = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            inx1 = (x + i) % cells.shape[0]
            inx2 = (y + j) % cells.shape[1]

            total += cells[inx1][inx2] * conv[conv_inx]

            conv_inx += 1

    return total


@cuda.jit
def main(cells, updated_cells, conv, screen_array, len_x, len_y, func_inx):
    x, y = cuda.grid(2)

    if x >= cells.shape[0] or y >= cells.shape[1]:
        return

    new_val = find_new_value(x, y, conv, cells, func_inx)

    updated_cells[x][y] = new_val

    for i in range(len_x):
        for j in range(len_y):
            inx1 = x * len_x + i
            inx2 = y * len_y + j

            blue = 0x0000FF

            screen_array[inx1][inx2] = new_val * blue