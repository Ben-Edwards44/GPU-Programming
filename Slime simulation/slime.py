import settings
import math
from numba import cuda


@cuda.jit
def update_agent_pos(pos, angles, screen_pixels):
    inx = cuda.grid(1)

    if inx < angles.shape[0]:
        theta = angles[inx]

        new_x = settings.SPEED * math.cos(theta)
        new_y = settings.SPEED * math.sin(theta)

        pos[inx][0] += new_x
        pos[inx][1] += new_y

        check_collision(pos[inx], angles, inx)
        update_screen_pixels(screen_pixels, pos[inx])
        steer_to_trail(inx, pos[inx], angles, screen_pixels)


@cuda.jit
def check_collision(pos, angles, inx):
    x, y = pos

    if 0 <= x < settings.WIDTH and 0 <= y < settings.HEIGHT:
        return
        
    if x < 0:
        pos[0] = 0
    elif x > settings.WIDTH - 1:
        pos[0] = settings.WIDTH - 1

    if y < 0:
        pos[1] = 0
    elif y > settings.HEIGHT - 1:
        pos[1] = settings.HEIGHT - 1

    angles[inx] = generate_random_num(inx) % (2 * math.pi)


@cuda.jit
def steer_to_trail(inx, pos, angles, screen_pixels):
    angle = angles[inx]

    forward = sense(pos, screen_pixels, angle)
    left = sense(pos, screen_pixels, angle - settings.SENSOR_ANGLE)
    right = sense(pos, screen_pixels, angle + settings.SENSOR_ANGLE)

    steer_strength = generate_random_num(inx) % 50 / 100 + 0.75

    if (forward < left and forward < right) and not (forward > left and forward > right):
        num = generate_random_num(inx)

        if num > 1e8:
            angles[inx] -= settings.TURN_ANGLE * settings.TURN_SPEED * steer_strength
        else:
            angles[inx] += settings.TURN_ANGLE * settings.TURN_SPEED * steer_strength
    elif left > forward and left > right:
        angles[inx] -= settings.TURN_ANGLE * settings.TURN_SPEED * steer_strength
    elif right > forward and right > left:
        angles[inx] += settings.TURN_ANGLE * settings.TURN_SPEED * steer_strength


@cuda.jit
def sense(pos, screen_pixels, new_angle):
    x, y = pos

    total = 0
    for i in range(settings.SENSOR_LENGTH):
        new_x = x + i * math.cos(new_angle)
        new_y = y + i * math.sin(new_angle)

        if 0 <= new_x < screen_pixels.shape[0] and 0 <= new_y < screen_pixels.shape[1]:
            total += screen_pixels[int(new_x)][int(new_y)]

    return total


@cuda.jit
def generate_random_num(thread_inx):
    thread_inx ^= 2747636419
    thread_inx *= 2654435769
    thread_inx ^= thread_inx >> 16
    thread_inx *= 2654435769
    thread_inx ^= thread_inx >> 16
    thread_inx *= 2654435769

    return thread_inx


@cuda.jit
def update_screen_pixels(screen_pixels, pos):
    x, y = pos
    screen_pixels[int(x)][int(y)] = 1