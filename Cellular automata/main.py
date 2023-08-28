import main_simulation
import data
import pygame
import pygame_ui_toolkit


OPTIONS = {"worms" : data.worms,
           "waves" : data.waves,
           "paths" : data.paths,
           "mitosis" : data.mitosis,
           "flickers" : data.flickers}


pygame.init()
window = pygame.display.set_mode((1500, 760))
pygame.display.set_caption("Neural Cellular Automata")

start_pressed = False

conv = data.worms[0]
act_func_inx = data.worms[1]


def start():
    global start_pressed

    if start_pressed:
        return

    start_pressed = True
    
    update()
    main_simulation.main(conv, act_func_inx)


def on_option_changed(option):
    global conv, act_func_inx

    name = option.text_wrapper.text
    conv, act_func_inx = OPTIONS[name]


def create_ui_elements():
    global dropdown, go_button, text

    option_names = [i for i in OPTIONS.keys()]

    dropdown = pygame_ui_toolkit.dropdown.RectDropdown(window, option_names, 375, 175, (255, 255, 255), 150, 75, (0, 0, 0), 36, on_option_changed=on_option_changed, corner_radius=5)

    go_button = pygame_ui_toolkit.button.RectButton(window, 750, 650, (255, 255, 255), 100, 20, start, corner_radius=5)
    pygame_ui_toolkit.button_size_change.change_existing_button(go_button, (200, 125), (220, 145), (180, 105))
    go_button = pygame_ui_toolkit.button.TextWrapper(go_button, "Start", (0, 0, 0), 40)

    text = pygame_ui_toolkit.text.RectTextBox(window, 375, 100, 150, 75, "Choose option:", (0, 0, 0), (255, 255, 255), 28)


def update():
    window.fill((0, 0, 0))

    dropdown.update()
    go_button.update()
    text.draw()

    pygame.display.update()


def main():
    create_ui_elements()

    while True:
        update()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                quit()


main()