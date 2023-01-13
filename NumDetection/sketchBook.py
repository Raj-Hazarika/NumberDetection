import pygame
import glob
import os


def sketch():
    pygame.init()
    screen = pygame.display.set_mode((280, 280))
    pygame.display.set_caption("Trace")
    clock = pygame.time.Clock()

    loop = True
    press = False
    color = "white"
    [os.remove(png) for png in glob.glob("*png")]
    while loop:
        try:
            # pygame.mouse.set_visible(False)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    loop = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        screen.fill(pygame.Color(0, 0, 0))
                    if event.key == pygame.K_s:
                        pygame.image.save(screen, "Train/num.png")
                        loop = False

            px, py = pygame.mouse.get_pos()
            if pygame.mouse.get_pressed() == (1, 0, 0):
                pygame.draw.rect(screen, (255, 255, 255), (px, py, 10, 10))
            if pygame.mouse.get_pressed() == (0, 0, 1):
                pygame.draw.rect(screen, (0, 0, 0), (px, py, 10, 10))

            if event.type == pygame.MOUSEBUTTONUP:
                press == False
            pygame.display.update()
            clock.tick(1000)
        except Exception as e:
            print(e)
            pygame.quit()

    pygame.quit()
