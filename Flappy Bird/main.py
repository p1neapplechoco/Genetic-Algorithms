import pygame
import random
import numpy as np
from typing import List

# === CONFIG ===
FPS = 30
MAX_SEC = 10
WIN_WIDTH = 500
WIN_HEIGHT = 800
GRAVITY = 2
MUTATION_CHANCE = 0.1
MAX_POPULATION = 100
ELITE_INDIVIDUALS = 40
NUM_OF_GENERATIONS = 500
RENDER = False  # Turn to True if you want to see the simulation

# === INIT ===
if RENDER:
    pygame.init()
    WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Flappy Bird GA")
else:
    WIN = None  # avoid using WIN if rendering is disabled

# === CLASSES ===


class BirdAI:
    def __init__(self, genes=None):
        self.x = 100
        self.y = 300
        self.velocity = 0.0
        self.tick_count = 0
        self.width = 100
        self.height = 100
        self.alive = True
        self.time_survived = 0.0
        self.genes = (
            genes if genes else [np.random.randint(0, 1) for _ in range(FPS * MAX_SEC)]
        )

    def flap(self):
        self.velocity = -8
        self.tick_count = 0

    def move(self):
        self.tick_count += 1
        dy = self.velocity * self.tick_count + 0.5 * GRAVITY * self.tick_count**2
        self.y += min(18, dy)
        self.y = max(0, min(self.y, WIN_HEIGHT - self.height))


class Pipe:
    GAP = 200
    VELOCITY = 8
    WIDTH = 80
    BODY_HEIGHT = 400
    random.seed(42)

    def __init__(self, x):
        self.x = x
        self.set_height()

    def set_height(self):
        self.height = random.randint(100, WIN_HEIGHT - self.GAP - 300)
        self.top = self.height
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VELOCITY

    def off_screen(self):
        return self.x + self.WIDTH < 0

    def collides_with(self, bird: BirdAI) -> bool:
        bird_rect = pygame.Rect(bird.x, bird.y, bird.width, bird.height)
        top_rect = pygame.Rect(
            self.x, self.top - self.BODY_HEIGHT, self.WIDTH, self.BODY_HEIGHT
        )
        bottom_rect = pygame.Rect(self.x, self.bottom, self.WIDTH, self.BODY_HEIGHT)
        return bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect)


# === GA FUNCTIONS ===


def mate(parent_1: BirdAI, parent_2: BirdAI) -> List[int]:
    n = len(parent_1.genes)
    child = []
    for i in range(n):
        p = np.random.random()
        if p < (1 - MUTATION_CHANCE) / 2:
            child.append(parent_1.genes[i])
        elif p < (1 - MUTATION_CHANCE):
            child.append(parent_2.genes[i])
        else:
            child.append(np.random.randint(2))
    return child


def run_generation_simulation(population: List[BirdAI]) -> List[BirdAI]:
    pipes = [Pipe(600)]
    frame = 0
    base_y = WIN_HEIGHT - 100
    random.seed(42)

    while frame < FPS * MAX_SEC:
        for pipe in pipes:
            pipe.move()
        if pipes[-1].x < 250:
            pipes.append(Pipe(WIN_WIDTH))
        pipes = [pipe for pipe in pipes if not pipe.off_screen()]

        alive_count = 0
        for bird in population:
            if not bird.alive:
                continue

            if frame < len(bird.genes):
                if bird.genes[frame] == 1:
                    bird.flap()
                bird.move()

            if bird.y + bird.height >= base_y:
                bird.alive = False
                bird.time_survived = frame / FPS
                continue

            for pipe in pipes:
                if pipe.collides_with(bird):
                    bird.alive = False
                    bird.time_survived = frame / FPS
                    break

            if bird.alive:
                alive_count += 1

        if alive_count == 0:
            break
        frame += 1

    for bird in population:
        if bird.alive:
            bird.time_survived = frame / FPS

    return population


def genetic_alg():
    population = [BirdAI() for _ in range(MAX_POPULATION)]

    for generation in range(NUM_OF_GENERATIONS):
        print(f"<< GENERATION {generation} >>")
        population = run_generation_simulation(population)
        population.sort(key=lambda b: b.time_survived, reverse=True)
        print(f"Best survival: {population[0].time_survived:.2f}s")

        next_gen = [
            BirdAI(population[i].genes.copy()) for i in range(ELITE_INDIVIDUALS)
        ]

        mated = set()
        for i, p1 in enumerate(population[: ELITE_INDIVIDUALS + 3]):
            if i in mated:
                continue
            for j, p2 in enumerate(population[: ELITE_INDIVIDUALS + 3]):
                if i == j or j in mated:
                    continue

                num_kids = random.randint(1, 4)
                while num_kids and len(next_gen) < MAX_POPULATION:
                    child_genes = mate(p1, p2)
                    next_gen.append(BirdAI(child_genes))
                    num_kids -= 1

                mated.add(i)
                mated.add(j)

        population = next_gen

    best = population[0]
    display_best_individual(best)


from moviepy import ImageSequenceClip
import os
import tempfile


def display_best_individual(best_bird: BirdAI):
    pygame.init()
    WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Flappy Bird - Best Bird Demo")

    clock = pygame.time.Clock()
    bird = BirdAI(best_bird.genes.copy())
    pipes = [Pipe(600)]
    base_y = WIN_HEIGHT - 100
    frame = 0
    run = True

    temp_dir = tempfile.mkdtemp()
    frame_paths = []

    while run and frame < FPS * MAX_SEC:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Pipe management
        for pipe in pipes:
            pipe.move()
        if pipes[-1].x < 250:
            pipes.append(Pipe(WIN_WIDTH))
        pipes = [pipe for pipe in pipes if not pipe.off_screen()]

        # Bird action
        if frame < len(bird.genes) and bird.genes[frame] == 1:
            bird.flap()
        bird.move()

        # Collision
        for pipe in pipes:
            if pipe.collides_with(bird):
                run = False
        if bird.y + bird.height >= base_y:
            run = False

        # Drawing
        WIN.fill((135, 206, 250))
        for pipe in pipes:
            pygame.draw.rect(
                WIN,
                (0, 255, 0),
                (pipe.x, pipe.top - Pipe.BODY_HEIGHT, Pipe.WIDTH, Pipe.BODY_HEIGHT),
            )
            pygame.draw.rect(
                WIN, (0, 255, 0), (pipe.x, pipe.bottom, Pipe.WIDTH, Pipe.BODY_HEIGHT)
            )
        pygame.draw.rect(WIN, (255, 255, 0), (bird.x, bird.y, bird.width, bird.height))
        pygame.draw.rect(
            WIN, (139, 69, 19), (0, base_y, WIN_WIDTH, WIN_HEIGHT - base_y)
        )
        pygame.display.update()

        # Save frame
        frame_path = os.path.join(temp_dir, f"frame_{frame:04d}.png")
        pygame.image.save(WIN, frame_path)
        frame_paths.append(frame_path)

        frame += 1

    # Use moviepy to turn frames into video
    video_clip = ImageSequenceClip(frame_paths, fps=FPS)
    video_path = "best_bird_moviepy.mp4"
    video_clip.write_videofile(video_path, codec="libx264", fps=FPS)
    print(f"🎞️ Saved video to {video_path}")

    # Cleanup
    for f in frame_paths:
        os.remove(f)
    os.rmdir(temp_dir)
    pygame.quit()


if __name__ == "__main__":
    genetic_alg()
