import tempfile
import os
from moviepy.editor import ImageSequenceClip
import pygame
import random
import numpy as np
from typing import List
import torch.nn as nn
import torch.optim as optim
import torch

# === CONFIG ===
# game constants
FPS = 30
MAX_SEC = 20
WIN_WIDTH = 500
WIN_HEIGHT = 800
GRAVITY = 2
VELOCITY = 8
BIRD_SHAPE = (30, 30)  # width, height of the bird
PIPE_SHAPE = (80, 400)  # width, height of the pipe
PIPE_GAP = 150  # gap between top and bottom pipe
PIPE_DISTANCE = 200  # distance between pipes
SCORE_AWARD = 10  # score for passing a pipe
RENDER = False  # Turn to True if you want to see the simulation
# genetic algorithm constants
MUTATION_RATE = 0.02
MAX_POPULATION = 100
ELITE_INDIVIDUALS = 20
NUM_OF_GENERATIONS = 50

# === INIT ===
if RENDER:
    pygame.init()
    WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Flappy Bird GA")
else:
    WIN = None  # avoid using WIN if rendering is disabled

# === CLASSES ===


class BirdAI:
    def __init__(self):
        self.x = 100
        self.y = 300
        self.velocity = 0.0
        self.tick_count = 0
        self.width = BIRD_SHAPE[0]
        self.height = BIRD_SHAPE[1]
        self.alive = True
        # genetic algorithm properties
        self.alive = True
        self.time_survived = 0.0
        self.score = 0
        self.fitness = 0.0
        # bird's brain (neural network) used for decision making
        # inputs: bird's y, pipe's x, pipe's top y, pipe's bottom y, distance to next pipe
        self.input_nodes = 5
        self.hidden_nodes = 10  # hidden layer nodes
        self.output_nodes = 1  # output: flap or not
        self.neural_network = nn.Sequential(
            nn.Linear(self.input_nodes, self.hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes, self.output_nodes)
        )
        self.optimizer = optim.Adam(self.neural_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        # HE initialization
        self.w1 = nn.Parameter(
            torch.randn(self.input_nodes, self.hidden_nodes) * np.sqrt(2.0 / self.input_nodes))
        self.w2 = nn.Parameter(
            torch.randn(self.hidden_nodes, self.output_nodes) * np.sqrt(2.0 / self.hidden_nodes))
        self.b1 = nn.Parameter(torch.zeros(self.hidden_nodes))
        self.b2 = nn.Parameter(torch.zeros(self.output_nodes))

    def think(self, pipe_x, pipe_top_y, pipe_bottom_y):
        inputs = [
            self.y / WIN_HEIGHT,  # normalize bird's y position
            pipe_x / WIN_WIDTH,  # normalize pipe's x position
            pipe_top_y / WIN_HEIGHT,  # normalize top pipe's y position
            pipe_bottom_y / WIN_HEIGHT,  # normalize bottom pipe's y position
            (pipe_x - self.x) / WIN_WIDTH  # distance to next pipe normalized
        ]
        x = torch.tensor(inputs, dtype=torch.float32)
        x = torch.matmul(x, self.w1) + self.b1
        x = torch.relu(x)
        x = torch.matmul(x, self.w2) + self.b2
        x = torch.sigmoid(x)
        return x.item() > 0.5

    def flap(self):
        self.velocity = -8
        self.tick_count = 0

    def move(self):
        self.tick_count += 1
        self.velocity += GRAVITY
        self.y += self.velocity
        self.y = max(0, min(self.y, WIN_HEIGHT - self.height))

    def get_weights(self):
        return {
            'w1': self.w1.data.numpy(),
            'w2': self.w2.data.numpy(),
            'b1': self.b1.data.numpy(),
            'b2': self.b2.data.numpy()
        }


class Pipe:
    GAP = PIPE_GAP
    VELOCITY = VELOCITY
    WIDTH = PIPE_SHAPE[0]
    BODY_HEIGHT = PIPE_SHAPE[1]

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
        bottom_rect = pygame.Rect(
            self.x, self.bottom, self.WIDTH, self.BODY_HEIGHT)
        return bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect)


# === GA FUNCTIONS ===


def mate(parent_1: BirdAI, parent_2: BirdAI):

    # flatten weights to genes
    parent_1_genes = np.concatenate(
        [parent_1.get_weights()[k].flatten() for k in ['w1', 'w2', 'b1', 'b2']])
    parent_2_genes = np.concatenate(
        [parent_2.get_weights()[k].flatten() for k in ['w1', 'w2', 'b1', 'b2']])
    genes_length = len(parent_1_genes)

    number_of_children = random.randint(1, 3)
    children_genes = []
    for _ in range(number_of_children):
        # 2) crossover
        crossover_points = np.random.choice(
            [0, 1], size=genes_length, p=[0.5, 0.5])
        child_genes = np.where(crossover_points == 0,
                               parent_1_genes, parent_2_genes)

        # 3) mutation
        for i in range(genes_length):
            if np.random.random() < MUTATION_RATE:
                # gaussian noise
                child_genes[i] = np.random.normal(
                    loc=child_genes[i], scale=0.1)

        children_genes.append(child_genes)

    # 4) make BirdAI instances from genes
    children = []
    for child_gene in children_genes:
        # reshape weights to match the original shapes
        w1 = child_gene[:parent_1.w1.numel()].reshape(parent_1.w1.shape)
        w2 = child_gene[parent_1.w1.numel():parent_1.w1.numel(
        ) + parent_1.w2.numel()].reshape(parent_1.w2.shape)
        b1 = child_gene[parent_1.w1.numel() + parent_1.w2.numel():parent_1.w1.numel(
        ) + parent_1.w2.numel() + parent_1.b1.numel()].reshape(parent_1.b1.shape)
        b2 = child_gene[-parent_1.b2.numel():].reshape(parent_1.b2.shape)

        child = BirdAI()
        child.w1.data = torch.tensor(w1, dtype=torch.float32)
        child.w2.data = torch.tensor(w2, dtype=torch.float32)
        child.b1.data = torch.tensor(b1, dtype=torch.float32)
        child.b2.data = torch.tensor(b2, dtype=torch.float32)
        children.append(child)

    return children


def run_generation_simulation(population: List[BirdAI]) -> List[BirdAI]:
    pipes = [Pipe(WIN_WIDTH + 100)]  # khởi tạo pipe đầu tiên
    frame = 0  # khởi tạo frame
    base_y = WIN_HEIGHT - 100  # y-coordinate of the ground

    # Reset các giá trị
    for bird in population:
        bird.alive = True
        bird.time_survived = 0.0
        bird.score = 0
        bird.fitness = 0.0

    # Mảng để track những pipe đã được vượt qua (để cộng score mỗi pipe chỉ một lần)
    passed_pipes = {bird: set() for bird in population}

    while frame < FPS * MAX_SEC:
        # 1) Di chuyển pipes và sinh thêm khi cần
        for pipe in pipes:
            pipe.move()
        if pipes[-1].x < WIN_WIDTH - PIPE_DISTANCE:
            pipes.append(Pipe(WIN_WIDTH))
        pipes = [p for p in pipes if not p.off_screen()]

        alive_count = 0  # đếm số bird còn sống

        # 2) Với mỗi bird
        for bird in population:
            # nếu bird đã chết thì bỏ qua
            if not bird.alive:
                continue

            # a) Ra quyết định vỗ cánh hay không
            if bird.think(pipes[0].x, pipes[0].top, pipes[0].bottom):
                bird.flap()
            bird.move()

            # b) Kiểm tra va chạm với đất
            if bird.y + bird.height >= base_y:
                bird.alive = False
                bird.time_survived = frame / FPS
                continue

            # c) Kiểm tra va chạm với pipes
            for pipe in pipes:
                if pipe.collides_with(bird):
                    bird.alive = False
                    bird.time_survived = frame / FPS
                    break

            if not bird.alive:
                continue

            # d) Tính score: mỗi khi bird.x vượt qua tâm ống, cộng 1
            for pipe in pipes:
                # nếu pipe chưa qua và bird đã vượt tâm ống
                if pipe not in passed_pipes[bird] and bird.x > pipe.x + Pipe.WIDTH:
                    passed_pipes[bird].add(pipe)
                    bird.score += 1

            # e) Nếu vẫn sống, tăng alive_count
            alive_count += 1

        if alive_count == 0:
            break

        frame += 1

    # 3) Cuối vòng: bird vẫn sống thì time_survived = full time
    for bird in population:
        if bird.alive:
            bird.time_survived = frame / FPS
        # 4) Tính fitness cuối cùng
        #    fitness = time_survived (giây) + score * SCORE_AWARD
        bird.fitness = bird.time_survived + bird.score * SCORE_AWARD

    return population


def tournament_selection(population: List[BirdAI], k: int = 2) -> List[BirdAI]:
    selected = random.sample(population, k)
    winner = max(selected, key=lambda b: b.fitness)
    return winner


def genetic_alg():
    population = [BirdAI() for _ in range(MAX_POPULATION)]

    for generation in range(NUM_OF_GENERATIONS):
        print(f"<< GENERATION {generation} >>")
        population = run_generation_simulation(population)
        population.sort(key=lambda b: b.fitness, reverse=True)
        print(f"Best time survived: {population[0].time_survived:.2f}s")
        print(f"Best score: {population[0].score}")

        next_generation = population[:ELITE_INDIVIDUALS]

        while len(next_generation) < MAX_POPULATION:
            parent_1 = tournament_selection(population, k=5)
            parent_2 = tournament_selection(population, k=5)
            children = mate(parent_1, parent_2)
            next_generation.extend(children)

        population = next_generation[:MAX_POPULATION]

    population.sort(key=lambda b: b.fitness, reverse=True)
    best = population[0]
    print("\nAfter training:")
    print(f"\nBest Bird time survived: {best.time_survived:.2f}s")
    display_best_individual(best)


def display_best_individual(best_bird: BirdAI):
    pygame.init()
    WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Flappy Bird - Best Bird Demo")

    clock = pygame.time.Clock()

    # Create a new bird with the same weights as the best bird
    bird = BirdAI()
    bird.w1.data = best_bird.w1.data.clone()
    bird.w2.data = best_bird.w2.data.clone()
    bird.b1.data = best_bird.b1.data.clone()
    bird.b2.data = best_bird.b2.data.clone()

    # Set the same random seed as training
    random.seed(42)
    pipes = [Pipe(WIN_WIDTH + 100)]  # Match the training initialization
    base_y = WIN_HEIGHT - 100
    frame = 0
    run = True
    bird_score = 0

    temp_dir = tempfile.mkdtemp()
    frame_paths = []

    # Track passed pipes for scoring
    passed_pipes = set()

    while run and frame < FPS * MAX_SEC:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Pipe management (match training logic)
        for pipe in pipes:
            pipe.move()
        if pipes[-1].x < WIN_WIDTH - PIPE_DISTANCE:
            pipes.append(Pipe(WIN_WIDTH))
        pipes = [p for p in pipes if not p.off_screen()]

        # Bird decision using neural network (match training logic)
        if bird.think(pipes[0].x, pipes[0].top, pipes[0].bottom):
            bird.flap()
        bird.move()

        # Collision detection (match training logic)
        # Check ground collision
        if bird.y + bird.height >= base_y:
            run = False
            break

        # Check pipe collision
        for pipe in pipes:
            if pipe.collides_with(bird):
                run = False
                break

        if not run:
            break

        # Score calculation (match training logic)
        for pipe in pipes:
            if pipe not in passed_pipes and bird.x > pipe.x + Pipe.WIDTH:
                passed_pipes.add(pipe)
                bird_score += 1

        # Drawing
        WIN.fill((135, 206, 250))
        for pipe in pipes:
            pygame.draw.rect(
                WIN,
                (0, 255, 0),
                (pipe.x, pipe.top - Pipe.BODY_HEIGHT, Pipe.WIDTH, Pipe.BODY_HEIGHT),
            )
            pygame.draw.rect(
                WIN, (0, 255, 0), (pipe.x, pipe.bottom,
                                   Pipe.WIDTH, Pipe.BODY_HEIGHT)
            )
        pygame.draw.rect(WIN, (255, 255, 0),
                         (bird.x, bird.y, bird.width, bird.height))
        pygame.draw.rect(
            WIN, (139, 69, 19), (0, base_y, WIN_WIDTH, WIN_HEIGHT - base_y)
        )
        pygame.display.update()

        # Save frame
        frame_path = os.path.join(temp_dir, f"frame_{frame:04d}.png")
        pygame.image.save(WIN, frame_path)
        frame_paths.append(frame_path)

        frame += 1

    print("\nBest Bird Performance:")
    print(f"Training survival time: {best_bird.time_survived:.2f}s")
    print(f"Training score: {best_bird.score}")
    print(f"Demo survival time: {frame / FPS:.2f}s")
    print(f"Demo score: {bird_score}")

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
