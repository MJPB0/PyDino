import random
import sys

import numpy as np
import pygame

pygame.init()

screenX = 1200
screenY = 500
screen = pygame.display.set_mode((screenX, screenY))

myfont = pygame.font.SysFont('Times New Roman', 30)
pygame.display.set_caption("PyDino")
icon = pygame.image.load("sprites/dino0000.png")
pygame.display.set_icon(icon)
Cactusses = [pygame.image.load("sprites/cactusBig0000.png"),
             pygame.image.load("sprites/cactusSmall0000.png"),
             pygame.image.load("sprites/cactusSmallMany0000.png")]


class Population:
    MUTATION_PROBABILITY = 10
    CROSSOVER_THRESHOLD = 50

    def __init__(self, game, n):
        self.game = game
        self.n = n
        self.population = []

    def get_alive_amount(self):
        l = len(self.population)
        for dino in self.population:
            if dino.IsDead:
                l -= 1
        return l

    def over(self):
        self.new_population()

    def get_top(self):
        population_sorted = sorted(self.population, key=lambda b: b.result, reverse=True)
        return population_sorted

    def crossover(self, dinoA, dinoB):
        childA = Dino(self.game, self, 0)
        childB = Dino(self.game, self, 1)

        for i in range(len(dinoA.DinoBrain.layers)):
            for x in range(len(dinoA.DinoBrain.layers[i])):
                for k in range(len(dinoA.DinoBrain.layers[i][x].weights)):
                    r = random.randint(0, 100)
                    a = dinoA
                    b = dinoB
                    if r < Population.CROSSOVER_THRESHOLD:
                        a = dinoB
                        b = dinoA
                    childA.DinoBrain.layers[i][x].weights[k] = a.DinoBrain.layers[i][x].weights[k]
                    childB.DinoBrain.layers[i][x].weights[k] = b.DinoBrain.layers[i][x].weights[k]

        return childA, childB

    def mutate(self, dino):
        for i in range(len(dino.DinoBrain.layers)):
            for x in range(len(dino.DinoBrain.layers[i])):
                for k in range(len(dino.DinoBrain.layers[i][x].weights)):
                    if random.randint(0, 100) < Population.MUTATION_PROBABILITY:
                        dino.DinoBrain.layers[i][x].weights[k] = \
                            dino.DinoBrain.layers[i][x].weights[k] * random.uniform(-1., 1.)
        return dino

    def new_population(self):
        # print(f"Generation: {self.game.iteration}, "
        #       f"Highest score this gen: {self.game.best[self.game.iteration - 1]}"
        #       f", Highest overall: {max(self.game.best)}")

        top = self.get_top()
        for dino in self.population:
            del dino

        self.population = []

        print(f"res: {top[0].result}, best: {max(self.game.best)}")
        if top[0].result >= max(self.game.best):
            self.appraise(top[0], top)
        else:
            self.punish(top)

    def appraise(self, newBestDino, dinos):
        print("New Record! Good job my dinos!")
        for i in range(6):
            dino = Dino(self.game, self, i)
            dino.DinoBrain.layers = dinos[i].DinoBrain.layers
            self.population.append(dino)
        for i in range(15):
            dinoA, dinoB = self.crossover(newBestDino, dinos[i])
            dinoA.DinoBrain.layers = self.mutate(dinoA).DinoBrain.layers
            dinoB.DinoBrain.layers = self.mutate(dinoB).DinoBrain.layers
            self.population.append(dinoA)
            self.population.append(dinoB)
        for i in range(0, 14, 2):
            dinoA, dinoB = self.crossover(dinos[i], dinos[i + 1])
            dinoA.DinoBrain.layers = self.mutate(dinoA).DinoBrain.layers
            dinoB.DinoBrain.layers = self.mutate(dinoB).DinoBrain.layers
            self.population.append(dinoA)
            self.population.append(dinoB)

    def punish(self, dinos):
        print("You have to try harder my dinos!")
        selectedParents = 5
        for i in range(selectedParents):
            dino = dinos[i]
            dino.DinoBrain.layers = self.mutate(dino).DinoBrain.layers
            self.population.append(dino)
        n = round((round((self.n - selectedParents + 1) / 2)
                   if (self.n - selectedParents) % 2 == 1
                   else round((self.n - selectedParents) / 2)) / 2)
        for i in range(n):
            dinoA, dinoB = self.crossover(dinos[0], dinos[i])
            dinoA.DinoBrain.layers = self.mutate(dinoA).DinoBrain.layers
            dinoB.DinoBrain.layers = self.mutate(dinoB).DinoBrain.layers
            self.population.append(dinoA)
            self.population.append(dinoB)
        for i in range(self.n - 2 * n):
            dino = Dino(self.game, self, i + self.n / 2)
            dino.DinoBrain.layers = self.mutate(dino).DinoBrain.layers
            self.population.append(dino)

    def initialize_population(self):
        self.population = []
        for i in range(self.n):
            dino = Dino(self.game, self, i)
            self.population.append(dino)


class Neuron:
    def __init__(self):
        self.inputSum = 0
        self.impulse = 0
        self.connections = []
        self.weights = []

    def add_connection(self, connectedNeuron, weight):
        self.connections.append(connectedNeuron)
        self.weights.append(weight)

    def calculate_impulse(self):
        val = 0
        for i in range(len(self.connections)):
            val += self.weights[i] * self.connections[i].impulse
        self.impulse = self.activation(val)

    def activation(self, x):
        return 1 / (1 + np.exp(-x))


class Brain:
    DECISIONS = ["WAIT", "LOW_JUMP", "HIGH_JUMP", "DUCK", "STOP_DUCKING"]

    HIDDEN_LAYERS = 1
    NEURONS_PER_LAYER = 6
    INPUT_VALUES_COUNT = 11
    OUTPUT_VALUES_COUNT = 5

    def __init__(self):
        self.layers = []
        self.layers.append(self.create_layer(Brain.INPUT_VALUES_COUNT))

        for i in range(Brain.HIDDEN_LAYERS):
            self.layers.append(self.create_layer(Brain.NEURONS_PER_LAYER))
        self.layers.append(self.create_layer(Brain.OUTPUT_VALUES_COUNT))

        for i in range(len(self.layers)):
            self.create_connections(i)

    def create_layer(self, x):
        layer = []
        for i in range(x):
            layer.append(Neuron())
        return layer

    def normalize(self, values):
        normalized = []
        for val in values:
            normalized.append((val - min(values)) / (max(values) - min(values)))
        return normalized

    def create_connections(self, x):
        for neuron in self.layers[x]:
            if x == 0:
                continue
            elif x == Brain.HIDDEN_LAYERS + 1:
                for connectedNeuron in self.layers[x - 1]:
                    neuron.add_connection(connectedNeuron, random.uniform(-1., 1.))
            else:
                for connectedNeuron in self.layers[x - 1]:
                    neuron.add_connection(connectedNeuron, random.uniform(-1., 1.))

    def feed_forward(self, x):
        for neuron in self.layers[x]:
            if x == 0:
                continue
            else:
                neuron.calculate_impulse()

    def decision(self, inputImpulses):
        inputImpulses = self.normalize(inputImpulses)
        for i in range(Brain.INPUT_VALUES_COUNT):
            self.layers[0][i].impulse = inputImpulses[i]
        for i in range(len(self.layers)):
            if i == 0:
                continue
            else:
                self.feed_forward(i)
        mi = 0
        mx = 0
        for i in range(len(self.layers[Brain.HIDDEN_LAYERS + 1])):
            tmp = self.layers[Brain.HIDDEN_LAYERS + 1][i].impulse
            if tmp > mx:
                mx = tmp
                mi = i
        return Brain.DECISIONS[mi]


class Dino:
    def __init__(self, game, population, Id):
        self.game = game
        self.DinoRun = [pygame.image.load("sprites/dinorun0000.png"), pygame.image.load("sprites/dinorun0001.png")]
        self.DinoDeath = pygame.image.load("sprites/dinoDead0000.png")
        self.DinoJump = pygame.image.load("sprites/dinoJump0000.png")
        self.DinoDuck = [pygame.image.load("sprites/dinoduck0000.png"), pygame.image.load("sprites/dinoduck0001.png")]
        self.DinoBasic = pygame.image.load("sprites/dino0000.png")
        self.DinoDeath = pygame.transform.scale(self.DinoDeath, (100, 100))
        self.Rect = self.DinoBasic.get_rect()
        self.PreviousRect = self.Rect

        self.DinoBrain = Brain()
        self.Id = Id
        self.Population = population
        self.result = 0

        self.GroundY = 350
        self.X = 0
        self.Y = 350
        self.Speed = .05
        self.GravityStrength = 1
        self.JumpStrength = .55
        self.JumpTime = 0
        self.StartJumpTime = 0
        self.IsJumping = False
        self.IsDucking = False
        self.RunCount = -5
        self.IsDead = False
        self.CanJump = True

    def Duck(self):
        self.IsDucking = True

    def StopDucking(self):
        self.IsDucking = False

    def Jump(self, IsBig):
        self.JumpTime = 0
        self.StartJumpTime = pygame.time.get_ticks()
        if IsBig:
            self.JumpStrength = 6
            self.GravityStrength = .05
        else:
            self.JumpStrength = 4.5
            self.GravityStrength = .05
        if self.Y == self.GroundY:
            self.IsJumping = True

    def Gravity(self):
        if not self.IsJumping:
            return

        self.Y -= self.JumpStrength
        self.JumpTime = pygame.time.get_ticks() - self.StartJumpTime
        self.CanJump = False
        if self.Y < self.GroundY and self.IsJumping:
            self.Y -= self.JumpStrength - self.GravityStrength * self.JumpTime / 2
        elif self.Y >= self.GroundY:
            self.IsJumping = False
            self.CanJump = True
            self.Y = self.GroundY

    def Died(self, result):
        self.result = result
        self.IsDead = True
        if len(self.game.best) == self.game.iteration - 1:
            self.game.best.append(self.result)
        elif result > self.game.best[self.game.iteration - 1]:
            self.game.best[self.game.iteration - 1] = self.result

    def Think(self):
        next_obstacle_data = self.game.GetClosestObstacle()
        gaps = self.game.GetGapsBetweenTwoClosestObstacles()
        if next_obstacle_data is None or gaps is None:
            return

        speed = self.game.currentSpeed
        dx = next_obstacle_data.Rect.x - self.Rect.x
        dy = next_obstacle_data.Rect.y - self.Rect.y
        brainInput = [self.Rect.x, self.Rect.y,
                      next_obstacle_data.Rect.x, next_obstacle_data.Rect.y,
                      next_obstacle_data.Rect.width, next_obstacle_data.Rect.height,
                      dx, dy, speed, gaps[0], gaps[1]]
        dec = self.DinoBrain.decision(brainInput)

        if dec == Brain.DECISIONS[0]:
            return
        elif dec == Brain.DECISIONS[1] and self.CanJump:
            self.Jump(False)
        elif dec == Brain.DECISIONS[2] and self.CanJump:
            self.Jump(True)
        elif dec == Brain.DECISIONS[3] and not self.IsJumping:
            self.Duck()
        elif dec == Brain.DECISIONS[4] and self.IsDucking:
            self.StopDucking()

    def Display(self):
        # pygame.draw.rect(screen, (0, 0, 255), self.Rect)
        if self.IsDead:
            return
        if self.IsDucking and self.Y == self.GroundY:
            if self.RunCount < 0:
                screen.blit(self.DinoDuck[0],
                            (self.X + self.DinoDuck[0].get_width() / 2,
                             self.Y + .5 * self.DinoDuck[0].get_height() + 10))
                self.Rect = self.DinoDuck[0].get_rect()
                self.Rect.height /= 1.1
                self.Rect.center = (self.X + self.DinoDuck[1].get_width(),
                                    self.Y + self.DinoDuck[1].get_height() + 10)
            else:
                screen.blit(self.DinoDuck[1],
                            (self.X + self.DinoDuck[1].get_width() / 2,
                             self.Y + .5 * self.DinoDuck[1].get_height() + 10))
                self.Rect = self.DinoDuck[1].get_rect()
                self.Rect.height /= 1.1
                self.Rect.center = (self.X + self.DinoDuck[1].get_width(),
                                    self.Y + self.DinoDuck[1].get_height() + 10)
        elif not self.IsJumping:
            if self.RunCount < 0:
                screen.blit(self.DinoRun[0],
                            (self.X + self.DinoRun[0].get_width() / 2, self.Y))
                self.Rect = self.DinoRun[0].get_rect()
                self.Rect.width /= 1.8
                self.Rect.height /= 1.1
                self.Rect.center = (self.X + self.DinoJump.get_width(), self.Y + self.DinoJump.get_height() / 2)
            else:
                screen.blit(self.DinoRun[1],
                            (self.X + self.DinoRun[1].get_width() / 2, self.Y))
                self.Rect = self.DinoRun[1].get_rect()
                self.Rect.width /= 1.8
                self.Rect.height /= 1.1
                self.Rect.center = (self.X + self.DinoJump.get_width(), self.Y + self.DinoJump.get_height() / 2)
        elif self.IsJumping:
            screen.blit(self.DinoJump,
                        (self.X + self.DinoJump.get_width() / 2, self.Y))
            self.Rect = self.DinoJump.get_rect()
            self.Rect.width /= 1.8
            self.Rect.height /= 1.1
            self.Rect.center = (self.X + self.DinoJump.get_width(), self.Y + self.DinoJump.get_height() / 2)
        self.RunCount += self.Speed * self.game.currentSpeed * self.game.clock.get_fps()
        if self.RunCount > 5:
            self.RunCount -= 10

    def Update(self):
        if self.IsDead:
            return
        self.Think()
        self.Gravity()
        self.Display()


class Ground:
    def __init__(self, game):
        self.game = game
        self.X = screen.get_width()
        self.Y = random.randint(425, 475)
        self.Width = random.randint(3, 10)
        self.Height = random.randint(2, 5)

    def display(self):
        pygame.draw.rect(screen, (0, 0, 0), [self.X, self.Y, self.Width, self.Height])

    def move(self):
        self.X -= self.game.currentSpeed * self.game.clock.get_fps()

    def Update(self):
        self.move()
        self.display()


class Bird:
    def __init__(self, game):
        self.game = game
        self.BirdFly = [pygame.image.load("sprites/berd.png"), pygame.image.load("sprites/berd2.png")]
        self.Rect = self.BirdFly[0].get_rect()
        self.Rect.width /= 1.2
        self.Rect.height /= 1.2

        self.X = screen.get_width()
        self.Y = random.randint(200, 350)
        self.FlyCount = -5

    def Move(self):
        self.X -= self.game.currentSpeed * self.game.clock.get_fps()

    def Display(self):
        # pygame.draw.rect(screen, (255, 0, 0), self.Rect)
        if self.FlyCount < 0:
            screen.blit(self.BirdFly[0],
                        (self.X + self.BirdFly[0].get_width() / 2, self.Y + self.BirdFly[0].get_height() / 2))
            self.Rect = self.BirdFly[0].get_rect()
            self.Rect.width /= 1.2
            self.Rect.height /= 2
            self.Rect.center = (self.X + self.BirdFly[0].get_width(), self.Y + self.BirdFly[0].get_height())
        else:
            screen.blit(self.BirdFly[1],
                        (self.X + self.BirdFly[1].get_width() / 2, self.Y + self.BirdFly[1].get_height() / 2))
            self.Rect = self.BirdFly[1].get_rect()
            self.Rect.width /= 1.2
            self.Rect.height /= 2
            self.Rect.center = (self.X + self.BirdFly[1].get_width(), self.Y + self.BirdFly[1].get_height())
        self.FlyCount += self.game.currentSpeed * 0.05 * self.game.clock.get_fps()
        if self.FlyCount > 5:
            self.FlyCount -= 10

    def Update(self):
        self.Move()
        self.Display()


class Cactus:
    def __init__(self, game):
        self.game = game
        self.CactusIndex = random.randint(0, len(Cactusses) - 1)
        self.cactus = Cactusses[self.CactusIndex]
        self.Rect = self.cactus.get_rect()

        self.X = screen.get_width()
        self.Y = 340
        if self.CactusIndex == 0:
            self.Y -= 60

    def Move(self):
        self.X -= self.game.currentSpeed * self.game.clock.get_fps()

    def Display(self):
        # pygame.draw.rect(screen, (255, 0, 0), self.Rect)
        screen.blit(self.cactus,
                    (self.X + self.cactus.get_width() / 2,
                     self.Y + self.cactus.get_height() / 2))
        self.Rect.center = (self.X + self.cactus.get_width(), self.Y + self.cactus.get_height())

    def Update(self):
        self.Move()
        self.Display()


class Game:
    def __init__(self):
        self.groundY = 350
        self.currentSpeed = .1
        self.strongJumpTime = .5
        self.dotSpawnRate = 150
        self.obstacleSpawnRate = 1200
        self.milestones = 5000
        self.currentPoints = 0
        self.best = []
        self.population = Population(self, 50)
        self.population.initialize_population()
        self.ground = []
        self.obstacles = []
        self.iteration = 1
        self.clock = pygame.time.Clock()
        self.startTime = pygame.time.get_ticks()
        self.furthestDinoX = 0

    def SetSpeed(self, speed):
        self.currentSpeed = speed
        self.obstacleSpawnRate -= self.currentSpeed * 10

    def GetClosestObstacle(self):
        if len(self.obstacles) < 1:
            return None

        closestX = 120
        closestObstacle = self.obstacles[0]
        for obstacle in self.obstacles:
            if closestX > obstacle.Rect.x - self.furthestDinoX > 0:
                closestObstacle = obstacle
                closestX = obstacle.Rect.x
        return closestObstacle

    def GetGapsBetweenTwoClosestObstacles(self):
        if len(self.obstacles) < 2:
            return None

        closestX = [120, 120]
        closestTwoObstacles = [self.obstacles[0], self.obstacles[1]]
        for obstacle in self.obstacles:
            if 0 < obstacle.Rect.x - self.furthestDinoX < closestX[0]:
                closestTwoObstacles[1] = closestTwoObstacles[0]
                closestTwoObstacles[0] = obstacle
                closestX[1] = closestX[0]
                closestX[0] = obstacle.X
            elif 0 < obstacle.Rect.x - self.furthestDinoX < closestX[1] and \
                    obstacle.Rect.x - self.furthestDinoX > closestX[0]:
                closestTwoObstacles[1] = obstacle
                closestX[1] = obstacle.X
        return closestTwoObstacles[1].X - closestTwoObstacles[0].X, closestTwoObstacles[1].Y - closestTwoObstacles[0].Y

    def PopulationDied(self):
        if self.population.get_alive_amount() == 0:
            self.reset()

    def reset(self):
        self.population.over()
        self.currentSpeed = .1
        self.dotSpawnRate = 150
        self.obstacleSpawnRate = 1200
        self.milestones = 5000
        self.currentPoints = 0
        self.startTime = pygame.time.get_ticks()
        self.ground = []
        self.obstacles = []
        self.iteration += 1


def Update():
    running = True
    PyDino = Game()

    dotSpawned = 0
    obstacleSpawned = 0

    PyDino.startTime = pygame.time.get_ticks()

    while running:
        screen.fill((255, 255, 255))
        PyDino.clock.tick_busy_loop(60)
        PyDino.PopulationDied()
        PyDino.currentPoints = pygame.time.get_ticks() - PyDino.startTime
        alive = myfont.render(f'dinos: {PyDino.population.get_alive_amount()}', False, (0, 0, 0))
        screen.blit(alive, (1050, 10))
        generation = myfont.render(f'generation: {PyDino.iteration}', False, (0, 0, 0))
        screen.blit(generation, (10, 10))
        score = myfont.render(f'score: {round(PyDino.currentPoints / 100)}', False, (0, 0, 0))
        screen.blit(score, (10, 50))
        highscore = myfont.render(f'highscore: {0 if len(PyDino.best) == 0 else round(max(PyDino.best) / 100)}', False,
                                  (0, 0, 0))
        screen.blit(highscore, (10, 100))
        if PyDino.currentPoints % PyDino.milestones == 0:
            PyDino.currentSpeed += .001

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        timeToSpawnDot = pygame.time.get_ticks() - dotSpawned
        timeToSpawnObstacle = pygame.time.get_ticks() - obstacleSpawned

        if timeToSpawnDot > PyDino.dotSpawnRate:
            PyDino.ground.append(Ground(PyDino))
            dotSpawned = pygame.time.get_ticks()
        if timeToSpawnObstacle + random.uniform(-1, 1) * 5 > PyDino.obstacleSpawnRate:
            if random.randint(0, 1) == 0:
                PyDino.obstacles.append(Bird(PyDino))
            else:
                PyDino.obstacles.append(Cactus(PyDino))
            obstacleSpawned = pygame.time.get_ticks()

        pygame.draw.rect(screen, (0, 0, 0), [0, 425, 1200, 1])
        dotsToDelete = []
        obstaclesToDelete = []
        for dot in PyDino.ground:
            dot.Update()
            if dot.X <= -dot.Width:
                dotsToDelete.append(dot)
        for obs in PyDino.obstacles:
            obs.Update()
            if obs.X <= -obs.Rect.width:
                obstaclesToDelete.append(obs)
        for dino in PyDino.population.population:
            dino.Update()
            if dino.Rect.x > PyDino.furthestDinoX:
                PyDino.furthestDinoX = dino.Rect.x

        for dot in dotsToDelete:
            PyDino.ground.remove(dot)
            del dot
        for obs in obstaclesToDelete:
            PyDino.obstacles.remove(obs)
            del obs

        for dino in PyDino.population.population:
            for obstacle in PyDino.obstacles:
                if dino.Rect.colliderect(obstacle.Rect) and not dino.IsDead:
                    dino.Died(PyDino.currentPoints)

        pygame.display.update()


Update()
