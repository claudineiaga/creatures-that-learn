import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import numpy as np
import random
import sys

SIM_WIDTH   = 800
PANEL_WIDTH = 340
HEIGHT      = 640
FPS         = 60

POP_SIZE      = 80
DNA_SIZE      = 400
MUTATION_RATE = 0.02
MAX_FORCE     = 2.5
TRAIL_LENGTH  = 18

TARGET_POS    = np.array([720.0, 320.0])
TARGET_RADIUS = 22
START_POS     = np.array([40.0,  320.0])

C_BG         = (10,  10,  18)
C_PANEL      = (28,  26,  46)
C_TARGET     = (255, 220,  50)
C_OBSTACLE   = (40,  70, 140)
C_OBS_BORDER = (80, 130, 220)
C_TEXT       = (255, 255, 255)
C_MUTED      = (140, 130, 170)
C_ACCENT     = (150, 100, 255)
C_GRAPH      = ( 60, 230, 140)
C_GOOD       = ( 60, 230, 110)
C_BAD        = (230,  60,  60)
C_WHITE      = (255, 255, 255)


class Obstacle:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)

    def collides(self, x, y, radius=7):
        cx = max(self.rect.left, min(x, self.rect.right))
        cy = max(self.rect.top,  min(y, self.rect.bottom))
        return (x - cx)**2 + (y - cy)**2 < radius**2

    def draw(self, screen):
        pygame.draw.rect(screen, C_OBSTACLE,   self.rect, border_radius=5)
        pygame.draw.rect(screen, C_OBS_BORDER, self.rect, 1, border_radius=5)


class Creature:
    _PALETTE = [
        (255,  60,  80),
        (255, 140,  20),
        (255, 230,   0),
        ( 50, 255, 100),
        (  0, 220, 255),
        ( 60, 120, 255),
        (180,  50, 255),
        (255,  50, 200),
        (255, 180,   0),
        (  0, 255, 200),
        (255, 100, 150),
        (100, 255,  80),
    ]
    _color_idx = 0

    def __init__(self, dna=None):
        self.pos           = START_POS.copy().astype(float)
        self.vel           = np.zeros(2, dtype=float)
        self.fitness       = 0.0
        self.dead          = False
        self.reached       = False
        self.step          = 0
        self.collision_pos = None
        self._best_dist    = float(np.linalg.norm(self.pos - TARGET_POS))
        self._trail        = []
        self.base_color    = Creature._PALETTE[
            Creature._color_idx % len(Creature._PALETTE)]
        Creature._color_idx += 1
        self.dna = self._random_dna() if dna is None else dna.copy()

    def _random_dna(self):
        angles = np.random.uniform(0, 2 * np.pi, DNA_SIZE)
        mags   = np.random.uniform(0.4, MAX_FORCE, DNA_SIZE)
        return np.column_stack([np.cos(angles) * mags,
                                 np.sin(angles) * mags])

    def move(self, obstacles):
        if self.dead or self.reached or self.step >= DNA_SIZE:
            return
        self.vel  = self.vel * 0.72 + self.dna[self.step]
        spd = np.linalg.norm(self.vel)
        if spd > MAX_FORCE * 1.6:
            self.vel = self.vel / spd * MAX_FORCE * 1.6
        self.pos += self.vel
        self.step += 1
        self._trail.append(self.pos.copy())
        if len(self._trail) > TRAIL_LENGTH:
            self._trail.pop(0)
        self.pos[0] = max(6.0, min(self.pos[0], SIM_WIDTH - 6.0))
        self.pos[1] = max(6.0, min(self.pos[1], HEIGHT - 6.0))
        for obs in obstacles:
            if obs.collides(self.pos[0], self.pos[1]):
                self.dead = True
                self.collision_pos = self.pos.copy()
                return
        dist = float(np.linalg.norm(self.pos - TARGET_POS))
        if dist < self._best_dist:
            self._best_dist = dist
        if dist < TARGET_RADIUS:
            self.reached = True

    def calc_fitness(self):
        dist = self._best_dist
        self.fitness = 1.0 / (dist + 1.0)
        if self.reached:
            self.fitness = 10.0 + 1.0 / (self.step + 1)
        if self.dead and not self.reached:
            self.fitness *= 0.3

    def draw(self, screen, idx=0):
        if self.dead:
            return
        px, py = int(self.pos[0]), int(self.pos[1])
        t  = min(self.fitness * 5, 1.0)
        br = 0.70 + 0.30 * t
        r  = min(255, int(self.base_color[0] * br))
        g  = min(255, int(self.base_color[1] * br))
        b  = min(255, int(self.base_color[2] * br))
        radius = 9 if idx < 3 else 6 + (idx % 3)
        n = len(self._trail)
        for i, tpos in enumerate(self._trail):
            frac   = (i + 1) / max(n, 1)
            alpha  = int(220 * frac)
            tr_r   = max(2, int(radius * 0.55 * frac))
            tr_col = (
                min(255, int(self.base_color[0] * 0.85 * frac)),
                min(255, int(self.base_color[1] * 0.85 * frac)),
                min(255, int(self.base_color[2] * 0.85 * frac)),
                alpha
            )
            ts = pygame.Surface((tr_r*2+1, tr_r*2+1), pygame.SRCALPHA)
            pygame.draw.circle(ts, tr_col, (tr_r, tr_r), tr_r)
            screen.blit(ts, (int(tpos[0]) - tr_r, int(tpos[1]) - tr_r))
        if self.reached:
            pygame.draw.circle(screen, (255, 255, 120), (px, py), radius + 8)
        pygame.draw.circle(screen, (r, g, b), (px, py), radius)
        pygame.draw.circle(screen, C_WHITE,   (px, py), 2)


class Population:
    def __init__(self):
        self.creatures       = [Creature() for _ in range(POP_SIZE)]
        self.generation      = 1
        self.best_history    = []
        self.reached_history = []
        self.total_reached   = 0

    def all_done(self):
        return all(c.dead or c.reached or c.step >= DNA_SIZE
                   for c in self.creatures)

    def evaluate(self):
        for c in self.creatures:
            c.calc_fitness()
        self.best_history.append(max(c.fitness for c in self.creatures))
        gen_reached = self.count_reached()
        self.reached_history.append(gen_reached)
        self.total_reached += gen_reached

    def _select(self):
        total = sum(c.fitness for c in self.creatures)
        if total <= 0:
            return random.choice(self.creatures)
        r, acc = random.uniform(0, total), 0
        for c in self.creatures:
            acc += c.fitness
            if acc >= r:
                return c
        return self.creatures[-1]

    def _crossover(self, dna_a, dna_b):
        mask = np.random.rand(DNA_SIZE) > 0.5
        return np.where(mask[:, None], dna_a, dna_b)

    def _mutate(self, dna):
        mask = np.random.rand(DNA_SIZE) < MUTATION_RATE
        if mask.any():
            n      = mask.sum()
            angles = np.random.uniform(0, 2 * np.pi, n)
            mags   = np.random.uniform(0.4, MAX_FORCE, n)
            dna[mask] = np.column_stack([np.cos(angles)*mags,
                                          np.sin(angles)*mags])
        return dna

    def next_generation(self):
        sorted_c = sorted(self.creatures, key=lambda c: c.fitness, reverse=True)
        new_c = [Creature(dna=c.dna) for c in sorted_c[:3]]
        for _ in range(POP_SIZE - 3):
            child_dna = self._crossover(self._select().dna, self._select().dna)
            child_dna = self._mutate(child_dna)
            new_c.append(Creature(dna=child_dna))
        self.creatures = new_c
        self.generation += 1

    def count_reached(self):
        return sum(1 for c in self.creatures if c.reached)


class Particle:
    def __init__(self, x, y):
        self.pos      = np.array([float(x), float(y)])
        angle         = random.uniform(0, 2 * np.pi)
        speed         = random.uniform(0.8, 2.8)
        self.vel      = np.array([np.cos(angle)*speed, np.sin(angle)*speed])
        self.life     = random.randint(12, 28)
        self.max_life = self.life
        self.radius   = random.randint(2, 4)

    def update(self):
        self.pos  += self.vel
        self.vel  *= 0.88
        self.life -= 1

    def draw(self, screen):
        if self.life <= 0:
            return
        alpha = int(255 * (self.life / self.max_life))
        r = 255
        g = int(140 * (self.life / self.max_life))
        b = 30
        s = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (r, g, b, alpha),
                           (self.radius, self.radius), self.radius)
        screen.blit(s, (int(self.pos[0]) - self.radius,
                        int(self.pos[1]) - self.radius))

    @property
    def alive(self):
        return self.life > 0


def draw_target(screen):
    tx, ty = int(TARGET_POS[0]), int(TARGET_POS[1])
    for r, a in [(38,20),(26,45),(TARGET_RADIUS,90)]:
        s = pygame.Surface((r*2,r*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*C_TARGET, a), (r,r), r)
        screen.blit(s, (tx-r, ty-r))
    pygame.draw.circle(screen, C_TARGET, (tx,ty), 11)
    for deg in range(0, 360, 45):
        rad = np.radians(deg)
        ex  = int(tx + np.cos(rad)*17)
        ey  = int(ty + np.sin(rad)*17)
        pygame.draw.line(screen, C_TARGET, (tx,ty), (ex,ey), 2)


def draw_hud(screen, font_sm, generation, frame, mutation_rate):
    base_y = HEIGHT - 80
    bg = pygame.Surface((300, 72), pygame.SRCALPHA)
    bg.fill((10, 10, 18, 220))
    pygame.draw.rect(bg, (80, 60, 160, 140), (0, 0, 300, 72), 1)
    screen.blit(bg, (8, base_y - 4))
    screen.blit(font_sm.render(f"Generation  {generation}",
                                True, (255, 255, 255)), (14, base_y + 4))
    bx, by, bw, bh = 14, base_y + 26, 200, 8
    pygame.draw.rect(screen, (35, 35, 55), (bx, by, bw, bh), border_radius=4)
    prog = min(frame / DNA_SIZE, 1.0)
    for xi in range(int(bw * prog)):
        t  = xi / bw
        rc = int(150 * (1-t))
        gc = int(100 * (1-t) + 220 * t)
        bc = int(255)
        pygame.draw.line(screen, (rc, gc, bc), (bx+xi, by), (bx+xi, by+bh))
    pygame.draw.rect(screen, (100, 80, 200), (bx, by, bw, bh), 1, border_radius=4)
    screen.blit(font_sm.render("progress", True, (150, 130, 200)),
                (bx + bw + 8, by))
    screen.blit(font_sm.render(f"Mutation: {mutation_rate*100:.1f}%  (+/-)",
                                True, (160, 140, 210)), (14, base_y + 46))


def draw_panel(screen, font_lg, font_md, font_sm, pop):
    px, pad = SIM_WIDTH, 20
    pygame.draw.rect(screen, C_PANEL, (px, 0, PANEL_WIDTH, HEIGHT))
    pygame.draw.line(screen, C_ACCENT, (px, 0), (px, HEIGHT), 2)
    screen.blit(font_lg.render("EVOLUTION", True, (255,250,250)),
                (px + pad, 16))
    pygame.draw.line(screen, (100, 70, 200),
                     (px + pad, 46), (px + PANEL_WIDTH - pad, 46), 1)
    reached_now   = pop.count_reached()
    reached_total = pop.total_reached
    stats = [
        ("GENERATION",    str(pop.generation),    (255, 255, 255)),
        ("POPULATION",    str(POP_SIZE),           (255, 255, 255)),
        ("REACHED (GEN)", str(reached_now),
            (80, 255, 120) if reached_now > 0 else (220, 80, 80)),
        ("REACHED (ALL)", str(reached_total),
            (80, 255, 120) if reached_total > 0 else (255, 255, 255)),
        ("DNA STEPS",     str(DNA_SIZE),           (255, 255, 255)),
    ]
    y = 58
    for label, value, val_col in stats:
        screen.blit(font_sm.render(label, True, (245,245,245)), (px + pad, y))
        val = font_md.render(value, True, val_col)
        screen.blit(val, (px + PANEL_WIDTH - pad - val.get_width(), y - 2))
        pygame.draw.line(screen, (60, 50, 90),
                         (px + pad, y + 24), (px + PANEL_WIDTH - pad, y + 24), 1)
        y += 34
    bf_y = 248
    pygame.draw.rect(screen, (28,28,28),
                     (px + pad, bf_y, PANEL_WIDTH - pad*2, 40), border_radius=6)
    pygame.draw.rect(screen, C_ACCENT,
                     (px + pad, bf_y, PANEL_WIDTH - pad*2, 40), 1, border_radius=6)
    screen.blit(font_sm.render("BEST FITNESS", True, C_TEXT),
                (px + pad + 10, bf_y + 13))
    if pop.best_history:
        best   = pop.best_history[-1]
        bf_col = C_GOOD if best > 1.0 else (240, 210, 80)
        bf_val = font_md.render(f"{best:.4f}", True, bf_col)
        screen.blit(bf_val,
                    (px + PANEL_WIDTH - pad - 10 - bf_val.get_width(), bf_y + 11))
    graph_top  = 310
    graph_bot  = HEIGHT - 20
    graph_left = px + pad
    graph_rgt  = px + PANEL_WIDTH - pad
    gw = graph_rgt - graph_left
    gh = graph_bot - graph_top
    screen.blit(font_sm.render("FITNESS / GENERATION", True, (180, 160, 255)),
                (graph_left, graph_top - 18))
    pygame.draw.rect(screen, (0,0,0),
                     (graph_left, graph_top, gw, gh), border_radius=6)
    pygame.draw.rect(screen, (100, 60, 220),
                     (graph_left, graph_top, gw, gh), 1, border_radius=6)
    for i in range(1, 5):
        gy = int(graph_top + gh * i / 5)
        pygame.draw.line(screen, (40, 32, 65),
                         (graph_left + 6, gy), (graph_rgt - 6, gy), 1)
    history = pop.best_history[-80:]
    if len(history) > 1:
        max_v = max(history) or 1
        ws    = gw / max(len(history) - 1, 1)
        pts   = [(int(graph_left + i * ws),
                  int(graph_bot - (v / max_v) * gh * 0.88))
                 for i, v in enumerate(history)]
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]; x2, y2 = pts[i+1]
            t = i / max(len(pts) - 1, 1)
            fill_col = (int(60*t), int(200*(1-t)+255*t),
                        int(255*(1-t)+150*t), 30)
            fill = pygame.Surface((abs(x2-x1)+2, graph_bot-min(y1,y2)+1),
                                   pygame.SRCALPHA)
            fill.fill(fill_col)
            screen.blit(fill, (min(x1,x2), min(y1,y2)))
        for i in range(len(pts) - 1):
            t  = i / max(len(pts) - 1, 1)
            lc = (int(150*(1-t)), int(100*(1-t)+230*t), int(255*(1-t)+200*t))
            pygame.draw.line(screen, lc, pts[i], pts[i+1], 2)
        lx, ly = pts[-1]
        for r, a in [(11,25),(7,55),(4,130)]:
            s = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (0, 255, 200, a), (r, r), r)
            screen.blit(s, (lx-r, ly-r))
        pygame.draw.circle(screen, (0, 255, 200), (lx, ly), 3)
    rh = pop.reached_history[-80:]
    if rh and max(rh) > 0:
        mr = max(rh)
        ws = gw / max(len(rh) - 1, 1)
        for i, rv in enumerate(rh):
            bh_ = int((rv / mr) * gh * 0.20)
            if bh_ > 0:
                bx_ = int(graph_left + i * ws)
                pygame.draw.line(screen, (50, 255, 100),
                                 (bx_, graph_bot-2), (bx_, graph_bot-bh_), 1)


def main():
    global MUTATION_RATE
    pygame.init()
    screen = pygame.display.set_mode((SIM_WIDTH + PANEL_WIDTH, HEIGHT))
    pygame.display.set_caption("Creatures That Learn — Gen 1")
    clock  = pygame.time.Clock()
    font_lg = pygame.font.SysFont("monospace", 22, bold=True)
    font_md = pygame.font.SysFont("monospace", 17, bold=True)
    font_sm = pygame.font.SysFont("monospace", 14)
    W, WT   = 26, 10
    CX      = [320, 560]
    obstacles = [
        Obstacle(0,       0,         SIM_WIDTH, WT),
        Obstacle(0,       HEIGHT-WT, SIM_WIDTH, WT),
        Obstacle(CX[0],   WT,        W, 220),
        Obstacle(CX[0],   410,       W, HEIGHT - 410 - WT),
        Obstacle(CX[1],   WT,        W, 220),
        Obstacle(CX[1],   410,       W, HEIGHT - 410 - WT),
    ]
    pop       = Population()
    frame     = 0
    fast_mode = False
    particles = []
    running   = True
    while running:
        clock.tick(FPS if not fast_mode else 0)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    frame = DNA_SIZE
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    MUTATION_RATE = min(MUTATION_RATE + 0.005, 0.25)
                if event.key == pygame.K_MINUS:
                    MUTATION_RATE = max(MUTATION_RATE - 0.005, 0.001)
                if event.key == pygame.K_f:
                    fast_mode = not fast_mode
        for c in pop.creatures:
            was_dead = c.dead
            c.move(obstacles)
            if c.dead and not was_dead and c.collision_pos is not None:
                if not fast_mode:
                    for _ in range(10):
                        particles.append(
                            Particle(c.collision_pos[0], c.collision_pos[1]))
        frame += 1
        if pop.all_done() or frame >= DNA_SIZE:
            pop.evaluate()
            pop.next_generation()
            frame = 0
            particles.clear()
            pygame.display.set_caption(
                f"Creatures That Learn — Gen {pop.generation}")
        if not fast_mode:
            screen.fill(C_BG)
            for obs in obstacles:
                obs.draw(screen)
            pygame.draw.circle(screen, (65, 65, 90),
                                (int(START_POS[0]), int(START_POS[1])), 11, 2)
            draw_target(screen)
            for idx, c in enumerate(pop.creatures):
                c.draw(screen, idx)
            particles = [p for p in particles if p.alive]
            for p in particles:
                p.update()
                p.draw(screen)
            draw_hud(screen, font_sm, pop.generation, frame, MUTATION_RATE)
            draw_panel(screen, font_lg, font_md, font_sm, pop)
            pygame.display.flip()
        else:
            if frame == 0:
                r = pop.reached_history[-1] if pop.reached_history else 0
                pygame.display.set_caption(
                    f"[FAST] Gen {pop.generation} | Reached: {r}")
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
