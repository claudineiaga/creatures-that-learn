# 🧬 Creatures That Learn — Genetic Algorithm Visualized

80 creatures. Zero rules. They figured it out on their own.

A genetic algorithm simulation where creatures evolve to navigate through barriers and reach a target — with no hardcoded path. Each creature has a strand of DNA (400 random force vectors). Generation 1 is pure chaos. By generation 50, they've learned to navigate with surprising elegance.

Built in Python with Pygame. No machine learning libraries required.

[![Watch the video](https://img.shields.io/badge/Watch%20on-YouTube-red?style=for-the-badge&logo=youtube)](YOUR_VIDEO_LINK_HERE)

---

## Preview

| Generation 1 | Generation 50 |
|---|---|
| Pure chaos — random movement | Evolved — they found the path |

> Replace this section with a GIF or screenshots from your simulation

---

## How It Works

Each creature carries a **DNA** — a list of 400 force vectors applied as acceleration, one per frame. After each generation:

1. **Fitness** is calculated based on how close each creature got to the target
2. **Selection** picks parents proportional to their fitness (roulette wheel)
3. **Crossover** mixes DNA from two parents into one offspring
4. **Mutation** randomly replaces ~2% of genes to maintain diversity
5. The top 3 creatures survive unchanged (**elitism**)

No one tells the creatures how to move. They discover the path through evolution.

---

## Quick Start

```bash
pip install pygame numpy
python creatures_evolution.py
```

### Controls

| Key | Action |
|---|---|
| `SPACE` | Skip to next generation |
| `F` | Toggle fast mode (no rendering) |
| `+` / `-` | Adjust mutation rate live |

---

## Parameters You Can Tweak

All parameters are at the top of the file. Change them and see what happens:

| Parameter | Default | What it does |
|---|---|---|
| `POP_SIZE` | 80 | Creatures per generation |
| `DNA_SIZE` | 400 | Movement steps per creature |
| `MUTATION_RATE` | 0.02 | Chance of random gene change (2%) |
| `MAX_FORCE` | 2.5 | Maximum force per gene |
| `TARGET_RADIUS` | 22 | How close counts as "reached" |

---

## Project Structure

```
creatures_evolution.py    ← Run this. Everything in one file.
README.md                 ← You're reading it.
```

The code is intentionally kept in a single file for simplicity. Classes inside:

- `Creature` — DNA, movement physics, fitness calculation, rendering
- `Population` — Selection, crossover, mutation, generation cycling
- `Obstacle` — Barrier collision detection
- `Particle` — Visual collision sparks

---

## Want the Full Breakdown?

The free code here runs perfectly. If you want to understand **every line** — why `1/(dist+1)` was chosen over other formulas, why the damping factor is 0.72, what alternatives exist for each algorithm, and how to extend this into neural networks or moving obstacles:

📦 **[Full commented code + PDF guide →](YOUR_GUMROAD_LINK_HERE)**

Includes:
- Fully commented source code with line-by-line explanations
- PDF guide covering the math, the reasoning, and 5 extension ideas
- Alternative approaches for each algorithm component

---

## Tech Stack

- Python 3.10+
- Pygame 2.x
- NumPy

---

## What's Next

This is the first project on **CodePulseAI** — a silent coding channel where Python + AI projects speak for themselves.

→ [Subscribe on YouTube](YOUR_CHANNEL_LINK_HERE)

---

## License

MIT — use it, modify it, learn from it. If you build something cool with it, I'd love to see it in the video comments.
