# TP01 — Timetable generation with a genetic algorithm

Automatic generation of university timetables using a genetic algorithm (GA),
in pure Python + matplotlib for visualization.

## Folder contents

| File | Purpose |
|---|---|
| `timetable_ga.py` | Main script: GA + console output + PNG export |
| `main.py` | Minimal project entry point (hello world) |
| `pyproject.toml` | Project metadata and dependencies (uv) |
| `.python-version` | Pinned Python version (3.13) |
| `timetable_G1.png`, `timetable_G2.png` | Generated timetables (one PNG per group) |
| `.venv/` | Local virtual environment |

## Installation

```bash
uv sync        # creates .venv and installs dependencies (matplotlib)
```

Or with plain pip:

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install matplotlib
```

## Usage

```bash
uv run python timetable_ga.py
# or
.venv/Scripts/python timetable_ga.py
```

The script:

1. evolves a population of timetables for 500 generations;
2. prints the best solution found to the console;
3. saves one PNG per group: `timetable_G1.png`, `timetable_G2.png`.

## How it works

- **Chromosome**: a list with one gene per lesson (`LESSONS`, expanded from `COURSES`).
- **Gene**: a `(slot, room)` pair, e.g. `(7, "A1")`.
- **Fitness**: a sum of penalties to minimize (0 = perfect timetable):
  - **hard constraints** (weight 100): teacher conflict, group conflict,
    room conflict, room too small for the class size;
  - **soft constraints** (weight 1): more than 3 lessons/day per group,
    gaps within a day.
- **Operators**: tournament selection (k=4), uniform crossover
  (rate 0.9), mutation (rate 0.03), elitism (top 5 kept).

## Data

- 5 days (`Dim`–`Jeu`) × 5 slots = 25 slots.
- 4 rooms: `A1` (40), `A2` (30), `B1` (60), `Labo` (25).
- 12 courses (subject, teacher, group, number of sessions, class size),
  2 groups `G1` (35 students) and `G2` (28 students).

## Customization

Parameters are set at the top of `timetable_ga.py`:
`POP_SIZE`, `GENERATIONS`, `TOURNAMENT_K`, `CROSSOVER_RATE`,
`MUTATION_RATE`, `ELITE`, `W_HARD`, `W_SOFT`, as well as the data
`DAYS`, `SLOTS_PER_DAY`, `ROOMS` and `COURSES`.
