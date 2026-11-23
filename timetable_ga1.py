"""Timetable generation with a genetic algorithm.

Chromosome : a list of genes, one per lesson.
Gene       : (time slot, room).
Fitness    : weighted penalties (0 = perfect timetable).
"""
import random

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

random.seed(42)

# ---------- Data ----------
DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu"]
SLOTS_PER_DAY = 5
N_SLOTS = len(DAYS) * SLOTS_PER_DAY

ROOMS = {"A1": 40, "A2": 30, "B1": 60, "Labo": 25}  # name -> capacity

# (subject, teacher, group, n_sessions, group_size)
COURSES = [
    ("Maths",   "Benali",  "G1", 3, 35),
    ("Maths",   "Benali",  "G2", 3, 28),
    ("Algorithms",    "Kaci",    "G1", 3, 35),
    ("Algorithms",    "Kaci",    "G2", 3, 28),
    ("Databases",     "Sahraoui","G1", 2, 35),
    ("Databases",     "Sahraoui","G2", 2, 28),
    ("AI",      "Zerrouki","G1", 3, 35),
    ("AI",      "Zerrouki","G2", 3, 28),
    ("English", "Haddad",  "G1", 2, 35),
    ("English", "Haddad",  "G2", 2, 28),
    ("Networks", "Kaci",    "G1", 2, 35),
    ("Networks", "Benali",  "G2", 2, 28),
]

# Expansion: one entry per session
LESSONS = []
for subject, teacher, group, n, size in COURSES:
    for _ in range(n):
        LESSONS.append({"subject": subject, "teacher": teacher,
                        "group": group, "size": size})
N = len(LESSONS)
ROOM_NAMES = list(ROOMS)

# ---------- GA parameters ----------
POP_SIZE = 150
GENERATIONS = 500
TOURNAMENT_K = 4
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.03
ELITE = 5

W_HARD = 100   # hard conflicts
W_SOFT = 1     # preferences


# ---------- GA ----------
def random_gene():
    return (random.randrange(N_SLOTS), random.choice(ROOM_NAMES))


def random_individual():
    return [random_gene() for _ in range(N)]


def fitness(ind):
    """Return the total penalty (to minimise)."""
    penalty = 0
    teacher_at, group_at, room_at = set(), set(), set()

    for lesson, (slot, room) in zip(LESSONS, ind):
        t, g = (lesson["teacher"], slot), (lesson["group"], slot)
        r = (room, slot)

        # Hard conflicts
        if t in teacher_at:
            penalty += W_HARD
        if g in group_at:
            penalty += W_HARD
        if r in room_at:
            penalty += W_HARD
        if ROOMS[room] < lesson["size"]:
            penalty += W_HARD

        teacher_at.add(t)
        group_at.add(g)
        room_at.add(r)

    # Soft constraints: at most 3 sessions/day per group,
    # and avoid idle gaps during the day.
    by_group_day = {}
    for lesson, (slot, _) in zip(LESSONS, ind):
        by_group_day.setdefault((lesson["group"], slot // SLOTS_PER_DAY), []).append(
            slot % SLOTS_PER_DAY)
    for slots in by_group_day.values():
        if len(slots) > 3:
            penalty += W_SOFT * (len(slots) - 3) * 5
        slots.sort()
        penalty += W_SOFT * (slots[-1] - slots[0] + 1 - len(slots))  # trous

    return penalty


def tournament(pop, fits):
    """Return the INDEX of the tournament winner."""
    return min(random.sample(range(len(pop)), TOURNAMENT_K), key=lambda i: fits[i])


def crossover(p1, p2):
    """Return (child1, child2, crossover_applied)."""
    if random.random() > CROSSOVER_RATE:
        return p1[:], p2[:], False
    c1, c2 = [], []
    for g1, g2 in zip(p1, p2):  # uniform crossover
        if random.random() < 0.5:
            c1.append(g1); c2.append(g2)
        else:
            c1.append(g2); c2.append(g1)
    return c1, c2, True


def mutate(ind):
    """Mutate the individual in place. Return the list of mutations."""
    changes = []
    for i in range(N):
        if random.random() < MUTATION_RATE:
            old = ind[i]
            slot, room = old
            r = random.random()
            if r < 0.4:
                ind[i] = (random.randrange(N_SLOTS), room)
            elif r < 0.7:
                ind[i] = (slot, random.choice(ROOM_NAMES))
            else:
                ind[i] = random_gene()
            changes.append((i, old, ind[i]))
    return changes


# ---------- Evolution log ----------
LOG_FILE = "evolution_log.txt"
DETAILED_GENERATIONS = 2   # generations 1 and 2 in detail
DETAILED_OFFSPRING = 6     # number of children detailed per generation
_log_lines = []


def log(msg=""):
    print(msg)
    _log_lines.append(msg)


def slot_name(slot):
    return f"{DAYS[slot // SLOTS_PER_DAY]}-S{slot % SLOTS_PER_DAY + 1}"


def gene_str(g):
    return f"{slot_name(g[0])}/{g[1]}"


def lesson_name(i):
    l = LESSONS[i]
    return f"#{i:02d} {l['subject']}-{l['group']}"


def run():
    pop = [random_individual() for _ in range(POP_SIZE)]
    best, best_fit, best_gen = None, float("inf"), 0
    history = []  # (gen, best, avg)

    log("=" * 70)
    log("EVOLUTION LOG - GENETIC ALGORITHM (timetable)")
    log(f"Population={POP_SIZE} | Tournament k={TOURNAMENT_K} | "
        f"Crossover={CROSSOVER_RATE} | Mutation={MUTATION_RATE} | Elites={ELITE}")
    log("=" * 70)

    for gen in range(GENERATIONS):
        fits = [fitness(ind) for ind in pop]
        order = sorted(range(POP_SIZE), key=lambda i: fits[i])
        avg = sum(fits) / POP_SIZE
        if fits[order[0]] < best_fit:
            best, best_fit, best_gen = pop[order[0]][:], fits[order[0]], gen
        history.append((gen, fits[order[0]], avg))

        detailed = gen <= DETAILED_GENERATIONS
        if gen == 0:
            log("\n[Generation 0] Initial population (random)")
        else:
            log(f"\n[Generation {gen}]")
        log(f"  Best={fits[order[0]]} | Average={avg:.1f} | Worst={fits[order[-1]]}")
        if detailed:
            log("  Top 5 : " + ", ".join(f"ind{i}({fits[i]})" for i in order[:5]))

        if best_fit == 0:
            log(f"\n>>> PERFECT SOLUTION (penalty 0) at generation {gen}")
            break

        # Elitism
        new_pop = [pop[i][:] for i in order[:ELITE]]
        if detailed:
            log(f"  Elites kept : " +
                ", ".join(f"ind{i}({fits[i]})" for i in order[:ELITE]))

        n_mut, n_off = 0, 0
        while len(new_pop) < POP_SIZE:
            i1, i2 = tournament(pop, fits), tournament(pop, fits)
            c1, c2, crossed = crossover(pop[i1], pop[i2])
            m1 = mutate(c1)
            m2 = mutate(c2) if len(new_pop) + 1 < POP_SIZE else []  # c2 discarded if population is full
            n_mut += len(m1) + len(m2)
            n_new = 2 if len(new_pop) + 1 < POP_SIZE else 1

            if detailed and n_off < DETAILED_OFFSPRING:
                log(f"  Children {n_off + 1}-{n_off + 2} : selected parents "
                    f"ind{i1}(fit={fits[i1]}) x ind{i2}(fit={fits[i2]}) | "
                    f"crossover={'yes' if crossed else 'no (copy)'}")
                for name, ms in (("child A", m1), ("child B", m2)):
                    if not ms:
                        log(f"     {name} : no mutation")
                    for idx, old, new in ms:
                        tag = " (unchanged)" if old == new else ""
                        log(f"     {name} : MUTATION {lesson_name(idx)} : "
                            f"{gene_str(old)} -> {gene_str(new)}{tag}")
            n_off += n_new
            new_pop.append(c1)
            if len(new_pop) < POP_SIZE:
                new_pop.append(c2)

        if detailed:
            log(f"  Total mutations in this generation : {n_mut} "
                f"(over {n_off} children)")
        elif gen % 10 == 0:
            log(f"  Mutations in this generation : {n_mut}")
        pop = new_pop

    log("\n" + "=" * 70)
    log(f"Best solution found at generation {best_gen} "
        f"(penalty = {best_fit})")
    log("=" * 70)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines))
    save_convergence_png(history)
    return best, best_fit


def save_convergence_png(history, path="convergence.png"):
    gens = [h[0] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gens, [h[2] for h in history], label="Population average", alpha=0.7)
    ax.plot(gens, [h[1] for h in history], label="Best individual", linewidth=2)
    ax.set_yscale("symlog")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Penalty (symlog scale)")
    ax.set_title("Convergence of the genetic algorithm")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_timetable(ind):
    groups = sorted({l["group"] for l in LESSONS})
    for g in groups:
        print(f"\n=== Group {g} ===")
        grid = {}
        for lesson, (slot, room) in zip(LESSONS, ind):
            if lesson["group"] == g:
                grid[slot] = f'{lesson["subject"]} ({lesson["teacher"]}, {room})'
        for d, day in enumerate(DAYS):
            row = [grid.get(d * SLOTS_PER_DAY + s, "-") for s in range(SLOTS_PER_DAY)]
            print(f"{day:4s} | " + " | ".join(f"{c:34s}" for c in row))


def save_timetable_png(ind, prefix="timetable"):
    """Save one PNG per group: timetable_G1.png, timetable_G2.png..."""
    subjects = sorted({l["subject"] for l in LESSONS})
    cmap = plt.get_cmap("Pastel1")
    color = {s: cmap(i % 9) for i, s in enumerate(subjects)}

    for g in sorted({l["group"] for l in LESSONS}):
        cells = {}
        for lesson, (slot, room) in zip(LESSONS, ind):
            if lesson["group"] == g:
                cells[slot] = (f'{lesson["subject"]}\n{lesson["teacher"]} | {room}',
                               color[lesson["subject"]])

        text = [[cells.get(d * SLOTS_PER_DAY + s, ("", "white"))[0]
                 for s in range(SLOTS_PER_DAY)] for d in range(len(DAYS))]
        face = [[cells.get(d * SLOTS_PER_DAY + s, ("", "white"))[1]
                 for s in range(SLOTS_PER_DAY)] for d in range(len(DAYS))]

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.axis("off")
        tbl = ax.table(cellText=text, cellColours=face,
                       rowLabels=DAYS,
                       colLabels=[f"Slot {i + 1}" for i in range(SLOTS_PER_DAY)],
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 3.2)
        ax.set_title(f"Timetable - Group {g}", fontsize=14, pad=10)
        fig.savefig(f"{prefix}_{g}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Image saved: {prefix}_{g}.png")


if __name__ == "__main__":
    solution, score = run()
    print(f"\nFinal penalty : {score}")
    print_timetable(solution)
    save_timetable_png(solution)