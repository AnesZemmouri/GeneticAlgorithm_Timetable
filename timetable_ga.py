"""Emploi du temps par algorithme génétique.

Chromosome : une liste de gènes, un par séance (lesson).
Gène       : (créneau, salle).
Fitness    : pénalités pondérées (0 = emploi du temps parfait).
"""
import random

import matplotlib
matplotlib.use("Agg")  # pas besoin d'écran
import matplotlib.pyplot as plt

random.seed(42)

# ---------- Données ----------
DAYS = ["Dim", "Lun", "Mar", "Mer", "Jeu"]
SLOTS_PER_DAY = 5
N_SLOTS = len(DAYS) * SLOTS_PER_DAY

ROOMS = {"A1": 40, "A2": 30, "B1": 60, "Labo": 25}  # nom -> capacité

# (matière, enseignant, groupe, nb_séances, effectif)
COURSES = [
    ("Maths",   "Benali",  "G1", 3, 35),
    ("Maths",   "Benali",  "G2", 3, 28),
    ("Algo",    "Kaci",    "G1", 3, 35),
    ("Algo",    "Kaci",    "G2", 3, 28),
    ("BDD",     "Sahraoui","G1", 2, 35),
    ("BDD",     "Sahraoui","G2", 2, 28),
    ("IA",      "Zerrouki","G1", 3, 35),
    ("IA",      "Zerrouki","G2", 3, 28),
    ("Anglais", "Haddad",  "G1", 2, 35),
    ("Anglais", "Haddad",  "G2", 2, 28),
    ("Réseaux", "Kaci",    "G1", 2, 35),
    ("Réseaux", "Benali",  "G2", 2, 28),
]

# Expansion : une entrée par séance
LESSONS = []
for subject, teacher, group, n, size in COURSES:
    for _ in range(n):
        LESSONS.append({"subject": subject, "teacher": teacher,
                        "group": group, "size": size})
N = len(LESSONS)
ROOM_NAMES = list(ROOMS)

# ---------- Paramètres GA ----------
POP_SIZE = 150
GENERATIONS = 500
TOURNAMENT_K = 4
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.03
ELITE = 5

W_HARD = 100   # conflits durs
W_SOFT = 1     # préférences


# ---------- GA ----------
def random_gene():
    return (random.randrange(N_SLOTS), random.choice(ROOM_NAMES))


def random_individual():
    return [random_gene() for _ in range(N)]


def fitness(ind):
    """Retourne la pénalité totale (à minimiser)."""
    penalty = 0
    teacher_at, group_at, room_at = set(), set(), set()

    for lesson, (slot, room) in zip(LESSONS, ind):
        t, g = (lesson["teacher"], slot), (lesson["group"], slot)
        r = (room, slot)

        # Conflits durs
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

    # Contraintes douces : pas plus de 3 séances/jour par groupe,
    # et éviter les trous dans la journée.
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
    best = min(random.sample(range(len(pop)), TOURNAMENT_K), key=lambda i: fits[i])
    return pop[best]


def crossover(p1, p2):
    if random.random() > CROSSOVER_RATE:
        return p1[:], p2[:]
    c1, c2 = [], []
    for g1, g2 in zip(p1, p2):  # croisement uniforme
        if random.random() < 0.5:
            c1.append(g1); c2.append(g2)
        else:
            c1.append(g2); c2.append(g1)
    return c1, c2


def mutate(ind):
    for i in range(N):
        if random.random() < MUTATION_RATE:
            slot, room = ind[i]
            r = random.random()
            if r < 0.4:
                ind[i] = (random.randrange(N_SLOTS), room)
            elif r < 0.7:
                ind[i] = (slot, random.choice(ROOM_NAMES))
            else:
                ind[i] = random_gene()
    return ind


def run():
    pop = [random_individual() for _ in range(POP_SIZE)]
    best, best_fit = None, float("inf")

    for gen in range(GENERATIONS):
        fits = [fitness(ind) for ind in pop]
        i_best = min(range(POP_SIZE), key=lambda i: fits[i])
        if fits[i_best] < best_fit:
            best, best_fit = pop[i_best][:], fits[i_best]
        if gen % 25 == 0:
            print(f"Gen {gen:4d} | best = {best_fit}")
        if best_fit == 0:
            print(f"Solution parfaite à la génération {gen}")
            break

        # Élitisme
        order = sorted(range(POP_SIZE), key=lambda i: fits[i])
        new_pop = [pop[i][:] for i in order[:ELITE]]

        while len(new_pop) < POP_SIZE:
            c1, c2 = crossover(tournament(pop, fits), tournament(pop, fits))
            new_pop.append(mutate(c1))
            if len(new_pop) < POP_SIZE:
                new_pop.append(mutate(c2))
        pop = new_pop

    return best, best_fit


def print_timetable(ind):
    groups = sorted({l["group"] for l in LESSONS})
    for g in groups:
        print(f"\n=== Groupe {g} ===")
        grid = {}
        for lesson, (slot, room) in zip(LESSONS, ind):
            if lesson["group"] == g:
                grid[slot] = f'{lesson["subject"]} ({lesson["teacher"]}, {room})'
        for d, day in enumerate(DAYS):
            row = [grid.get(d * SLOTS_PER_DAY + s, "-") for s in range(SLOTS_PER_DAY)]
            print(f"{day:4s} | " + " | ".join(f"{c:34s}" for c in row))


def save_timetable_png(ind, prefix="timetable"):
    """Sauvegarde un PNG par groupe : timetable_G1.png, timetable_G2.png..."""
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
        ax.set_title(f"Emploi du temps - Groupe {g}", fontsize=14, pad=10)
        fig.savefig(f"{prefix}_{g}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Image sauvegardée : {prefix}_{g}.png")


if __name__ == "__main__":
    solution, score = run()
    print(f"\nPénalité finale : {score}")
    print_timetable(solution)
    save_timetable_png(solution)