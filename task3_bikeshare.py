"""7042SCN Task 3 — reproducible 120-minute bike-share simulation.

The assignment mainly asks for DESIGN, including pseudocode, staged testing, and
an extension to two stations. This complete executable implementation supplies
verified evidence to support the eventual written design discussion.

Die model (fair six-sided die, independently rolled each minute):
  1 or 2 => hire attempt (probability 2/6)
  3      => return attempt (probability 1/6)
  4,5,6  => no event       (probability 3/6)

The optional 'rolls' sequence is a deterministic test input; otherwise a
local random.Random(seed) generator is used to make runs reproducible.
"""

import csv
import random
from pathlib import Path

OUTCOMES = (
    "successful_hires", "successful_returns",
    "rejected_hires", "rejected_returns", "no_change"
)


def require_integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def check_station_settings(capacity, initial_bikes, minutes, seed):
    require_integer("capacity", capacity, minimum=1)
    require_integer("initial_bikes", initial_bikes)
    require_integer("minutes", minutes)
    require_integer("seed", seed)
    if initial_bikes > capacity:
        raise ValueError("Initial bicycles cannot exceed station capacity.")


def check_rolls(rolls, minutes):
    if rolls is None:
        return None
    rolls = list(rolls)
    if len(rolls) != minutes:
        raise ValueError("Provide exactly one die roll for each minute.")
    if any(type(roll) is not int or not 1 <= roll <= 6 for roll in rolls):
        raise ValueError("Die outcomes must be integers from 1 through 6.")
    return rolls


def process_event(bikes, capacity, roll):
    """Apply one roll without ever violating 0 <= bikes <= capacity."""
    require_integer("capacity", capacity, minimum=1)
    require_integer("bikes", bikes)
    if bikes > capacity:
        raise ValueError("Current bicycles exceed capacity.")
    if type(roll) is not int or not 1 <= roll <= 6:
        raise ValueError("roll must be an integer from 1 through 6.")

    if roll in (1, 2):
        if bikes > 0:
            return bikes - 1, "successful_hires"
        return bikes, "rejected_hires"
    if roll == 3:
        if bikes < capacity:
            return bikes + 1, "successful_returns"
        return bikes, "rejected_returns"
    return bikes, "no_change"


def simulate_station(capacity=10, initial_bikes=6, minutes=120, seed=42, rolls=None):
    """Return minute history, event counters, and initial/final state."""
    check_station_settings(capacity, initial_bikes, minutes, seed)
    rolls = check_rolls(rolls, minutes)
    generator = random.Random(seed)
    bikes = initial_bikes
    counters = {key: 0 for key in OUTCOMES}
    history = []

    for minute in range(1, minutes + 1):
        roll = rolls[minute - 1] if rolls is not None else generator.randint(1, 6)
        bikes, outcome = process_event(bikes, capacity, roll)
        counters[outcome] += 1
        history.append({"minute": minute, "die": roll,
                        "outcome": outcome, "bikes_after": bikes})

        assert 0 <= bikes <= capacity, "Station capacity invariant violated"

    assert len(history) == minutes
    assert sum(counters.values()) == minutes
    assert bikes == initial_bikes - counters["successful_hires"] + counters["successful_returns"]
    return {
        "capacity": capacity, "initial_bikes": initial_bikes, "minutes": minutes,
        "seed": seed if rolls is None else "scripted",
        "final_bikes": bikes, "counters": counters, "history": history,
    }


def balance_transfer(bikes_a, bikes_b, capacity, max_transfer=2):
    """Move up to two bikes from fuller to emptier without over-correcting.

    Differences of zero or one result in no transfer because moving a bicycle
    would not improve the balance. The vehicle is assumed instantaneous at
    the end of every 30th minute (after both customer events).
    """
    require_integer("max_transfer", max_transfer)
    if not (0 <= bikes_a <= capacity and 0 <= bikes_b <= capacity):
        raise ValueError("Invalid station inventory.")
    difference = abs(bikes_a - bikes_b)
    transferable = min(max_transfer, difference // 2)
    total_before = bikes_a + bikes_b
    if bikes_a > bikes_b:
        transferable = min(transferable, bikes_a, capacity - bikes_b)
        bikes_a -= transferable
        bikes_b += transferable
        direction = "A_to_B" if transferable else "none"
    elif bikes_b > bikes_a:
        transferable = min(transferable, bikes_b, capacity - bikes_a)
        bikes_b -= transferable
        bikes_a += transferable
        direction = "B_to_A" if transferable else "none"
    else:
        direction = "none"
    assert total_before == bikes_a + bikes_b, "Transfer must conserve bicycles"
    return bikes_a, bikes_b, transferable, direction


def simulate_two_stations(capacity=10, initial_a=6, initial_b=6, minutes=120,
                          seed=42, rolls_a=None, rolls_b=None):
    """Roll independently for A/B each minute; rebalance at 30-minute intervals."""
    check_station_settings(capacity, initial_a, minutes, seed)
    check_station_settings(capacity, initial_b, minutes, seed)
    if (rolls_a is None) != (rolls_b is None):
        raise ValueError("Supply BOTH scripted roll streams or neither.")
    rolls_a = check_rolls(rolls_a, minutes)
    rolls_b = check_rolls(rolls_b, minutes)
    # A different reproducible stream for B; the first station uses seed.
    rng_a = random.Random(seed)
    rng_b = random.Random(seed + 1)
    bikes_a, bikes_b = initial_a, initial_b
    counters_a = {key: 0 for key in OUTCOMES}
    counters_b = {key: 0 for key in OUTCOMES}
    history, transfers = [], []

    for minute in range(1, minutes + 1):
        roll_a = rolls_a[minute - 1] if rolls_a is not None else rng_a.randint(1, 6)
        roll_b = rolls_b[minute - 1] if rolls_b is not None else rng_b.randint(1, 6)
        bikes_a, outcome_a = process_event(bikes_a, capacity, roll_a)
        bikes_b, outcome_b = process_event(bikes_b, capacity, roll_b)
        counters_a[outcome_a] += 1
        counters_b[outcome_b] += 1

        transfer_count, transfer_direction = 0, "none"
        if minute % 30 == 0:
            before_a, before_b = bikes_a, bikes_b
            bikes_a, bikes_b, transfer_count, transfer_direction = balance_transfer(
                bikes_a, bikes_b, capacity
            )
            transfers.append({"minute": minute, "from_a": before_a, "from_b": before_b,
                              "moved": transfer_count, "direction": transfer_direction,
                              "to_a": bikes_a, "to_b": bikes_b})

        assert 0 <= bikes_a <= capacity and 0 <= bikes_b <= capacity
        history.append({"minute": minute, "die_a": roll_a, "outcome_a": outcome_a,
                        "die_b": roll_b, "outcome_b": outcome_b,
                        "transfer": transfer_count, "direction": transfer_direction,
                        "bikes_a_after": bikes_a, "bikes_b_after": bikes_b})

    expected_total = (initial_a + initial_b - counters_a["successful_hires"]
                      - counters_b["successful_hires"]
                      + counters_a["successful_returns"]
                      + counters_b["successful_returns"])
    assert bikes_a + bikes_b == expected_total, "Total bike conservation failed"
    assert len(transfers) == minutes // 30
    return {"initial_a": initial_a, "initial_b": initial_b,
            "final_a": bikes_a, "final_b": bikes_b, "minutes": minutes,
            "counters_a": counters_a, "counters_b": counters_b,
            "history": history, "transfers": transfers}


def export_csv(path, rows):
    """Save text-based minute records for direct inspection (not screenshots)."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_demonstrations():
    base = Path(__file__).resolve().parent / "outputs"
    result = simulate_station()
    export_csv(base / "task3_one_station_120min.csv", result["history"])
    print("TASK 3 — REPRODUCIBLE 120-MINUTE SINGLE-STATION SIMULATION")
    print("Parameters: capacity=10, initial=6, minutes=120, seed=42")
    print("Final inventory:", result["final_bikes"])
    print("Counters:", result["counters"])
    print("First ten minute records:")
    for item in result["history"][:10]:
        print(item)
    print("Invariants: bounds, 120 events, and bike stock conservation PASSED")

    two = simulate_two_stations(initial_a=10, initial_b=0, minutes=120,
                                rolls_a=[4] * 120, rolls_b=[4] * 120)
    export_csv(base / "task3_two_station_120min.csv", two["history"])
    print("\nTASK 3 — CONTROLLED TWO-STATION TRANSFER TEST")
    print("Initial inventories: A=10, B=0; all die rolls are 4 (no customers)")
    print("Vehicle can move at most two bicycles every 30 minutes.")
    for transfer in two["transfers"]:
        print(transfer)
    print("Final inventories:", two["final_a"], two["final_b"])
    print("Two-station conservation invariant PASSED")


if __name__ == "__main__":
    run_demonstrations()
