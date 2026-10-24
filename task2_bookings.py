"""7042SCN Task 2A/2B/2C — greedy interval scheduling and evaluation.

2A: show earliest_finish(), select_bookings(), and validation; no trace required.
2B: use trace=True; ALL execution commentary originates in code print calls.
2C: compare an optimal schedule with an independently chosen valid schedule.
"""

import math
from numbers import Real


def validate_bookings(starts, finishes):
    """Check that both lists encode finite, nonnegative, nonempty intervals.

    Time zero is the earliest permitted booking time in this implementation.
    Empty input is permitted and has a valid empty result.
    """
    if not isinstance(starts, (list, tuple)) or not isinstance(finishes, (list, tuple)):
        raise TypeError("Starts and finishes must be lists or tuples.")
    if len(starts) != len(finishes):
        raise ValueError("Starts and finishes must have equal lengths.")

    for i, (start, finish) in enumerate(zip(starts, finishes)):
        if (isinstance(start, bool) or isinstance(finish, bool)
                or not isinstance(start, Real) or not isinstance(finish, Real)):
            raise TypeError(f"Booking {i}: times must be real numbers (not booleans).")
        if not math.isfinite(start) or not math.isfinite(finish):
            raise ValueError(f"Booking {i}: times must be finite.")
        if start < 0:
            raise ValueError(f"Booking {i}: start time must be at least zero.")
        if finish <= start:
            raise ValueError(f"Booking {i}: finish must be after start.")


def earliest_finish(starts, finishes, available_from, trace=False):
    """Return the index of the compatible booking that finishes first.

    Tied finishing times are broken in favour of the smaller original index.
    Return -1 if there are no compatible bookings remaining.
    """
    validate_bookings(starts, finishes)
    if not isinstance(available_from, Real) or not math.isfinite(available_from):
        raise ValueError("available_from must be a finite number.")
    best_finish = None
    best_index = -1

    if trace:
        print(f"\nCurrent available_from: {available_from}")

    for i in range(len(starts)):
        if starts[i] >= available_from:
            if trace:
                print(f"  Compatible booking {i}: start={starts[i]}, finish={finishes[i]}")
                print(f"  Finish comparison: booking {i} finishes at {finishes[i]}; "
                      f"current best finish={best_finish}, index={best_index}")

            if (best_finish is None or finishes[i] < best_finish
                    or (finishes[i] == best_finish and i < best_index)):
                best_finish = finishes[i]
                best_index = i
                if trace:
                    print(f"  Updated earliest finish: booking {i} ({best_finish})")
            elif trace:
                print(f"  Kept existing booking {best_index}")

    if trace and best_index == -1:
        print("  No compatible bookings remain (index -1).")
    return best_index


def select_bookings(starts, finishes, trace=False):
    """Choose a maximum-cardinality set of pairwise non-overlapping bookings."""
    validate_bookings(starts, finishes)
    selected = []
    available_from = 0

    while True:
        index = earliest_finish(starts, finishes, available_from, trace=trace)
        if index == -1:
            break

        selected.append(index)
        if trace:
            print(f"  Selected booking {index}: [{starts[index]}, {finishes[index]})")
            print(f"  Advancing available_from from {available_from} to {finishes[index]}")
        # The original bug used starts[index] here; that can admit overlaps.
        available_from = finishes[index]

    if trace:
        print(f"\nFinal selected order: {selected}")
    return selected


def evaluate_schedule(starts, finishes, selected):
    """Evaluate feasibility, cardinality, inter-booking gaps, and final finish.

    Mirrors the supplied evaluator's four returned values. Additional input
    checks stop invalid indices and repeated bookings being silently accepted.
    """
    validate_bookings(starts, finishes)
    if not isinstance(selected, (list, tuple)):
        raise TypeError("selected must be a list or tuple of booking indices.")
    if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(starts)
           for i in selected):
        raise ValueError("Each selected booking index must be a valid integer index.")

    previous_finish = 0
    gaps = []
    valid = len(selected) == len(set(selected))

    for index in selected:
        gap = starts[index] - previous_finish
        gaps.append(gap)
        if gap < 0:
            valid = False
        previous_finish = finishes[index]

    return valid, len(selected), gaps, previous_finish


def run_demonstrations():
    starts = [1, 3, 0, 5, 8, 5]
    finishes = [2, 4, 6, 7, 9, 9]
    print("TASK 2A — PROVIDED EXAMPLE: CORRECTED FUNCTIONS")
    print("Selection:", select_bookings(starts, finishes))
    assert select_bookings(starts, finishes) == [0, 1, 3, 4]

    print("\nTASK 2B — COMPLETE EXECUTION TRACE: PROVIDED EXAMPLE")
    select_bookings(starts, finishes, trace=True)

    # Booking 2 and 3 tie on finish time 5; booking 2 must win.
    extra_starts = [0, 1, 3, 3, 5, 5, 7, 8, 9]
    extra_finishes = [2, 4, 5, 5, 7, 8, 9, 10, 11]
    print("\nTASK 2B — COMPLETE EXECUTION TRACE: OWN EXAMPLE WITH TIE")
    extra_optimal = select_bookings(extra_starts, extra_finishes, trace=True)
    assert extra_optimal == [0, 2, 4, 6, 8]

    print("\nTASK 2C — OPTIMISATION AND EVALUATION")
    supplied_optimal = select_bookings(starts, finishes)
    print("Provided starts:", starts)
    print("Provided finishes:", finishes)
    print("Provided selected:", supplied_optimal)
    print("Provided evaluated (valid, count, gaps, finish):",
          evaluate_schedule(starts, finishes, supplied_optimal))

    # A distinct non-overlapping schedule of four instead of five bookings.
    alternative = [0, 3, 5, 8]
    opt_eval = evaluate_schedule(extra_starts, extra_finishes, extra_optimal)
    alternative_eval = evaluate_schedule(extra_starts, extra_finishes, alternative)
    print("Own starts:", extra_starts)
    print("Own finishes:", extra_finishes)
    print("Greedy order:", extra_optimal)
    print("Greedy evaluation:", opt_eval)
    print("Alternative order:", alternative)
    print("Alternative evaluation:", alternative_eval)
    assert opt_eval[0] and alternative_eval[0]
    assert alternative_eval[1] <= opt_eval[1]
    print("Both schedules valid; alternative does not contain more bookings.")


if __name__ == "__main__":
    run_demonstrations()
