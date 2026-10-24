"""Independent automated tests for 7042SCN Tasks 1-4.

Run: python -m unittest -v test_all_tasks
Tests verify output contracts, invalid inputs, boundary conditions, tie
handling, a brute-force optimisation oracle, and leakage-safe predictions.
"""

import itertools
import random
import unittest
from pathlib import Path

from task1_ipv4 import convert_ip
from task2_bookings import earliest_finish, select_bookings, evaluate_schedule
from task3_bikeshare import (
    process_event, simulate_station, balance_transfer, simulate_two_stations
)
from task4_text_classification import load_data, evaluate, CATEGORIES

ROOT = Path(__file__).resolve().parent


class TestTask1(unittest.TestCase):
    def test_supplied_assertions(self):
        self.assertEqual(convert_ip("192.168.1.1"),
                         "11000000.10101000.00000001.00000001")
        self.assertEqual(convert_ip("127.0.0.1"),
                         "There is no place like localhost")

    def test_minimum_and_maximum(self):
        self.assertEqual(convert_ip("0.0.0.0"), "00000000." * 3 + "00000000")
        self.assertEqual(convert_ip("255.255.255.255"), "11111111." * 3 + "11111111")

    def test_octet_padding(self):
        self.assertEqual(convert_ip("1.2.3.4"),
                         "00000001.00000010.00000011.00000100")

    def test_invalid_inputs(self):
        for bad in ("256.2.3.4", "1.2.3", "1.2.3.4.5", "a.b.c.d", "1..3.4",
                    "-1.2.3.4", " 1.2.3.4", "1.2.3.4 "):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                convert_ip(bad)
        with self.assertRaises(TypeError):
            convert_ip(None)


class TestTask2(unittest.TestCase):
    def test_required_answer(self):
        self.assertEqual(select_bookings([1, 3, 0, 5, 8, 5], [2, 4, 6, 7, 9, 9]),
                         [0, 1, 3, 4])

    def test_tie_break_and_no_candidate(self):
        starts = [0, 2, 2, 4]
        finishes = [2, 4, 4, 6]
        self.assertEqual(select_bookings(starts, finishes), [0, 1, 3])
        self.assertEqual(earliest_finish(starts, finishes, 4), 3)
        self.assertEqual(earliest_finish(starts, finishes, 6), -1)

    def test_empty_and_validation(self):
        self.assertEqual(select_bookings([], []), [])
        for starts, finishes in [([1], []), ([2], [2]), ([3], [1]), ([-1], [1]),
                                 ([1], [float("nan")])]:
            with self.subTest(starts=starts, finishes=finishes), self.assertRaises(ValueError):
                select_bookings(starts, finishes)
        with self.assertRaises(TypeError):
            select_bookings([True], [2])

    def test_evaluator(self):
        self.assertEqual(evaluate_schedule([1, 3, 0, 5, 8, 5],
                                           [2, 4, 6, 7, 9, 9],
                                           [0, 1, 3, 4]),
                         (True, 4, [1, 1, 1, 1], 9))
        self.assertFalse(evaluate_schedule([0, 1], [2, 3], [0, 1])[0])
        self.assertFalse(evaluate_schedule([0], [2], [0, 0])[0])
        with self.assertRaises(ValueError):
            evaluate_schedule([0], [2], [99])

    def test_greedy_matches_brute_force_cardinality(self):
        """Check greedy result against ALL subsets in 220 seeded small cases."""
        rng = random.Random(20261024)
        for trial in range(220):
            n = rng.randint(0, 8)
            starts = [rng.randint(0, 11) for _ in range(n)]
            finishes = [start + rng.randint(1, 5) for start in starts]
            chosen = select_bookings(starts, finishes)
            is_valid, count, _, _ = evaluate_schedule(starts, finishes, chosen)
            self.assertTrue(is_valid, f"invalid greedy schedule on trial {trial}")
            self.assertEqual(count, len(set(chosen)))
            best = 0
            for count_subset in range(n + 1):
                for subset in itertools.combinations(range(n), count_subset):
                    ordered = sorted(subset, key=lambda i: (finishes[i], i))
                    if evaluate_schedule(starts, finishes, ordered)[0]:
                        best = max(best, count_subset)
            self.assertEqual(len(chosen), best, f"suboptimal on trial {trial}")


class TestTask3(unittest.TestCase):
    def test_all_die_outcomes_at_every_inventory(self):
        for bikes in range(11):
            for roll in range(1, 7):
                with self.subTest(bikes=bikes, roll=roll):
                    later, outcome = process_event(bikes, 10, roll)
                    self.assertTrue(0 <= later <= 10)
                    if roll in (1, 2):
                        self.assertEqual(later, max(0, bikes - 1))
                        self.assertEqual(outcome, "rejected_hires" if bikes == 0
                                         else "successful_hires")
                    elif roll == 3:
                        self.assertEqual(later, min(10, bikes + 1))
                        self.assertEqual(outcome, "rejected_returns" if bikes == 10
                                         else "successful_returns")
                    else:
                        self.assertEqual(later, bikes)
                        self.assertEqual(outcome, "no_change")

    def test_empty_repeated_hires(self):
        result = simulate_station(initial_bikes=0, rolls=[1] * 120)
        self.assertEqual(result["final_bikes"], 0)
        self.assertEqual(result["counters"]["rejected_hires"], 120)

    def test_full_repeated_returns(self):
        result = simulate_station(initial_bikes=10, rolls=[3] * 120)
        self.assertEqual(result["final_bikes"], 10)
        self.assertEqual(result["counters"]["rejected_returns"], 120)

    def test_stock_exhaustion_and_filling(self):
        emptying = simulate_station(initial_bikes=6, rolls=[1] * 120)
        filling = simulate_station(initial_bikes=6, rolls=[3] * 120)
        self.assertEqual(emptying["counters"]["successful_hires"], 6)
        self.assertEqual(emptying["counters"]["rejected_hires"], 114)
        self.assertEqual(filling["counters"]["successful_returns"], 4)
        self.assertEqual(filling["counters"]["rejected_returns"], 116)

    def test_no_change_and_reproducibility(self):
        still = simulate_station(rolls=[4] * 120)
        self.assertEqual(still["final_bikes"], 6)
        self.assertEqual(still["counters"]["no_change"], 120)
        self.assertEqual(simulate_station(seed=42), simulate_station(seed=42))
        self.assertEqual(len(simulate_station()["history"]), 120)

    def test_randomised_stock_invariants(self):
        for seed in range(100):
            result = simulate_station(seed=seed)
            self.assertTrue(all(0 <= item["bikes_after"] <= 10
                                for item in result["history"]))
            self.assertEqual(sum(result["counters"].values()), 120)

    def test_bad_simulation_inputs(self):
        with self.assertRaises(ValueError):
            simulate_station(initial_bikes=11)
        with self.assertRaises(ValueError):
            simulate_station(rolls=[4] * 119)
        with self.assertRaises(ValueError):
            simulate_station(rolls=[7] * 120)
        with self.assertRaises(ValueError):
            simulate_station(capacity=0)

    def test_transfer_conservation_and_balance(self):
        self.assertEqual(balance_transfer(10, 0, 10), (8, 2, 2, "A_to_B"))
        self.assertEqual(balance_transfer(0, 10, 10), (2, 8, 2, "B_to_A"))
        self.assertEqual(balance_transfer(5, 4, 10), (5, 4, 0, "none"))
        trial = simulate_two_stations(initial_a=10, initial_b=0, minutes=120,
                                      rolls_a=[4] * 120, rolls_b=[4] * 120)
        self.assertEqual((trial["final_a"], trial["final_b"]), (5, 5))
        self.assertEqual([t["moved"] for t in trial["transfers"]], [2, 2, 1, 0])
        self.assertEqual(sum(t["moved"] for t in trial["transfers"]), 5)

    def test_randomised_two_station_invariants(self):
        for seed in range(40):
            x = simulate_two_stations(seed=seed, initial_a=9, initial_b=1)
            self.assertEqual(len(x["history"]), 120)
            self.assertEqual(len(x["transfers"]), 4)
            self.assertTrue(all(0 <= item["bikes_a_after"] <= 10 and
                                0 <= item["bikes_b_after"] <= 10 for item in x["history"]))
            expected = (10 + x["counters_a"]["successful_returns"]
                        + x["counters_b"]["successful_returns"]
                        - x["counters_a"]["successful_hires"]
                        - x["counters_b"]["successful_hires"])
            self.assertEqual(x["final_a"] + x["final_b"], expected)


class TestTask4(unittest.TestCase):
    def test_dataset_layout(self):
        texts, labels, counts = load_data(ROOT / "7042SCN_Text_Classification_Dataset.txt")
        self.assertEqual(len(texts), 75)
        self.assertEqual(len(labels), 75)
        self.assertEqual(set(labels), set(CATEGORIES))
        self.assertTrue(all(count == 15 for count in counts.values()))

    def test_identical_folds_and_fully_assigned_predictions(self):
        texts, labels, _ = load_data(ROOT / "7042SCN_Text_Classification_Dataset.txt")
        predictions, fold_ids, fold_results = evaluate(texts, labels)
        self.assertEqual(sorted(fold_ids).count(1), 15)
        self.assertEqual(len(fold_results), 5)
        for library in ("scikit-learn", "NLTK"):
            self.assertEqual(len(predictions[library]), 75)
            self.assertTrue(all(value in CATEGORIES for value in predictions[library]))
        # Regression check against the verified run (not a claim of generalisation).
        self.assertEqual(sum(a == p for a, p in zip(labels, predictions["scikit-learn"])), 44)
        self.assertEqual(sum(a == p for a, p in zip(labels, predictions["NLTK"])), 36)


if __name__ == "__main__":
    unittest.main(verbosity=2)
