# Programming for Data Science (7042SCN)

**Saride Pavan Rahul · 17509497**  
Coventry University — Individual Coursework

Python implementations and reproducible outputs for the four coursework tasks.

## Contents

- `task1_ipv4.py` — dotted-decimal IPv4 conversion and assertions.
- `task2_bookings.py` — earliest-finish booking selection, execution traces, schedule evaluation.
- `task3_bikeshare.py` — seeded station simulations, fixed-event tests, two-station transfers.
- `task4_text_classification.py` — independent scikit-learn and NLTK methods with held-out evaluation.
- `task4_robustness.py` — sensitivity to different cross-validation splits.
- `test_all_tasks.py` — automated checks across the four tasks.
- `7042SCN_Text_Classification_Dataset.txt` — original supplied Task 4 data (75 enquiries).
- `outputs/` — recorded console runs, metrics, predictions and simulation histories.
- `figures/` — Task 4 confusion matrices.
- `notes/` — Task 3 design and technical interpretation used to prepare the written report.

## Run

Requires Python 3.10+.

```bash
python3 -m pip install -r requirements.txt
python3 run_all.py
```

Run the tests on their own:

```bash
python3 -m unittest -v test_all_tasks
```

`run_all.py` saves the latest results in `outputs/`; Task 4 regenerates the figures. All evaluation procedures train exclusively on the training partitions of the dataset.


