# machine-learning

A simple perceptron model that predicts student placement using DSA question count and CGPA. The training data is in `placements.csv`.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
cd Perceptron
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas
```

## Run

```bash
python perceptron.py
```

The script trains the model on `placements.csv`, reports its training accuracy, and prompts for a student's DSA question count and CGPA.

The CSV columns are `dsa_questions`, `cgpa`, and `placed` (0 or 1).
