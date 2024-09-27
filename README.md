# Brando LLM Eval Demo


## Setup

(Optional) Update conda:

`conda update -n base -c defaults conda -y`

Create a conda environment with the required packages:

`conda create -n eleuther_lm_eval_harness_20240927 python=3.11`

To activate the environment:

`conda activate eleuther_lm_eval_harness_20240927`

If running on SNAP, make sure you have sufficient disk space in `/afs/cs.stanford.edu/u/<your username>/`:

`rm -rf /afs/cs.stanford.edu/u/<your username>/.cache`

Clone and install the evaluation harness:

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness && pip install -e . && cd ..
```

## Running

There are 2 config files: 

1. `configs/benchmarks.csv`: This CSV contains all the benchmark tasks.
2. `configs/models.csv`: This CSV contains all the models.

To queue running evals, create a new `krbtmux` session, reauthenticate, and run the following command
from the project directory:

```bash
export PYTHONPATH=.
python -u queue_evals.py 2, # Change this to whatever GPU you want to use.
```

I recommend running multiple sessions in parallel.

The per-sample outputs will be written to disk in a directory called `eval_results`.