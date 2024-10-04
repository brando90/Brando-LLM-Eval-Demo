# Brando LLM Eval Demo


## Setup

(Optional) Update conda:

```bash
conda update -n base -c defaults conda -y
```

Create a conda environment with the required packages:

```bash
conda create -n eleuther_lm_eval_harness_20240927 python=3.11
```

To activate the environment:

```bash
conda activate eleuther_lm_eval_harness_20240927
```

If running on SNAP, make sure you have sufficient disk space in `/afs/cs.stanford.edu/u/<your username>/`:

```bash
rm -rf /afs/cs.stanford.edu/u/<your username>/.cache
```

**Warning: if your path points to AFS e.g., if you use Brando's ASF set up, you really need to make sure you point to a different location other than afs**

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
# - Open kerberos tmux in ampere cluster & reauth
krbtmux
reauth

# - Activate right env and run experiment
conda activate eleuther_lm_eval_harness_20240927
cd ~/beyond-scale-language-data-diversity/Brando-LLM-Eval-Demo
export PYTHONPATH=.
# change gpu if needed
export CUDA_VISIBLE_DEVICES=6
# Run experiment
python -u queue_evals.py ${CUDA_VISIBLE_DEVICES} 
```

I recommend running multiple `krbtmux` sessions in parallel.

The per-sample outputs will be written to disk in a directory called `eval_results` (see `queue_evals.py` for location).
