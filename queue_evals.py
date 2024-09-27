import numpy as np
import os
import pandas as pd
import random
import subprocess
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
gpu_ids = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)
print("CUDA_VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])


models_df = pd.read_csv("configs/models.csv", index_col=None)
# models_df = pd.read_csv("configs/models_nonscaling.csv", index_col=None)
benchmarks_df = pd.read_csv("configs/benchmarks_nlp.csv", index_col=None)
# benchmarks_df = pd.read_csv("configs/benchmarks_code.csv", index_col=None)

# Shuffle models and benchmarks.
models_df = models_df.sample(frac=1)
benchmarks_df = benchmarks_df.sample(frac=1)

results_dir = "eval_results"

for model_idx, model_row in models_df.iterrows():
    # Compute and (optionally) create the model output directory.
    model_nickname = model_row["Model Nickname"]
    huggingface_revision = model_row["HuggingFace Revision"]
    huggingface_path = model_row["HuggingFace Path"]
    for benchmarks_idx, benchmark_row in benchmarks_df.iterrows():
        if benchmark_row["Library"] == "LM Evaluation Harness":
            model_args = f"pretrained={huggingface_path}"
            if not pd.isna(huggingface_revision):
                model_args += f",revision={huggingface_revision}"
            # model_args += ",trust_remote_code=True"  # ,parallelize=True
            model_args += ",parallelize=True"

            benchmark_and_optional_task = benchmark_row["Benchmark"]
            task = benchmark_row["Task"]
            if not pd.isna(task):
                benchmark_and_optional_task += f"_{task}"

            # Compute and (optionally) create the task output directory.
            model_task_output_path = os.path.join(
                results_dir, model_nickname, benchmark_and_optional_task
            )
            os.makedirs(model_task_output_path, exist_ok=True)

            # Skip if results.json exists
            results_json_path = os.path.join(model_task_output_path, "results.json")
            if os.path.exists(results_json_path):
                print(
                    f"Skipping {model_nickname} for {benchmark_and_optional_task} because results.json exists."
                )
                continue

            command = f"""
                lm_eval --model hf \
                    --model_args {model_args} \
                    --tasks {benchmark_and_optional_task} \
                    --trust_remote_code \
                    --batch_size auto:4 \
                    --device cuda \
                    --output_path {model_task_output_path} \
                    --log_samples
                """

        else:
            print(f"Unknown evaluation library: {benchmark_row['Library']}")
            continue

        print("Command: ", command)
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            continue