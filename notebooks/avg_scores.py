import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Path to evaluation results
eval_results_path = '/lfs/skampere1/0/brando9/data/beyond_scale/eval_results_back_up'

# Mapping of model names to diversity coefficients
model_name_folder_2_div_coeff = {
    "LLama2_Uspto_Ckpt_1": 0.158,
    "LLama2_Pubmed_Ckpt_2": 0.168,
    "LLama2_Uspto_Pubmed_Ckpt_3": 0.195,
    "LLama2_Uspto_Pubmed_Ckpt_4": 0.195,
    "LLama2_Uspto_Pubmed_Ckpt_5": 0.195,
    "LLama2_Uspto_Pubmed_Ckpt_6": 0.195,
    "LLama2_Uspto_Pubmed_Ckpt_7": 0.168,

    "GPT2_51M_1.31B_USPTO": 0.158,
    "GPT2_51M_1.31B_PubMedAbs": 0.168,
    "GPT2_51M_1.31B_USPTOAndPubMedAbs": 0.195,

    "GPT2_51M_557M_USPTO": 0.158,
    "GPT2_51M_557M_PubMedAbs": 0.168,
    "GPT2_51M_557M_USPTOAndPubMedAbs": 0.195,

    "GPT2_117M_2.2B_USPTO": 0.158,
    "GPT2_117M_2.2B_PubMedAbs": 0.168,
    "GPT2_117M_2.2B_USPTOAndPubMedAbs": 0.195,

    "GPT2_204M_USPTO": 0.158,
    "GPT2_204M_PubMedAbs": 0.168,
    "GPT2_204M_USPTOAndPubMedAbs": 0.195,

    "GPT2_345M_2.2B_USPTO": 0.158,
    "GPT2_345M_2.2B_PubMedAbs": 0.168,
    "GPT2_345M_2.2B_USPTOAndPubMedAbs": 0.195,

    "GPT2_810M_PubMedAbs": 0.168,
    "GPT2_810M_2.2B_USPTOAndPubMedAbs": 0.195,

    "GPT2_1.5B_180M_USPTO": 0.158,
    "GPT2_1.5B_180M_PubMedAbs": 0.168,
    "GPT2_1.5B_180M_USPTOAndPubMedAbs": 0.195
}

# Function to extract the model name from the directory path
def get_model_name_in_root(root, model_names) -> str:
    for model_name in model_names:
        if model_name in root:
            return model_name
    return ''

# Function to determine the model family from the model name
def get_model_family(model_name):
    # For LLama2 models
    if model_name.startswith('LLama2'):
        return 'LLama2'
    # For GPT2 models, extract the base model family (e.g., 'GPT2_51M')
    elif model_name.startswith('GPT2'):
        tokens = model_name.split('_')
        if len(tokens) >= 2:
            return '_'.join(tokens[:2])  # e.g., 'GPT2_51M'
    return 'Other'

model_names = list(model_name_folder_2_div_coeff.keys())
print(f'{model_names=}')

# Data structures to hold collected data
# For each model family, for each diversity coefficient, store the list of log likelihoods
model_family_div_coeff_data = defaultdict(lambda: defaultdict(list))

# Traverse the directory and collect data
for root, dirs, filenames in os.walk(eval_results_path):
    model_name = get_model_name_in_root(root, model_names)
    if model_name and filenames:
        div_coeff = model_name_folder_2_div_coeff[model_name]
        model_family = get_model_family(model_name)
        for filename in filenames:
            if filename.endswith('.jsonl'):
                path_to_results = os.path.join(root, filename)
                with open(path_to_results, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        correct_index = data['doc']['answer']
                        log_likelihood = float(data['filtered_resps'][correct_index][0])
                        # Add data point
                        model_family_div_coeff_data[model_family][div_coeff].append(log_likelihood)

# Now, for each model family, compute the mean log likelihood at each diversity coefficient
# Prepare data for plotting
plt.figure(figsize=(12, 8))

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']

for idx, (model_family, div_coeff_data) in enumerate(model_family_div_coeff_data.items()):
    div_coeffs = []
    mean_log_likelihoods = []
    std_log_likelihoods = []
    for div_coeff in sorted(div_coeff_data.keys()):
        log_likelihoods = div_coeff_data[div_coeff]
        mean_ll = np.mean(log_likelihoods)
        std_ll = np.std(log_likelihoods)
        div_coeffs.append(div_coeff)
        mean_log_likelihoods.append(mean_ll)
        std_log_likelihoods.append(std_ll)
    # Sort the data points for plotting
    div_coeffs = np.array(div_coeffs)
    mean_log_likelihoods = np.array(mean_log_likelihoods)
    std_log_likelihoods = np.array(std_log_likelihoods)
    # Plot the mean log likelihoods vs diversity coefficients
    plt.errorbar(div_coeffs, mean_log_likelihoods, yerr=std_log_likelihoods,
                 color=colors[idx % len(colors)],
                 marker=markers[idx % len(markers)], linestyle='-',
                 label=model_family, capsize=5)

plt.xlabel('Diversity Coefficient')
plt.ylabel('Mean Log Likelihood of Correct Answer')
plt.title('Mean Log Likelihood vs. Diversity Coefficient per Model Family')
plt.xticks([0.158, 0.168, 0.195], ['USPTO', 'PubMed', 'USPTOAndPubMed'])
plt.grid(True)
plt.legend()

# Save the plot in three formats
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'mean_log_likelihood_vs_diversity_coeff.png'), format='png', dpi=300)
plt.savefig(os.path.join(output_dir, 'mean_log_likelihood_vs_diversity_coeff.pdf'), format='pdf')
plt.savefig(os.path.join(output_dir, 'mean_log_likelihood_vs_diversity_coeff.svg'), format='svg')

plt.show()
