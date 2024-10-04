import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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
    # For GPT2 models, extract the base model family (e.g., 'GPT2_51M', 'GPT2_117M')
    elif model_name.startswith('GPT2'):
        tokens = model_name.split('_')
        if len(tokens) >= 2:
            return '_'.join(tokens[:2])
    return 'Other'

model_names = list(model_name_folder_2_div_coeff.keys())
print(f'{model_names=}')

# Data structures to hold collected data
data_points = []  # List to hold all data points
model_family_data = defaultdict(list)  # Dictionary to hold data per model family

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
                        data_point = {
                            'model_family': model_family,
                            'diversity_coefficient': div_coeff,
                            'log_likelihood': log_likelihood
                        }
                        data_points.append(data_point)
                        model_family_data[model_family].append(data_point)

# Plotting setup
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']

# For overall regression
all_div_coeffs = []
all_log_likelihoods = []

# Iterate over each model family to plot data and compute regression
for idx, (model_family, data_list) in enumerate(model_family_data.items()):
    # Extract data for the model family
    div_coeffs = np.array([dp['diversity_coefficient'] for dp in data_list]).reshape(-1, 1)
    log_likelihoods = np.array([dp['log_likelihood'] for dp in data_list])
    
    # Append to overall data
    all_div_coeffs.extend(div_coeffs.flatten())
    all_log_likelihoods.extend(log_likelihoods)
    
    # Scatter plot for the model family
    plt.scatter(div_coeffs, log_likelihoods, color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)], alpha=0.6, label=model_family)
    
    # Linear regression for the model family
    reg = LinearRegression().fit(div_coeffs, log_likelihoods)
    predicted = reg.predict(div_coeffs)
    r2 = r2_score(log_likelihoods, predicted)
    
    # Plot regression line
    x_vals = np.unique(div_coeffs)
    y_vals = reg.predict(x_vals.reshape(-1, 1))
    plt.plot(x_vals, y_vals, color=colors[idx % len(colors)], linestyle='--')
    
    # Annotate R² value
    plt.text(x_vals.mean(), y_vals.mean(), f'{model_family} R²={r2:.2f}', color=colors[idx % len(colors)])

# Overall regression
all_div_coeffs_np = np.array(all_div_coeffs).reshape(-1, 1)
all_log_likelihoods_np = np.array(all_log_likelihoods)
overall_reg = LinearRegression().fit(all_div_coeffs_np, all_log_likelihoods_np)
overall_predicted = overall_reg.predict(all_div_coeffs_np)
overall_r2 = r2_score(all_log_likelihoods_np, overall_predicted)

# Plot overall regression line
x_vals_overall = np.linspace(min(all_div_coeffs), max(all_div_coeffs), 100).reshape(-1, 1)
y_vals_overall = overall_reg.predict(x_vals_overall)
plt.plot(x_vals_overall, y_vals_overall, color='black', linestyle='-', label='Overall Fit')

# Annotate overall R²
plt.text(x_vals_overall.mean(), y_vals_overall.mean(), f'Overall R²={overall_r2:.2f}', color='black')

# Plot settings
plt.xlabel('Diversity Coefficient')
plt.ylabel('Log Likelihood of Correct Answer')
plt.title('Diversity Coefficient vs. Log Likelihood of Correct Answers by Model Family')
plt.xticks([0.158, 0.168, 0.195], ['USPTO', 'PubMed', 'USPTOAndPubMed'])
plt.grid(True)
plt.legend()

# Save the plot in three formats
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'diversity_vs_log_likelihood_with_regression.png'), format='png', dpi=300)
plt.savefig(os.path.join(output_dir, 'diversity_vs_log_likelihood_with_regression.pdf'), format='pdf')
plt.savefig(os.path.join(output_dir, 'diversity_vs_log_likelihood_with_regression.svg'), format='svg')

plt.show()
