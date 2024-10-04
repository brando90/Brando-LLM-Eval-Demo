import os
import json

eval_results_path = '/lfs/skampere1/0/brando9/data/beyond_scale/eval_results_back_up'

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

    # "GPT2_810M_PubMedAbs": 0.168,
    # "GPT2_810M_2.2B_USPTOAndPubMedAbs": 0.195,

    "GPT2_1.5B_180M_USPTO": 0.158,
    "GPT2_1.5B_180M_PubMedAbs": 0.168,
    "GPT2_1.5B_180M_USPTOAndPubMedAbs": 0.195
}

model_family_2_div_coeff = {
    "LLama2": 0.158,

    "GPT2_51M_1.31B": 0.158,
    
    "GPT2_51M_557M": 0.158,

    "GPT2_117M_2.2B": 0.158,

    "GPT2_204M": 0.158,

    "GPT2_345M_2.2B": 0.158,

    # "GPT2_810M_2.2B": 0.195,

    "GPT2_1.5B_180M": 0.158,
}

model_family_2_div_coeff_2_likelihoods = {
    "LLama2": {0.158: [], 0.168: [], 0.195: []},

    "GPT2_51M_1.31B": {0.158: [], 0.168: [], 0.195: []},
    
    "GPT2_51M_557M": {0.158: [], 0.168: [], 0.195: []},

    "GPT2_117M_2.2B": {0.158: [], 0.168: [], 0.195: []},

    "GPT2_204M": {0.158: [], 0.168: [], 0.195: []},

    "GPT2_345M_2.2B": {0.158: [], 0.168: [], 0.195: []},

    # "GPT2_810M_2.2B": {0.158: [], 0.168: [], 0.195: []},

    "GPT2_1.5B_180M": {0.158: [], 0.168: [], 0.195: []},
}

def get_model_name_in_root(root, model_names) -> str:
    for model_name in model_names:
        if model_name in root:
            return model_name
    return ''

model_names = list(model_name_folder_2_div_coeff.keys())
model_family_names = list(model_family_2_div_coeff)
print(f'{model_names=}')

div_coeff_2_dataset = {0.158: 'USPTO', 0.168: 'PubMed', 0.195: 'USPTOAndPubMed'}
div_coeff_2_log_p_v_correct = {0.158: [], 0.168: [], 0.195: []}
for root, dirs, filenames in os.walk(eval_results_path):
    model_name: str = get_model_name_in_root(root, model_names)
    if model_name and len(filenames) > 0: 
        model_family_name: str = get_model_name_in_root(root, model_family_names)
        for filename in filenames:
            div_coeff: float = model_name_folder_2_div_coeff[model_name]
            path_2_results = os.path.expanduser(f'{root}/{filename}')
            if path_2_results.endswith('.jsonl'):
                with open(path_2_results, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        correct_index = data['doc']['answer']
                        log_likelihood = float(data['filtered_resps'][correct_index][0])
                        div_coeff_2_log_p_v_correct[div_coeff].append(log_likelihood)
                        model_family_2_div_coeff_2_likelihoods[model_family_name][div_coeff].append(log_likelihood)

# plot div coeff vs likelihoods
import matplotlib.pyplot as plt

# Prepare data for plotting
diversity_coefficients = []
log_likelihoods = []
for div_coeff, log_p_list in div_coeff_2_log_p_v_correct.items():
    diversity_coefficients.extend([div_coeff] * len(log_p_list))
    log_likelihoods.extend(log_p_list)

# Plot diversity coefficient vs. log likelihoods
plt.figure(figsize=(10, 6))
plt.scatter(diversity_coefficients, log_likelihoods, alpha=0.6)
plt.xlabel('Diversity Coefficient')
plt.ylabel('Log Likelihood of Correct Answer')
plt.title('Diversity Coefficient vs. Log Likelihood of Correct Answers')
plt.xticks([0.158, 0.168, 0.195], ['USPTO', 'PubMed', 'USPTOAndPubMed'])
plt.grid(True)

# Save the plot in three formats
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'diversity_vs_log_likelihood.png'), format='png', dpi=300)
plt.savefig(os.path.join(output_dir, 'diversity_vs_log_likelihood.pdf'), format='pdf')
plt.savefig(os.path.join(output_dir, 'diversity_vs_log_likelihood.svg'), format='svg')

plt.show()