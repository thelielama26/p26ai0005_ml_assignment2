import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Set aesthetic styling for CVPR publication quality
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 13,
    'figure.autolayout': True
})

FIGURES_DIR = "figures"
LOGS_DIR = "logs"

def ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)

def load_json_log(filename):
    filepath = os.path.join(LOGS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Skipping related plots.")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

# =============================================================================
# Q1 PLOTS
# =============================================================================
def plot_q1_bagging_metrics(q1_data):
    """Plot each Q1(c) bagging metric separately against ensemble size B."""
    bag_results = q1_data.get("bagging_sweep", [])
    if not bag_results:
        return

    B_vals = [item["B"] for item in bag_results]
    oob_acc = [item["oob_acc"] for item in bag_results]
    disagreement = [item["disagreement"] for item in bag_results]
    margin = [item["vote_margin"] for item in bag_results]

    metrics = [
        (
            oob_acc,
            'OOB Accuracy',
            '#2ca02c',
            'q1c_oob_accuracy.png',
            'Q1(c): OOB Accuracy vs. Ensemble Size ($B$)',
        ),
        (
            margin,
            'Mean Vote Margin',
            '#ff7f0e',
            'q1c_vote_margin.png',
            'Q1(c): Mean Vote Margin vs. Ensemble Size ($B$)',
        ),
        (
            disagreement,
            'Pairwise Disagreement',
            '#d62728',
            'q1c_pairwise_disagreement.png',
            'Q1(c): Pairwise Disagreement vs. Ensemble Size ($B$)',
        ),
    ]

    for values, ylabel, color, filename, title in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(B_vals, values, color=color, marker='o', linewidth=2)
        ax.set_xlabel('Ensemble Size ($B$)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale('log')
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
        plt.close()
        print(f"Saved {filename}.")

def plot_q1_tree_leaves_analysis(q1_data):
    """Plot tree complexity and leaf occupancy against maximum tree depth."""
    baseline_results = q1_data.get("baseline_sweep", [])
    if not baseline_results:
        return

    # Filter for min_samples_leaf = 1 to isolate depth effect
    filtered = [r for r in baseline_results if r["min_samples_leaf"] == 1]
    depths = [str(r["max_depth"]) if r["max_depth"] is not None else "None" for r in filtered]
    num_leaves = [r["num_leaves"] for r in filtered]
    frac_single = [r["frac_single_sample_leaves"] * 100 for r in filtered]

    x = np.arange(len(depths))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x, num_leaves, width, color='#4c72b0')
    ax.set_ylabel('Number of Leaves')
    ax.set_xticks(x)
    ax.set_xticklabels(depths)
    ax.set_xlabel('Max Depth')
    ax.set_title('Q1(a): Tree Complexity vs. Max Depth')
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'q1a_tree_complexity.png'), dpi=300)
    plt.close()
    print("Saved Q1a Tree Complexity plot.")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x, frac_single, width, color='#c44e52')
    ax.set_ylabel('Single-Sample Leaves (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(depths)
    ax.set_xlabel('Max Depth')
    ax.set_title('Q1(a): Leaf Occupancy vs. Max Depth')
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'q1a_leaf_occupancy.png'), dpi=300)
    plt.close()
    print("Saved Q1a Leaf Occupancy plot.")


# =============================================================================
# Q2 PLOTS
# =============================================================================
def plot_q2_boosting_weight_dynamics(q2_data):
    """Plot each boosting weight-dynamics metric separately over rounds."""
    history = q2_data.get("selected_model_history", [])
    if not history:
        return

    rounds = [step["round"] for step in history]
    errors = [step["weighted_error"] for step in history]
    alphas = [step["alpha"] for step in history]
    max_weights = [step["max_sample_weight"] for step in history]
    top10_weights = [step["top10_weight_sum"] for step in history]

    metrics = [
        (errors, 'Weighted Training Error ($\\epsilon_t$)', 'Error', '#d62728',
         'q2b_weighted_error.png'),
        (alphas, 'Learner Weight ($\\alpha_t$)', 'Alpha', '#1f77b4',
         'q2b_learner_weight.png'),
        (max_weights, 'Maximum Sample Weight ($w_{max}$)', 'Weight', '#ff7f0e',
         'q2b_max_sample_weight.png'),
        (top10_weights, 'Top-10 Heaviest Samples Weight Sum', 'Cumulative Weight', '#2ca02c',
         'q2b_top10_weight_sum.png'),
    ]

    for values, title, ylabel, color, filename in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rounds, values, color=color, linewidth=2)
        ax.set_title(f'Q2(b): {title}')
        ax.set_xlabel('Round ($t$)')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=':')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
        plt.close()
        print(f"Saved {filename}.")

def plot_q2_bagging_vs_boosting_time_acc(q2_data):
    """Plot accuracy and training time separately for bagging and boosting."""
    comp = q2_data.get("bagging_vs_boosting_comparison", [])
    if not comp:
        return

    learners = [item["learners"] for item in comp]
    bag_acc = [item["bagging"]["test_acc"] for item in comp]
    boost_acc = [item["boosting"]["test_acc"] for item in comp]
    bag_time = [item["bagging"]["train_time_sec"] for item in comp]
    boost_time = [item["boosting"]["train_time_sec"] for item in comp]

    metrics = [
        (bag_acc, boost_acc, 'Clean Test Accuracy', 'Accuracy',
         'q2c_test_accuracy.png'),
        (bag_time, boost_time, 'Training Time', 'Training Time (seconds)',
         'q2c_training_time.png'),
    ]

    for bag_values, boost_values, title, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(learners, bag_values, marker='o', label='Bagging',
                color='#1f77b4', linewidth=2)
        ax.plot(learners, boost_values, marker='s', label='Boosting',
                color='#d62728', linewidth=2)
        ax.set_xlabel('Number of Weak Learners')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Q2(c): {title} vs. Number of Weak Learners')
        ax.legend()
        ax.grid(True, linestyle=':')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
        plt.close()
        print(f"Saved {filename}.")


# =============================================================================
# Q3 PLOTS
# =============================================================================
def plot_q3_depth_bias_variance(q3_data):
    """Plot each bias-variance component separately against tree depth."""
    depth_res = q3_data.get("part_b_depth_analysis", [])
    if not depth_res:
        return

    depths = [str(r["depth"]) for r in depth_res]
    
    tree_b2 = [r["single_tree"]["bias_squared"] for r in depth_res]
    tree_var = [r["single_tree"]["variance"] for r in depth_res]
    
    bag_b2 = [r["bagging_100"]["bias_squared"] for r in depth_res]
    bag_var = [r["bagging_100"]["variance"] for r in depth_res]

    x = np.arange(len(depths))
    metrics = [
        (tree_b2, 'Single Tree: $\\mathrm{Bias}^2$', '#6baed6',
         'q3b_single_tree_bias_squared.png'),
        (tree_var, 'Single Tree: Variance', '#08519c',
         'q3b_single_tree_variance.png'),
        (bag_b2, 'Bagging (100): $\\mathrm{Bias}^2$', '#fdae6b',
         'q3b_bagging_bias_squared.png'),
        (bag_var, 'Bagging (100): Variance', '#e6550d',
         'q3b_bagging_variance.png'),
    ]

    for values, label, color, filename in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, values, marker='o', color=color, linewidth=2)
        ax.set_xlabel('Tree Depth')
        ax.set_ylabel('Measured Magnitude')
        ax.set_title(f'Q3(b): {label} vs. Tree Depth')
        ax.set_xticks(x)
        ax.set_xticklabels(depths)
        ax.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
        plt.close()
        print(f"Saved {filename}.")

def plot_q3_model_comparison_bar(q3_data):
    """Plot each Q3(d) model-comparison metric separately."""
    summary = q3_data.get("part_d_summary_table", {})
    if not summary:
        return

    models = list(summary.keys())
    clean_acc = [summary[m]["clean_test_acc"] * 100 for m in models]
    masked_acc = [summary[m]["masked_test_acc"] * 100 for m in models]
    bias_sq = [summary[m]["bias_squared"] for m in models]
    variance = [summary[m]["variance"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    metrics = [
        (clean_acc, 'Clean Test Accuracy (%)', '#2ca02c',
         'q3d_clean_test_accuracy.png'),
        (masked_acc, 'Masked Test Accuracy (%)', '#d62728',
         'q3d_masked_test_accuracy.png'),
        (bias_sq, '$\\mathrm{Bias}^2$', '#1f77b4',
         'q3d_bias_squared.png'),
        (variance, 'Variance', '#ff7f0e',
         'q3d_variance.png'),
    ]

    for values, ylabel, color, filename in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x, values, width, color=color)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Q3(d): {ylabel} by Model')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300)
        plt.close()
        print(f"Saved {filename}.")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def generate_all_figures():
    ensure_dirs()
    print("Starting figure generation...")

    # Load data logs
    q1_data = load_json_log("q1_results.json")
    if q1_data:
        plot_q1_bagging_metrics(q1_data)
        plot_q1_tree_leaves_analysis(q1_data)

    q2_data = load_json_log("q2_results.json")
    if q2_data:
        plot_q2_boosting_weight_dynamics(q2_data)
        plot_q2_bagging_vs_boosting_time_acc(q2_data)

    q3_data = load_json_log("q3_results.json")
    if q3_data:
        plot_q3_depth_bias_variance(q3_data)
        plot_q3_model_comparison_bar(q3_data)

    print(f"\\nAll figures successfully generated and saved to '{FIGURES_DIR}/' in .png format")

if __name__ == "__main__":
    generate_all_figures()