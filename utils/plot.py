from matplotlib import pyplot as plt
import numpy as np
import os
from settings import RESULTS_PATH
import seaborn as sns

# def plot_density_estimate(rkde, data_point_index, true_density=None):
# 	plt.figure(figsize=(10, 6))

# 	evaluation_points = np.linspace(rkde.grid_min, rkde.grid_max, 1000)
# 	evaluation_density = rkde.evaluate(evaluation_points)

# 	if true_density is not None:
# 		plt.plot(evaluation_points, true_density(evaluation_points), label='True Density', color='red', linestyle='--')


# 	plt.plot(evaluation_points, evaluation_density, label='RKDE Estimate', color='blue')
# 	plt.title(f'Recursive KDE after {data_point_index + 1} Data Points')
# 	plt.xlabel('x')
# 	plt.ylabel('Density')
# 	plt.ylim(0, np.max(evaluation_density) * 1.1)
# 	plt.legend()
# 	plt.grid()
# 	plt.show()

sns.set_style("whitegrid")

custom_serif_fonts = ['Times New Roman', 'DejaVu Serif', 'serif']

# 2. Apply this sequence to the font configuration
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = custom_serif_fonts

# 3. Apply to specific elements for consistent output
# (Optional, but recommended for titles and labels)
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# Set the default font family for all text elements.
# plt.rcParams['font.family'] = 'Times New Roman'

# Set the default font size for all text. You can use 'xx-small', 'x-small', 'small',
# 'medium', 'large', 'x-large', 'xx-large', or a numeric value.
plt.rcParams['font.size'] = 16

# Set the font size specifically for the plot title.
plt.rcParams['axes.titlesize'] = 22
# Set the font weight for the title.
plt.rcParams['axes.titleweight'] = 'bold'

# Set the font size for the x and y axis labels.
plt.rcParams['axes.labelsize'] = 12
# Set the font style for the labels.
# plt.rcParams['axes.labelstyle'] = 'italic'

# Set the font size for the legend text.
plt.rcParams['legend.fontsize'] = 14

def animate(i, ax, evaluation_points, evaluation_density_history):
    """Updates the plot for frame 'i'."""
    
    # --- Clear the previous frame and plot the new one ---
    ax.cla() # Clear the axis
    
    # 1. Plot the current density estimate
    line, = ax.plot(evaluation_points, evaluation_density_history[i], 'b-', label="KDE Estimate") 
    
    # 2. Plot the true density (optional, for reference)
    true_density_values = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * evaluation_points**2)
    ax.plot(evaluation_points, true_density_values, 'k--', label="True Density")
    
        
    # 3. Set plot limits and title
    ax.set_ylim(0, 0.5) 
    ax.set_title(f"Recursive KDE Update - Samples: {i}")
    ax.legend(loc='upper right')
    ax.set_xlabel("X")
    ax.set_ylabel("Density")


def plot_and_save_density_estimations(oscs, grid_min, grid_max, test_X, decisions, **kwargs):

    fig, ax = plt.subplots(figsize=(8, 5))
    evaluation_points = np.linspace(grid_min, grid_max, 1000)

    oscs.rkde_estimator.evaluate(evaluation_points)
    oscs.kde_estimator(evaluation_points)

    ax.hist(test_X, bins=50, alpha=0.7, label='Test Data Scores', density=True)
    ax.hist(test_X[np.array(decisions) == 1], bins=50, alpha=0.7, label='Selected Test Data Scores', density=True)
    ax.plot(evaluation_points, oscs.kde_estimator(evaluation_points), label='KDE on Calibration Data', color='red')
    ax.plot(evaluation_points, oscs.rkde_estimator.evaluate(evaluation_points), label='Recursive KDE on Test Data', color='green')
    ax.set_title('Density Estimation Comparison')
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.legend()

    fig.savefig(os.path.join(RESULTS_PATH, 'density_estimations.pdf'), bbox_inches='tight')


def plot_and_save_scores_histogram(validation_scores, test_clean_scores, test_poison_scores, score_name):
    fig, ax = plt.subplots()
    ax.hist(validation_scores, bins=10, alpha=0.5, label='Validation Clean', density=True)
    ax.hist(test_clean_scores, bins=10, alpha=0.5, label='Test Clean', density=True)
    ax.hist(test_poison_scores, bins=10, alpha=0.5, label='Test Poison', density=True)
    ax.legend()
    fig.savefig(os.path.join(RESULTS_PATH, f"{score_name}_histogram.pdf"), bbox_inches='tight')

def plot_and_save_overall_performance(avg_fsr_history, std_fsr_history, avg_power_history, std_power_history, alpha):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(avg_fsr_history, label='Average FSR over Trials')
    axes[0].fill_between(range(len(avg_fsr_history)), avg_fsr_history - std_fsr_history, avg_fsr_history + std_fsr_history, color='blue', alpha=0.2)
    axes[0].axhline(y=alpha, color='r', linestyle='--', label='Target FSR Level')
    axes[0].set_xlabel('Number of Test Samples Processed')
    axes[0].set_ylabel('FSR')
    axes[0].set_title('FSR over Time')
    axes[0].legend()

    axes[1].plot(avg_power_history, label='Average Power over Trials')
    axes[1].fill_between(range(len(avg_power_history)), avg_power_history - std_power_history, avg_power_history + std_power_history, color='orange', alpha=0.2)
    axes[1].set_xlabel('Number of Test Samples Processed')
    axes[1].set_ylabel('Power')
    axes[1].set_title('Power over Time')
    axes[1].legend()

    fig.tight_layout()

    fig.savefig(os.path.join(RESULTS_PATH, 'performance_over_time.pdf'), bbox_inches='tight')
    plt.close(fig)

def plot_and_save_training_loss_curve(loss_list, loss_name):

    fig, ax = plt.subplots()

    plot_name = ""
    for i, name in enumerate(loss_name):
        if i == len(loss_name) - 1:
            plot_name += name
        else:
            plot_name += name + "_"
    
    for loss, name in zip(loss_list, loss_name):
        ax.plot(loss, label=name)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.savefig(os.path.join(RESULTS_PATH, f'training_loss_curve_{plot_name}.pdf'), bbox_inches='tight')
    plt.close(fig)