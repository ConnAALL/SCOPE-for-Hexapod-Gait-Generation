# Import statements
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# Plot styling
plt.style.use(['science', 'no-latex'])

# Constants
METRIC_KEYS = {
    'avg': 'DistanceDifference_avg',
    'best': 'DistanceDifference_best'
}
TOP_K = 500
NUM_GENS = 5000
INPUT_DIR = './processed_data'
OUTPUT_DIR = './out'

def load_all_metrics(directory):
    """Load all metrics from the specified directory"""
    results = {}

    # Load the best and average metrics
    for key, metric in METRIC_KEYS.items():
        data_list, names, final_vals = [], [], []
        for fname in sorted(os.listdir(directory)):  # For each folder
            if not fname.endswith('.npy'):  # If it is not a numpy file, skip it
                continue
            name = os.path.splitext(fname)[0]  # Get the run name
            rec = np.load(os.path.join(directory, fname), allow_pickle=True).item()  # Access the record
            gens, vals = rec.get('Generation'), rec.get(metric)
            # If we have less than the required generations or no values, skip it
            if gens is None or vals is None or gens.max() < NUM_GENS - 10:
                continue
            # Filter generations after the generation limit
            mask = gens <= NUM_GENS
            series = vals[mask]
            data_list.append(series)
            names.append(name)
            final_vals.append(series[-1])

        # Find the minimum length of the data arrays and filter all to this level
        min_len = min(len(arr) for arr in data_list)
        gens_common = gens[:min_len]
        matrix = np.vstack([arr[:min_len] for arr in data_list])

        # If we want to limit the number of top runs
        if TOP_K:
            def extract_timestamp(name):
                try:
                    parts = name.split('_')
                    return datetime.strptime(parts[-2] + '_' + parts[-1], '%y%m%d_%H%M%S')
                except Exception:
                    return datetime.min  # fallback if format is unexpected

            sorted_data = sorted(zip(names, data_list), key=lambda x: extract_timestamp(x[0]), reverse=True)
            names, data_list = zip(*sorted_data[:TOP_K])
            matrix = np.vstack(data_list)

        results[key] = (gens_common, matrix, names)
    return results


def style_axes(ax=None):
    """Style the axes for the plots"""
    ax = ax or plt.gca()

    # Show only bottom and left spines (real X and Y axes)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    # Style the real axes
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Show ticks and labels only on bottom and left
    ax.tick_params(
        axis='x',
        which='both',
        bottom=True,
        top=False,
        labelbottom=True
    )
    ax.tick_params(
        axis='y',
        which='both',
        left=True,
        right=False,
        labelleft=True
    )

    ax.grid(False)



def plot_individual_runs(gens, data, names, title, ylabel, out_path):
    """Plot individual runs of all time best for each run type"""
    fig, ax = plt.subplots(figsize=(12, 6))
    style_axes(ax)
    for row, nm in zip(data, names):
        ax.plot(gens, row, label=nm, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel('Generation')
    ax.set_ylabel(ylabel)
    ax.set_ylim((0, 22))
    fig.savefig(out_path, format='svg')
    plt.close(fig)


def plot_combined(curves, ylabel, out_path):
    """Plot combined curves for different run types"""
    fig, ax = plt.subplots(figsize=(12, 6))  # Create the new figure
    style_axes(ax)  # Apply the style to the axes
    for label, gens, curve in curves:  # Iterate over each curve
        color = 'black' if 'Discrete' in label else 'red' if 'Genetic' in label else 'gray'  # Depending on the label, choose color
        if "Discrete Cosine Transform" in label:
            label = "Sparse Cosine Optimized Policy Evolution"  # Rename for clarity
        if "Steady State Genetic Algorithm" in label:
            label = "Baseline Steady-State Genetic Algorithm"
        ax.plot(gens, curve, label=label, color=color, linewidth=2)
    ax.set_title(f"Sparse Cosine Optimized Policy Evolution vs. Baseline Steady-State Genetic Algorithm")  # Set the title of the plot
    ax.set_xlabel('Generation')  # Set the x-axis label
    ax.set_ylabel("Mean Fitness")  # Set the y-axis label
    ax.set_ylim((0, 16))  # Set the y-axis limits       
    ax.legend(loc='upper left')  # Add a legend to the plot
    fig.savefig(out_path, format='svg')  # Save the figure to the specified path
    plt.close(fig)  # Close the figure to free up memory


def report_summary(all_data_by_type):
    """Report the final generation distance difference summary""" 
    lines = ["Final Generation DistanceDifference Summary:"]  # Output construction
    for label, results in all_data_by_type.items():  # Iterate over each run type
        for key in METRIC_KEYS:  # For each metric key
            gens, mat, _ = results.get(key, (None, None, None))  # Get the data
            final = mat[:, -1]  # Get the last generation values
            # Construct the output
            lines.append(f"  {label} | {key.upper()}: mean={final.mean():.3f}, std={final.std():.3f}, n={len(final)}")
    print('\n'.join(lines))


def main():
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_base = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(out_base, exist_ok=True)

    # Define run types and their paths
    run_types = [
        ('Discrete Cosine Transform', os.path.join(INPUT_DIR, 'DCT')),
        ('Steady State Genetic Algorithm', os.path.join(INPUT_DIR, 'SSGA'))]

    # Load all metrics for each run type
    all_data_by_type = {}
    for label, path in run_types:
        all_data_by_type[label] = load_all_metrics(path)

    # Plot individual runs of all time best for each run type (SSGA and DCT)
    for label, results in all_data_by_type.items():
        gens, mat_best, names = results['best']  # Get the best runs
        plot_individual_runs(gens=gens,
                             data=mat_best,
                             names=names,
                             title=f"{label} Individual Runs (Best)",
                             ylabel=METRIC_KEYS['best'],
                             out_path=os.path.join(out_base, f"{label.replace(' ', '_')}_individual_best.svg"))

    # Plot the median and mean stats for the average fitnesses
    for stat in ('median', 'mean'):
        # Calculate the curves
        curves = []
        # For each run type, get the statistical curves
        for label, results in all_data_by_type.items():
            gens, mat, _ = results['avg']  # Get the average data

            # Generate the median or mean curve
            curve = np.median(mat, axis=0) if stat == 'median' else np.mean(mat, axis=0)
            curves.append((label, gens, curve))
        
        # Plot the combined curves
        plot_combined(curves=curves,
                      ylabel=f"{stat.title()} Avg DistanceDifference",
                      out_path=os.path.join(out_base, f"combined_{stat}.svg"))
    
    # Report summary of final generation distance differences
    report_summary(all_data_by_type)


if __name__ == '__main__':
    main()
