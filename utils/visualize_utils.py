import numpy as np
import matplotlib.pyplot as plt
import math
# Define richer color gradient function
def get_color_gradient(start_color, end_color, steps):
    return [
        (
            start_color[0] + (end_color[0] - start_color[0]) * i / (steps - 1),
            start_color[1] + (end_color[1] - start_color[1]) * i / (steps - 1),
            start_color[2] + (end_color[2] - start_color[2]) * i / (steps - 1),
        )
        for i in range(steps)
    ]

def plot_divergences_list(divergences_list, labels, title, xlabel="Layer", ylabel=r'$\text{D}_\text{l}$', save_path=None, figsize=(12, 7), title_fontsize=30, xlabel_fontsize=26, ylabel_fontsize=26, legend_fontsize=24, ticks_fontsize=24, ylim=None):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create chart with more modern size ratio
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # Use more elegant color scheme
    colors = ['#3498db', '#e74c3c', '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B',
              '#6B5B95', '#F7CAC9', '#92A8D1', '#88B04B', '#F78DA7',
              '#9B2335', '#5B5EA6', '#DD4124', '#009B77', '#B565A7',
              '#955251', '#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9']
    
    # Set background style
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    for i, (divergences, label) in enumerate(zip(divergences_list, labels)):
        # Assume divergences is a 2D array, each row represents a set of divergence
        # Calculate mean and standard deviation for each layer
        divergences_array = np.array(divergences)
        avg_divergences = np.mean(divergences_array, axis=0)
        std_divergences = np.std(divergences_array, axis=0)
        
        # Draw mean line with thicker lines and larger markers
        ax.plot(avg_divergences, marker='o', linestyle='-', color=colors[i], 
                label=label, linewidth=3, markersize=8, markeredgecolor='white', 
                markeredgewidth=1.5)
        
        a = 0.5
        # Draw standard deviation shadow area with softer transparency
        ax.fill_between(range(len(avg_divergences)), 
                       avg_divergences - a*std_divergences, 
                       avg_divergences + a*std_divergences, 
                       color=colors[i], alpha=0.2, edgecolor=colors[i], 
                       linewidth=0.5)
    
    # Beautify legend
    legend = ax.legend(loc='upper left', frameon=True, 
                      fancybox=True, shadow=True, fontsize=legend_fontsize,
                      framealpha=0.9, facecolor='white', edgecolor='#dee2e6',
                      borderpad=0.5, labelspacing=0.3)
    
    # Set title style
    ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=20, 
                color='#2c3e50', fontfamily='sans-serif')
    
    # Set axis label styles
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, fontweight='600', color='#34495e', 
                  fontfamily='sans-serif', labelpad=10)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, fontweight='600', color='#34495e', 
                  fontfamily='sans-serif', labelpad=10)
    
    # Set fixed 6 x-axis labels (using original layer index, evenly distributed at front)
    num_layers = len(avg_divergences) if divergences_list else 0
    if num_layers > 0:
        # Calculate step size, evenly distributed at front (round up)
        step = math.ceil(num_layers / 6)
        x_positions = [i * step for i in range(6)]
        x_labels = [str(pos) for pos in x_positions]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)
    
    # Beautify grid
    ax.grid(True, linestyle='--', alpha=0.3, color='#bdc3c7', linewidth=0.8)
    
    # Set axis styles
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Set tick label styles
    ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize, 
                   colors='#7f8c8d', width=1.5, length=6)
    
    # Set y-axis range
    if ylim is not None:
        # If y-axis range is specified, set fixed range and 5 interval ticks
        y_min, y_max = ylim
        ax.set_ylim(y_min, y_max)
        # Set 5 evenly distributed y-axis ticks
        y_ticks = np.linspace(y_min, y_max, 6)  # 6 points produce 5 intervals
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{tick:.2f}' for tick in y_ticks])
    else:
        # Default adaptive, leave some margins
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    plt.show()

def plot_label_distributions(values1, values2, correctness1, correctness2, labels, title="Distribution Plot", threshold=None, save_path=None, normalize_ratio=True, use_accuracy_alpha=True):
    import numpy as np
    import matplotlib.patches as mpatches
    
    # Convert input data to numpy arrays, handle possible tensor input
    def to_numpy(data):
        if hasattr(data, 'cpu'):  # PyTorch tensor
            return data.cpu().numpy()
        elif hasattr(data, 'numpy'):  # TensorFlow tensor
            return data.numpy()
        else:
            return np.array(data)
    
    values1 = to_numpy(values1).flatten()
    values2 = to_numpy(values2).flatten()
    correctness1 = to_numpy(correctness1).flatten().astype(bool)
    correctness2 = to_numpy(correctness2).flatten().astype(bool)
    
    # Set figure size and style
    plt.figure(figsize=(12, 7))
    plt.style.use('default')
    
    # Set colors
    base_colors = ['#3498db', '#e74c3c']  # Basic blue and red
    
    # Determine bins range
    all_values = np.concatenate([values1, values2])
    bins = np.linspace(np.min(all_values), np.max(all_values), 26)  # 25 bins
    
    # Only calculate ratios when accuracy-based transparency is needed
    if use_accuracy_alpha:
        # First calculate all bin ratios for normalization
        def calculate_all_ratios(values, correctness, bins):
            ratios = []
            for i in range(len(bins)-1):
                mask = (values >= bins[i]) & (values < bins[i+1])
                if np.sum(mask) > 0:
                    ratio = np.sum(correctness[mask]) / np.sum(mask)
                    ratios.append(ratio)
                else:
                    ratios.append(np.nan)  # Mark bins with no data using NaN
            return np.array(ratios)
        
        # Calculate all ratios for both groups of data
        ratios1 = calculate_all_ratios(values1, correctness1, bins)
        ratios2 = calculate_all_ratios(values2, correctness2, bins)
        
        # Normalization processing
        if normalize_ratio:
            # Normalize ratios within each group to enhance differences between bins
            def normalize_ratios(ratios):
                valid_ratios = ratios[~np.isnan(ratios)]
                if len(valid_ratios) > 1:  # At least 2 valid values needed for normalization
                    min_ratio = np.min(valid_ratios)
                    max_ratio = np.max(valid_ratios)
                    if max_ratio > min_ratio:
                        # Min-Max normalization: stretch this group's ratio range to [0, 1]
                        normalized = (ratios - min_ratio) / (max_ratio - min_ratio)
                        
                        # Apply non-linear transformation to enhance contrast
                        # Use smoother transformation to balance differences between low and high value regions
                        enhanced = np.power(normalized, 1.5)  # 1.5 power transformation, moderately amplify high value differences
                        # Or use sigmoid-like smooth transformation
                        # enhanced = normalized + 0.3 * (normalized - 0.5) * np.abs(normalized - 0.5)  # Smooth S-shaped transformation
                        
                        return np.where(np.isnan(ratios), np.nan, enhanced)
                    else:
                        # If all ratios are the same, set to medium transparency
                        return np.where(np.isnan(ratios), np.nan, 0.5)
                elif len(valid_ratios) == 1:
                    # Only one valid value, set to medium transparency
                    return np.where(np.isnan(ratios), np.nan, 0.5)
                return ratios  # No valid values, return original
            
            ratios1_norm = normalize_ratios(ratios1)
            ratios2_norm = normalize_ratios(ratios2)
        else:
            ratios1_norm = ratios1
            ratios2_norm = ratios2
    else:
        # When not using accuracy-based transparency, set to None
        ratios1_norm = None
        ratios2_norm = None
    
    # Draw histogram with correctness ratio transparency
    def plot_ratio_based_histogram(values, correctness, ratios_norm, color, label):
        # Calculate statistics for each bin - use count instead of density
        hist_counts, bin_edges = np.histogram(values, bins=bins, density=False)
        # Convert to percentage: divide each bin's count by total and multiply by 100
        total_count = len(values)
        hist_percentages = (hist_counts / total_count) * 100
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Draw using pre-calculated normalized ratios
        for i in range(len(bin_edges)-1):
            mask = (values >= bin_edges[i]) & (values < bin_edges[i+1])
            if np.sum(mask) > 0:
                percentage = hist_percentages[i]
                
                # Decide whether to use accuracy-based transparency based on parameters
                if use_accuracy_alpha and ratios_norm is not None and not np.isnan(ratios_norm[i]):
                    # Set transparency based on ratio, use larger transparency range to enhance contrast
                    alpha = 0.1 + ratios_norm[i] * 0.8  # alpha range: 0.1-0.9, greater contrast
                else:
                    # Use uniform transparency
                    alpha = 0.7
                
                # Draw histogram bar for this bin
                if percentage > 0:
                    plt.bar(bin_centers[i], percentage, width=bin_width*0.8, 
                           color=color, alpha=alpha, edgecolor='white', linewidth=0.5)
    
    # Draw histograms for both groups of data
    plot_ratio_based_histogram(values1, correctness1, ratios1_norm, base_colors[0], labels[0])
    plot_ratio_based_histogram(values2, correctness2, ratios2_norm, base_colors[1], labels[1])
    
    # Then draw prominent KDE curves, adjusted to percentage scale
    from scipy import stats
    
    def plot_kde_as_percentage(values, color, label):
        # Calculate KDE
        kde = stats.gaussian_kde(values)
        # Create x-axis points
        x_range = np.linspace(np.min(all_values), np.max(all_values), 200)
        # Calculate density
        density = kde(x_range)
        # Convert to percentage: density × bin width × 100
        bin_width = bins[1] - bins[0]
        percentage_density = density * bin_width * 100
        # Draw curve
        plt.plot(x_range, percentage_density, color=color, linewidth=3, alpha=0.9, label=label)
    
    plot_kde_as_percentage(values1, base_colors[0], labels[0])
    plot_kde_as_percentage(values2, base_colors[1], labels[1])
    
    # Create legend, including transparency description
    legend_elements = [
        plt.Line2D([0], [0], color=base_colors[0], lw=3, label=labels[0]),
        plt.Line2D([0], [0], color=base_colors[1], lw=3, label=labels[1]),
    ]
    
    # if normalize_ratio:
    #     legend_elements.extend([
    #         mpatches.Patch(color='gray', alpha=0.9, label='Highest Correctness (Enhanced)'),
    #         mpatches.Patch(color='gray', alpha=0.5, label='Medium Correctness (Enhanced)'),
    #         mpatches.Patch(color='gray', alpha=0.1, label='Lowest Correctness (Enhanced)')
    #     ])
    # else:
    #     legend_elements.extend([
    #         mpatches.Patch(color='gray', alpha=0.8, label='High Correctness (0.8-1.0)'),
    #         mpatches.Patch(color='gray', alpha=0.5, label='Medium Correctness (0.4-0.6)'),
    #         mpatches.Patch(color='gray', alpha=0.2, label='Low Correctness (0.0-0.2)')
    #     ])
    
    # Optimize threshold line style
    if threshold is not None:
        plt.axvline(threshold, color='#2c3e50', linestyle='--', linewidth=2, 
                   label=f'Threshold = {threshold:.3f}', alpha=0.8)
        legend_elements.append(plt.Line2D([0], [0], color='#2c3e50', lw=2, 
                                         linestyle='--', label=f'Threshold = {threshold:.3f}'))
    
    # Style optimization
    plt.xlabel("TVI", fontsize=22, fontweight='bold')
    plt.ylabel("Percentage (%)", fontsize=22, fontweight='bold')
    plt.title(title, fontsize=24, fontweight='bold', pad=20)
    
    # Set Y-axis to percentage format - now directly display percentage, no conversion needed
    from matplotlib.ticker import FuncFormatter
    def to_percent(y, position):
        return f'{y:.1f}'
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    
    # Optimize legend style - use custom legend elements
    plt.legend(handles=legend_elements, frameon=True, fancybox=True, shadow=True, 
              fontsize=22, loc='upper right', framealpha=0.9)
    
    # Optimize grid and background
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.gca().set_facecolor('#fafafa')
    
    # Optimize coordinate axis styles
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('#cccccc')
    plt.gca().spines['bottom'].set_color('#cccccc')
    
    # Tick styles
    plt.tick_params(axis='both', which='major', labelsize=22, colors='#666666')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=800, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        plt.show()
