"""
This module offers diferent measures of the complexity of an individual grapheme.
Graphemic complexity is calculated by length_complexity, shape_count_complexity,
weighted_complexity, and size_weighted_complexity.
"""

from statistics import mean, median, pvariance, pstdev
import matplotlib.pyplot as plt
import adjustText
import numpy as np
import utils

def length_complexity(grapheme):
    """Complexity is measured as a direct function of the graphemic string length."""
    annotation = utils.get_annotation(grapheme)
    total_complexity = sum(1 for char in annotation)
    normalization = 4
    return total_complexity/normalization

def shape_count_complexity(grapheme):
    """Complexity is measured as the number of (non-null) shapes in the grapheme."""
    annotation = utils.get_annotation(grapheme)
    total_complexity = sum(1 for char in annotation if char.islower()
                           and char!='b')
    return total_complexity

def weighted_complexity(grapheme):
    """shape_count_complexity where different shapes are weighted differently."""
    annotation = utils.get_annotation(grapheme)
    total_complexity = 0
    for char in annotation:
        if char.islower():
            if char=='b':
                total_complexity += 0
            elif char in list('szhd') or list('ekwm'):
                total_complexity += 2
            else:
                total_complexity += 1
    return total_complexity

def size_weighted_complexity(grapheme):
    """size_weighted_complexity where the larger a shape is, the more weight it is given."""
    annotation = utils.get_annotation(grapheme)
    total_complexity = 0
    for char in annotation:
        if char=='H':
            total_complexity += 3
        elif char=='B':
            total_complexity += 2
        elif char=='M':
            total_complexity += 1
        elif char=='O':
            total_complexity += 3
        elif char=='U':
            total_complexity += 2
        elif char=='A':
            total_complexity += 1
    total_complexity += weighted_complexity(grapheme)
    return total_complexity

def script_complexity(script, complexity):
    total_complexity = 0
    for grapheme in script:
        total_complexity += complexity(grapheme)
    avg_complexity = total_complexity/len(script)
    return avg_complexity

def statistics(graphemes, complexity=weighted_complexity):
    complexities = [complexity(grapheme) for grapheme in graphemes]
    script = utils.get_script(graphemes)

    sig_figs = 3
    stats_info = {
        "Script": script,
        "Count": len(complexities),
        "Mean": round(mean(complexities), sig_figs),
        "Median": round(median(complexities), sig_figs),
        "Min": min(complexities),
        "Max": max(complexities),
        "Range": round(max(complexities) - min(complexities), sig_figs),
        "Variance (Population)": round(pvariance(complexities), sig_figs),
        "Std Dev (Population)": round(pstdev(complexities), sig_figs),
    }

    print(f"--- Statistics for {script} ---")
    for key, value in stats_info.items():
        print(f"{key}: {value}")

    return stats_info

def script_complexities(graphemes, complexity=weighted_complexity):
    print(f"Script Complexities ({complexity.__name__})")
    script_complexities = []
    for script in graphemes:
        script_name = utils.get_script(script)
        scriptcomplexity = script_complexity(script, complexity)
        script_length = len(script)
        script_complexities.append(
            (script_name, scriptcomplexity, script_length))
    script_complexities.sort(key=lambda x: x[1])
    # Print the sorted scripts
    for script_name, scriptcomplexity, script_length in script_complexities:
        print(f"{script_name} ({script_length}): {round(scriptcomplexity, 3)}")
    print('-------------------------------------')
    return script_complexities

def plot_complexities(script_complexities):
    # Extract data
    script_names = [entry[0] for entry in script_complexities]
    script_complexities_values = [entry[1] for entry in script_complexities]
    script_lengths = [entry[2] for entry in script_complexities]
    
    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(script_lengths, script_complexities_values, color='blue', label='Scripts')
    # for i, script_name in enumerate(script_names):
    #     plt.annotate(script_name, (script_lengths[i], script_complexities_values[i]), fontsize=8, ha='right')
    texts = []
    for i, script_name in enumerate(script_names):
        texts.append(plt.text(script_lengths[i], script_complexities_values[i], script_name))
    adjustText.adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))  

    #  Perform linear regression to find trend line
    z = np.polyfit(script_lengths, script_complexities_values, 1)
    p = np.poly1d(z)
    plt.plot(script_lengths, p(script_lengths), color='black', linestyle='--', label='Trend line')

    # Plot
    plt.ylim(bottom=0)
    plt.xlabel("Script Length")
    plt.ylabel("Script Complexity")
    plt.title("Script Length vs Script Complexity")
    plt.legend()
    plt.grid(True)
    plt.show()
