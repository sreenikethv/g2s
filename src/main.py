# to activate virtual environment: source g2s/bin/activate
import file
import utils
import complexity
import distance
import implicational

def main():
    filename = "input.csv"
    
    # input_conflated.csv combines numeral systems with their corresponding scripts,
    # and uppercase with lowercase letters,
    # and shows slightly better model performance
    
    # telugu_unchecked.csv removes the checkmarks from annotated Telugu
    graphemes = file.read(filename)
    
    # validate all graphemes
    utils.validate_all_graphemes(graphemes)
    
    # sort script into buckets
    for script in graphemes:
        print(utils.get_script(script), distance.script_distance(script, distance.bucket_distance))
    print('-------------------------------------')
    
    # Implicational hierarchy
    implicational.hierarchy_shapes(graphemes, granularity='low')
    implicational.hierarchy_operations(graphemes)
    print('-------------------------------------')
    
    # evaluate predictions
    distance.evaluate(graphemes, num_trials=30, print_results=True)

    # Find script complexity statistics
    for script in graphemes:
        complexity.statistics(script)
        
    # Find all script complexities and plot
    script_complexities = complexity.script_complexities(graphemes)
    complexity.plot_complexities(script_complexities)
    
if __name__ == "__main__":
    main()
