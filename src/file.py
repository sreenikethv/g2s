import pandas as pd
from utils import to_rudimentary_notation, flatten
from complexity import weighted_complexity

def read(filename):
    '''
    Read the CSV file and group by 'Script',
    such that the output is a list of list of dictionaries,
    wherein each list of dictionaries represents a script
    and each dictionary represents a grapheme.
    '''
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename, encoding='utf-8')
    # Remove completely empty rows
    df = df.dropna(how='all')
    # replace empty cells with None (or any other placeholder)
    df = df.where(pd.notnull(df), None)

    grouped = df.groupby('Script')
    graphemes = [group.to_dict(orient='records') for _, group in grouped]

    for script in graphemes:
        for grapheme in script:
            annotation = grapheme['Annotation']
            grapheme['rudimentary'] = to_rudimentary_notation(annotation)
            grapheme['complexity'] = weighted_complexity(annotation)

    return graphemes

def write(graphemes, output_filename):
    '''Outputs a file with additional data for each grapheme.'''
    # Flatten the list of list of dicts into a single list of dictionaries
    flattened_graphemes = flatten(graphemes)
    if not flattened_graphemes:
        raise ValueError("No data to write")
    df = pd.DataFrame(flattened_graphemes)
    df.to_csv(output_filename, index=False, encoding='utf-8')

