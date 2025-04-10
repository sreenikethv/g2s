import re
from sklearn.model_selection import train_test_split

def flatten(graphemes):
    '''Flattens a list of list of dicts (such as graphemes) into a list of dicts.'''
    return [grapheme for script in graphemes for grapheme in script]

def decompose_grapheme(s):
    '''
    Takes a graphemic string as input
    and outputs  a list of its component shape and operation strings.
    '''
    # Match transitions: letter -> non-letter, non-letter -> letter, or boundaries at parentheses
    pattern = re.compile(r'([a-zA-Z]+|[^a-zA-Z()]+|[()])')
    patterns = pattern.findall(s)

    i = 0
    while i < len(patterns):
        if patterns[i] == "(":
            patterns[i] = patterns[i] + patterns[i+1]
            patterns.pop(i+1)
        elif patterns[i] == ')':
            patterns[i-1] = patterns[i-1] + patterns[i]
            patterns.pop(i)
        i+=1
    return patterns

def validate_shape(s: str):
    '''Helper function for validate() ensuring a shape string is valid.'''
    pattern1 = r'^BATb$'
    pattern2 = r'^[HBMS][OUAI]C[KV][lfvy]$'
    pattern3 = r'^[HBMS][OUAI][TCD][a-z]$'
    if re.match(pattern1, s) or re.match(pattern2, s) or re.match(pattern3, s):
        return True
    return False
def validate_operation(s: str):
    '''Helper function for validate() ensuring an operation string is valid.'''
    pattern1 = r'^(?:[1-5]{2})+\+(?:[1-5]{2})+$'
    if re.match(pattern1, s):
        left, right = s.split('+')
        return len(left) == len(right)

    # Pattern for + operator
    pattern2 = r'^(?:[1-5]{2})+\+(?:[1-5]{2})+$'
    # Pattern for & operator
    pattern3 = r'^(?:[1-5]{2})+&(?:[1-5]{2})+$'
    # Pattern for % operator
    pattern4 = r'^(?:[0-6]{2})+%(?:[0-6]{2})+$'
    pattern1 = r'^[1-5]{2}\*$'

    if re.match(pattern1, s):
        return True

    if re.match(pattern2, s):
        left, right = s.split('+')
    elif re.match(pattern3, s):
        left, right = s.split('&')
    elif re.match(pattern4, s):
        left, right = s.split('%')
    else:
        return False

    return len(left) == len(right)

def validate(grapheme):
    '''Validates a graphemic string to ensure it is properly formatted.'''
    annotation = get_annotation(grapheme)
    letter = get_annotation(grapheme, "Letter")
    
    elements = decompose_grapheme(annotation)
    elements = [element.strip('()') for element in elements]
    elements = [element for element in elements if element.strip()]

    # special circumstance when grapheme begins with a blank shape
    if elements[0] == "BUTb":
        elements = elements[2:]

    for i, element in enumerate(elements):
        if i%2==0: # if shape
            if validate_shape(element) is False:
                print(f"Invalid Shape: {letter}, {annotation}; Problem area: \"{element}\"")
                return False
        else: # if grapheme
            if validate_operation(element) is False:
                print(elements)
                print(f"Invalid Operation: {letter}, {annotation}; Problem area: {element}")
                return False
    return True

def validate_all_graphemes(graphemes, print_output=True):
    all_valid = True
    for script in graphemes:
            for grapheme in script:
                curr_valid = validate(grapheme)
                if not curr_valid:
                    all_valid = False
    if print_output and all_valid:
        print("All graphemes are validated!")
    return all_valid
                
def unique_values(graphemes, key):
    '''Return all unique values for a given key in a list of dictionaries.'''
    unique_vals = set()
    for script in graphemes:
        for grapheme_dict in script:
            if key in grapheme_dict:
                unique_vals.add(grapheme_dict[key])
    return unique_vals

def to_rudimentary_notation(annotation):
    '''
    Converts a grapheme string to a rudimentary string annotation,
    preserving only the basic shape (capitalized if of size H or B),
    as well as parentheses and special operations *&%.
    '''
    s = re.sub(r'[^a-zHB*&%()]','',annotation)
        
    result = []
    i = 0
    while i < len(s):
        if s[i].isupper():
            if i + 1 < len(s):
                result.append(s[i + 1].upper()) 
            i += 2
        else:
            result.append(s[i])
            i += 1
    output = ''.join(result)
    return output

def get_script(script):
    return script[0]["Script"]
    
def get_annotation(grapheme, label="Annotation"):
    if isinstance(grapheme, dict):
        annotation = grapheme[label]
    else:
        annotation = grapheme
    return annotation

def get_headshape(grapheme, many_headshapes_allowed=False):
    '''
    Returns the headshape (or headshapes) of a grapheme,
    defined as the largest shape which determines the nature of a grapheme.
    Currently, the horizontal and vertical shapes are dispreferenced as headshapes.
    '''
    annotation = get_annotation(grapheme, "rudimentary")

    # do not use the blank shape <b> as a headshape
    annotation = annotation.replace('B','').replace('b','')
    
    # treat shapes 'l' and 'f' as medium
    annotation = annotation.replace('L', 'l').replace('F', 'f')
    
    all_caps = ''.join([char for char in annotation if char.isupper()])
    if len(all_caps) > 0:
        return all_caps[0]
    
    if many_headshapes_allowed:
        return annotation
    return annotation[0]

def train_test_split_uniform(graphemes, test_size=0.25):
    """
    Randomizes all graphemes and chooses samples from this random list.
    Does not account for variations in script length which may lead to over-sampling.
    """
    graphemes = flatten(graphemes)
    train_data, test_data = train_test_split(
            graphemes, test_size=test_size)
    return train_data, test_data

def train_test_split_stratified(graphemes, test_size=0.25):
    """
    Performs a stratified train-test split, ensuring each sublist is split separately.
    Keeps train_data in its original nested format while flattening test_data.
    """
    train_data, test_data = [], []

    for script in graphemes:
        train_sample, test_sample = train_test_split(script, test_size=test_size)
        train_data.append(train_sample)
        test_data.extend(test_sample)

    return train_data, test_data
