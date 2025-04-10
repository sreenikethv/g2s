import utils

"""
Helps find the implicational hierarchy of shapes and operations across scripts,
at various levels of granularity.
A + marks the presence of the corresponding shape-series/operation within that script.
A - marks its absence.
"""

def hierarchy_shapes(graphemes, granularity='low', ignore_numerals=False):
    if granularity == 'low':
        print("ocvrfesi")

    for script in graphemes:
        script_name = utils.get_script(script)
        
        if ignore_numerals and "Numerals" in script_name:
            continue
        
        unique_shapes = set()
        output = []

        for grapheme_dict in script:
            all_chars = [char for char in grapheme_dict["Annotation"] if char.islower()]
            for char in all_chars:
                unique_shapes.add(char)
        
        if granularity == 'low':
            output.append('+' if 'o' in unique_shapes else '-')
            output.append('+' if {'c', 'a', 'u', 'n'} & unique_shapes else '-')
            output.append('+' if {'v', 'y'} & unique_shapes else '-')
            output.append('+' if {'r', 'g', 't', 'j'} & unique_shapes else '-')
            output.append('+' if {'f', 'l'} & unique_shapes else '-')
            output.append('+' if {'e', 'k', 'm', 'w'} & unique_shapes else '-')
            output.append('+' if {'s', 'z', 'h', 'd'} & unique_shapes else '-')
            output.append('+' if 'i' in unique_shapes else '-')

        elif granularity == 'medium':
            output.append('+' if {'v', 'y'} & unique_shapes else '-')
            output.append('+' if {'c', 'a'} & unique_shapes else '-')
            output.append('+' if 'o' in unique_shapes else '-')
            output.append('+' if 'f' in unique_shapes else '-')
            output.append('+' if 'l' in unique_shapes else '-')
            output.append('+' if {'r', 'j'} & unique_shapes else '-')
            output.append('+' if {'g', 't'} & unique_shapes else '-')
            output.append('+' if {'u', 'n'} & unique_shapes else '-')
            output.append('+' if {'e', 'k'} & unique_shapes else '-')
            output.append('+' if {'m', 'w'} & unique_shapes else '-')
            output.append('+' if {'s', 'z'} & unique_shapes else '-')
            output.append('+' if {'h', 'd'} & unique_shapes else '-')
            output.append('+' if 'i' in unique_shapes else '-')

        elif granularity == 'high':
            for shape in list('flvycaorjgtunekmwszhdi'):
                output.append('+' if shape in unique_shapes else '-')

        print(f"{''.join(output)}: {script_name}")


def hierarchy_operations(graphemes):
    print("+&%*")
    for script in graphemes:
        script_name = utils.get_script(script)
        unique_operations = set()
        output = []

        for grapheme_dict in script:
            all_special_chars = [char for char in grapheme_dict["Annotation"] if not char.isalpha()]
            for char in all_special_chars:
                unique_operations.add(char)

        output.append('+' if '+' in unique_operations else '-')
        output.append('+' if '&' in unique_operations else '-')
        output.append('+' if '%' in unique_operations else '-')
        output.append('+' if '*' in unique_operations else '-')
        # conclusion: no hierarchy found
        print(f"{''.join(output)}: {script_name}")
