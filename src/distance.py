import re
from collections import defaultdict, Counter
from zss import simple_distance, Node
import utils
from complexity import weighted_complexity
from ngram import NGramModel

def build_tree(node_list):
    """Reconstructs from a graphemic string input its underlying tree structure."""
    if not node_list:
        return None

    root = Node(node_list[0])
    current = root
    stack = []

    for item in node_list[1:]:
        if item.startswith("("):
            item = item[1:]
            new_node = Node(item)
            current.addkid(new_node)
            stack.append(current)
            current = new_node

        elif item.endswith(")"):
            item = item[:-1]
            new_node = Node(item)
            current.addkid(new_node)
            if stack:
                current = stack.pop()

        else:
            new_node = Node(item)
            current.addkid(new_node)
            current = new_node

    return root

def tree_distance(grapheme1, grapheme2, complexity=weighted_complexity):
    annotation1 = utils.get_annotation(grapheme1)
    annotation2 = utils.get_annotation(grapheme2)
    list1 = utils.decompose_grapheme(annotation1)
    list2 = utils.decompose_grapheme(annotation2)
    tree1 = build_tree(list1)
    tree2 = build_tree(list2)
        
    difference = simple_distance(tree1, tree2)
    similarity = (len(list1) + len(list2) - 2 * difference)
    return similarity, difference
    
def bos_distance(grapheme1, grapheme2, complexity=weighted_complexity):
    '''Bag-of-Shapes model'''
    annotation1 = utils.get_annotation(grapheme1)
    annotation2 = utils.get_annotation(grapheme2)
    similarity = 0
    distance = 0
    shapes1 = [string.strip('()') for string in utils.decompose_grapheme(annotation1) if string.strip('()').isalpha()]
    shapes2 = [string.strip('()') for string in utils.decompose_grapheme(annotation2) if string.strip('()').isalpha()]
    # remove all duplicates
    i = 0
    while i < len(shapes1):
        if shapes1[i] in shapes2:
            # similarity += grapheme_complexity(shapes1[i])
            similarity += len(shapes1[i])
            shapes2.remove(shapes1[i])
            shapes1.pop(i)
        else:
            i += 1

    # find similarities and distances for remaining shapes
    for element1 in shapes1:
        element1_basic_shape = ''.join(char for char in element1 if char.islower())
        basic_shapes2 = [''.join(char for char in element2 if char.islower()) for element2 in shapes2]
        matching_positions = [index for index, item in enumerate(basic_shapes2) if item == element1_basic_shape]
        
        # find closest match and calculate their similarity and difference
        champ_i = -1
        champ_similarity = float('-inf')
        champ_distance = 0
        for i in matching_positions:
            curr_similarity = len(set(element1) & set(shapes2[i]))
            curr_difference = len(set(element1) - set(shapes2[i]))
            if curr_similarity > champ_similarity:
                champ_similarity = curr_similarity
                champ_distance = curr_difference
                champ_i = i

        if champ_i != -1:
            similarity += champ_similarity
            distance += champ_distance
            shapes1.remove(element1)
            shapes2.pop(champ_i)

    # calculate distances
    for shape in shapes1:
        distance += complexity(shape)
    for shape in shapes2:
        distance += complexity(shape)

    # normalization = 4
    # similarity /= normalization
    # distance /= normalization
    #print(grapheme1["Letter"], grapheme2["Letter"], similarity,distance)
    return similarity, distance


def bucket_distance(grapheme1, grapheme2, complexity=weighted_complexity):
    annotation1 = utils.get_annotation(grapheme1, "rudimentary")
    annotation2 = utils.get_annotation(grapheme2, "rudimentary")
    
    similarity = 0
    difference = 0
    
    # clean data
    ignore_chars = '[()&*%]'
    annotation1 = re.sub(ignore_chars, '', annotation1)
    annotation2 = re.sub(ignore_chars, '', annotation2)
    
    # convert to lists
    list1 = re.findall(r'[a-zA-Z]|[^a-zA-Z].', annotation1)
    list2 = re.findall(r'[a-zA-Z]|[^a-zA-Z].', annotation2)

    # find overlap
    count1 = Counter(list1)
    count2 = Counter(list2)

    overlap = list((count1 & count2).elements())

    for element in overlap:
        similarity += complexity(element.lower())
        list1.remove(element)
        list2.remove(element)
    
    # find partially overlapping: C vs c, *k vs k

    # find total non-overlap
    for element in list1:
        difference += complexity(element.lower())
    for element in list2:
        difference += complexity(element.lower())
    
    return similarity, difference

def grapheme_script_distance(grapheme1, script, distance):
    '''Return the average distance between a grapheme and the graphemes of a script.
    Takes as input a grapheme dict, and a list of graphemes representing a script.'''
    
    # ensure grapheme1 is not in script
    script_minus_grapheme = [g for g in script if g != grapheme1]
    similarities = 0
    differences = 0
    size = 0
    for grapheme2 in script_minus_grapheme:
        similarity, difference = distance(grapheme1, grapheme2)
        if similarity > 0:
            similarities += similarity
            differences += difference
            size += 1
            
    if size == 0:
        return 0,0
    return similarities/size, differences/size

def script_distance(script, distance, complexity=weighted_complexity):
    def compute_distance(pairs):
        similarities, differences, count = 0, 0, 0
        for i, j in pairs:
            similarity, difference = distance(script[i], script[j], complexity)
            if similarity > 0:
                similarities += similarity
                differences += difference
                count += 1
        return similarities / count if count else 0, differences / count if count else 0

    if not script:
        return 0, 0
    
    if distance is bucket_distance:
        buckets = defaultdict(list)
        for grapheme in script:
            headshape = utils.get_headshape(grapheme)
            buckets[headshape].append(grapheme)

        for bucket in buckets.values():
            print([grapheme["Letter"] for grapheme in bucket])

        # Compute pairwise distances only within each bucket
        pairs = [(i, j) for bucket in buckets.values() for i in range(len(bucket)) for j in range(i + 1, len(bucket))]
    else:
        pairs = [(i, j) for i in range(len(script)) for j in range(i + 1, len(script))]

    similarities, differences = compute_distance(pairs)

    return round(similarities, 2), round(differences, 2)

def get_weirdest_grapheme(graphemes, distance_function):
    """For each script, return the grapheme which is most distant to the rest of the graphemes."""
    for script in graphemes:
        champ_grapheme = None
        champ_distance = (float('inf'), -float('inf'))
        for grapheme in script:
            curr_distance = grapheme_script_distance(grapheme, script, distance_function)
            if champ_distance[1] < curr_distance[1]:
                champ_distance = curr_distance
                champ_grapheme = grapheme
        print(f"The most dissimilar in script {utils.get_script(script)} is {champ_grapheme["Letter"]}")
        
        champ_grapheme = None
        champ_distance = (-float('inf'), -float('inf'))
        for grapheme in script:
            curr_distance = grapheme_script_distance(grapheme, script, distance_function)
            if champ_distance[0] < curr_distance[0]:
                champ_distance = curr_distance
                champ_grapheme = grapheme
                        
        print(f"The least similar grapheme in script {utils.get_script(script)} is {champ_grapheme["Letter"]}")
    print('-------------------------------------')


def evaluate(graphemes, test_size=0.25, num_trials=10, print_results=False):
    distance_functions = [bos_distance, bucket_distance, tree_distance]
    n_gram_models = []
    min_n = 2
    max_n = 4
    for n in range(min_n,max_n+1):
        n_gram_models.append(f"{n}-Gram Model")
    
    all_models = [func.__name__ for func in distance_functions] + n_gram_models

    accuracy_sums = {func.__name__: 0.0 for func in distance_functions}
    for n_gram in n_gram_models:
        accuracy_sums[n_gram] = 0.0
    
    script_accuracy = {model: defaultdict(float) for model in all_models}
    script_counts = defaultdict(float)
    
    confusion = {model: defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0}) for model in all_models}
    
    for i in range(num_trials):
        print(f"Running Trial {i+1}...")
        train_data, test_data = utils.train_test_split_stratified(graphemes, test_size=test_size)    
        
        for test_grapheme in test_data:
            true_script = test_grapheme["Script"]
            script_counts[true_script] += 1
    
        for distance_function in distance_functions:
            model_name = distance_function.__name__
            accuracy = 0
            for test_grapheme in test_data:
                true_script = test_grapheme["Script"]
                champ_sim = float('-inf')
                champ_diff = float('inf')
                champ_script = ''
                for script in train_data:
                    curr_sim, curr_diff = grapheme_script_distance(test_grapheme, script, distance_function)
                    script_name = utils.get_script(script)

                    if champ_sim < curr_sim:
                        champ_sim = curr_sim
                        champ_diff = curr_diff
                        champ_script = script_name

                true_script = test_grapheme["Script"]
                if champ_script == true_script:
                    accuracy += 1
                    script_accuracy[model_name][true_script] += 1
                    confusion[model_name][true_script]['TP'] += 1
                else:
                    confusion[model_name][champ_script]['FP'] += 1
                    confusion[model_name][true_script]['FN'] += 1

            total_accuracy = accuracy / len(test_data)
            accuracy_sums[distance_function.__name__] += total_accuracy

        # n-gram models
        for n in range(min_n,max_n+1):
            model_name = f"{n}-Gram Model"
            model = NGramModel(n=n, graphemes=graphemes)
            model.train(utils.flatten(train_data))
            accuracy = model.test(test_data)
            accuracy_sums[model_name] += accuracy
            predictions = model.predict_all(test_data)  # returns list of (true_script, predicted_script)

            for true_script, predicted_script in predictions:
                if true_script == predicted_script:
                    script_accuracy[model_name][true_script] += 1
                    confusion[model_name][true_script]['TP'] += 1
                else:
                    confusion[model_name][predicted_script]['FP'] += 1
                    confusion[model_name][true_script]['FN'] += 1

    average_accuracies = [(name, accuracy_sums[name] / num_trials) for name in accuracy_sums]
    
    # Print results
    if print_results:
        print("Average Accuracy over", num_trials, "trials:")
        for score in average_accuracies:
            print(score)
            
        print('-------------------------------------')
            
        print("Average Percentage Accuracy by Script over", num_trials, "trials:")
        for model_name in all_models:
            print(f"\n{model_name}:")
            for script_name in script_counts:
                correct = script_accuracy[model_name][script_name]
                total = script_counts[script_name]
                percent = (correct / total) * 100 if total > 0 else 0
                print(f"  {script_name}: {percent:.2f}%")
        print('-------------------------------------')

        # Artificial intelligence was used to assist in the construction of F1 score computation code
        print("\nF1 Scores per Script:")
        for model_name in all_models:
            print(f"\nModel: {model_name}")
            for script, counts in confusion[model_name].items():
                tp = counts['TP']
                fp = counts['FP']
                fn = counts['FN']
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                print(f"  {script}: F1 = {f1:.4f}  (Precision={precision:.4f}, Recall={recall:.4f})")

        print("\nCumulative (Macro-Averaged) F1 Scores:")
        for model_name in all_models:
            f1_total = 0
            num_scripts = 0
            for script, counts in confusion[model_name].items():
                tp = counts['TP']
                fp = counts['FP']
                fn = counts['FN']
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                f1_total += f1
                num_scripts += 1

            macro_f1 = f1_total / num_scripts if num_scripts > 0 else 0
            print(f"  {model_name}: Macro F1 = {macro_f1:.4f}")
        print('-------------------------------------')
        
        # Compute baseline Random F1 Score
        all_test_labels = []
        for i in range(num_trials):
            _, test_data = utils.train_test_split_stratified(graphemes, test_size=test_size)
            all_test_labels.extend([g["Script"] for g in test_data])

        total = len(all_test_labels)
        label_counts = Counter(all_test_labels)
        probs = [count / total for count in label_counts.values()]
        random_f1 = sum(prob ** 2 for prob in probs)

        print(f"Expected Random Baseline F1 Score: {random_f1:.4f}")
        print('-------------------------------------')

    return average_accuracies, script_accuracy, confusion
