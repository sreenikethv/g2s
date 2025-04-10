"""Create an n-gram model for predicting a grapheme's script label."""
# Artificial intelligence was used to assist in the construction of this code file
import math
from collections import defaultdict, Counter
import numpy as np
import utils

class NGramModel:
    """Instantiate an NGram model."""
    def __init__(self, n=2, graphemes=None):
        self.n = n
        self.models = defaultdict(lambda: defaultdict(Counter))
        self.vocab = defaultdict(set)
        self.graphemes_unflattened = graphemes
        self.graphemes = utils.flatten(graphemes)
        
    def pad(self, annotation):
        start_token = '$'
        end_token = '@'
        return start_token*(self.n-1) + annotation + end_token
        
    def train(self, train_data=None):
        """Train model on a set of input training data."""
        if train_data is None:
            train_data = self.graphemes
        
        for grapheme in train_data:
            script = grapheme["Script"]
            annotation = grapheme["Annotation"]
            padded = self.pad(annotation)
            
            for i in range(len(padded) - self.n + 1):
                prefix = padded[i:i+self.n-1]
                next_char = padded[i+self.n-1]
                self.models[script][prefix][next_char] += 1
                self.vocab[script].update(annotation)
                    
    def get_probability(self, example_string, script, alpha=1):
        """Compute probability of example_string belonging to a given input script."""
        if script not in self.models:
            return 0.0  # Script label not found in training
        
        padded = self.pad(example_string)
        log_prob = 0.0
        
        for i in range(len(padded) - self.n + 1):
            prefix = padded[i:i+self.n-1]
            next_char = padded[i+self.n-1]
            
            prefix_counts = self.models[script][prefix]
            prefix_total = sum(prefix_counts.values())
            char_count = prefix_counts[next_char] if next_char in prefix_counts else 0
            
            # add-alpha smoothing
            prob = (char_count + alpha) / (prefix_total + len(self.vocab[script]) + alpha)
            log_prob += np.log(prob)
        
        return np.exp(log_prob)  # convert log probability back to normal probability
    
    def predict(self, example_string, print_bool=False):
        """Predict most probable script for a given example string."""
        probabilities = {script: self.get_probability(example_string, script) for script in self.models}
        predicted_script = max(probabilities, key=probabilities.get)
        
        if print_bool:
            print(f"Predicted script: {predicted_script}")
            print("Probabilities:")
            for key, val in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
                print(f"{key}: {round(-math.log(val)/len(example_string),2)}")

        return predicted_script, probabilities
    
    def perplexity(self, example_string, script):
        """Compute the perplexity of a given example string for a script."""
        prob = self.get_probability(example_string, script)
        if prob == 0:
            return float('inf')
        return np.power(1 / prob, 1 / len(example_string))
    
    def test(self, test_data):
        """Evaluate the model by calculating accuracy on labeled test data."""
        correct = 0
        total = 0
        for sample in test_data:
            example_string = sample["Annotation"]
            true_label = sample["Script"]
            predicted_label, _ = self.predict(example_string)
            if predicted_label == true_label:
                correct += 1
            total += 1
        return correct / total if total > 0 else 0.0
    
    def train_and_test(self, test_size=0.25, print_bool=False):
        """Split the data into training and test sets, then train and evaluate the model."""
        train_data, test_data = utils.train_test_split_stratified(self.graphemes_unflattened, test_size=test_size)
        
        # train and test
        self.train(utils.flatten(train_data))
        accuracy = self.test(test_data)
        
        if print_bool:
            print(f"Model accuracy on test set: {accuracy * 100:.2f}%")
        return accuracy

    def predict_all(self, test_data):
        """Return a list of (true_script, predicted_script) for evaluation."""
        predictions = []
        for sample in test_data:
            example_string = sample["Annotation"]
            true_label = sample["Script"]
            predicted_label, _ = self.predict(example_string)
            predictions.append((true_label, predicted_label))
        return predictions
