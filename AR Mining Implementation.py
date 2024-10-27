import pandas as pd
import time
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Define function to load and preprocess dataset
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # Convert to basket format
    basket_data = pd.get_dummies(df['itemDescription']).groupby([df['Member_number'], df['Date']]).max()
    return basket_data

# Function to estimate brute-force time
def estimate_brute_force_time(num_items, baseline_time):
    brute_force_time_seconds = baseline_time * (2 ** num_items)
    return brute_force_time_seconds

# Define file paths for all datasets
file_paths = {
    "size_varied_dataset_1": "size_varied_dataset_5k.csv",
    "size_varied_dataset_2": "size_varied_dataset_20k.csv",
    "size_varied_dataset_3": "size_varied_dataset_30k.csv"
}

group_varied_file_paths = {
    "group_varied_dataset_10": "group_varied_dataset_20.csv",
    "group_varied_dataset_20": "group_varied_dataset_23.csv",
    "group_varied_dataset_30": "group_varied_dataset_28.csv"
}

# Baseline time for one itemset (hypothetical)
baseline_time_constant = 0.01  # in seconds

# Set further reduced support and confidence thresholds
min_support_threshold = 0.001
min_confidence_threshold = 0.001

# Initialize results dictionary
results = {}

# Process each size varied dataset
for name, path in file_paths.items():
    print(f"\nProcessing {name}...")
    
    # Load and preprocess data
    basket_data = load_and_preprocess_data(path)
    
    # Get the number of unique items
    num_items = basket_data.shape[1]
    
    # Run Apriori with optimizations
    start_time = time.time()
    frequent_itemsets = apriori(basket_data, min_support=min_support_threshold, use_colnames=True)
    apriori_time = time.time() - start_time
    
    # Estimate brute-force time
    brute_force_time = estimate_brute_force_time(num_items, baseline_time_constant)

    # Generate rules with further reduced confidence threshold
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence_threshold)
    
    # Store results
    results[name] = {
        "apriori_time": apriori_time,
        "brute_force_time": brute_force_time,
        "num_items": num_items,
    }
    
    # Print specified details for each dataset
    print(f"  Apriori Time: {apriori_time:.4f} seconds")
    print(f"  Estimated Brute-Force Time: {brute_force_time:.4f} seconds")
    print(f"  Number of Unique Items: {num_items}")
    print(f"  Generated Rules:\n{rules[['antecedents', 'consequents', 'confidence']].head() if not rules.empty else 'No rules generated'}")

# Process each group varied dataset
for name, path in group_varied_file_paths.items():
    print(f"\nProcessing {name}...")
    
    # Load and preprocess data
    basket_data = load_and_preprocess_data(path)
    
    # Get the number of unique items
    num_items = basket_data.shape[1]
    
    # Run Apriori with optimizations
    start_time = time.time()
    frequent_itemsets = apriori(basket_data, min_support=min_support_threshold, use_colnames=True)
    apriori_time = time.time() - start_time
    
    # Estimate brute-force time
    brute_force_time = estimate_brute_force_time(num_items, baseline_time_constant)

    # Generate rules with further reduced confidence threshold
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence_threshold)
    
    # Store results
    results[name] = {
        "apriori_time": apriori_time,
        "brute_force_time": brute_force_time,
        "num_items": num_items,
    }
    
    # Print specified details for each dataset
    print(f"  Apriori Time: {apriori_time:.4f} seconds")
    print(f"  Estimated Brute-Force Time: {brute_force_time:.4f} seconds")
    print(f"  Number of Unique Items: {num_items}")
    print(f"  Generated Rules:\n{rules[['antecedents', 'consequents', 'confidence']].head() if not rules.empty else 'No rules generated'}")

# Prepare data for plotting
size_varied_datasets = ["size_varied_dataset_1", "size_varied_dataset_2", "size_varied_dataset_3"]
group_varied_datasets = ["group_varied_dataset_10", "group_varied_dataset_20", "group_varied_dataset_30"]

size_apriori_times = [results[name]["apriori_time"] for name in size_varied_datasets]
size_brute_force_times = [results[name]["brute_force_time"] for name in size_varied_datasets]

group_apriori_times = [results[name]["apriori_time"] for name in group_varied_datasets]
group_brute_force_times = [results[name]["brute_force_time"] for name in group_varied_datasets]

# Plotting for Size Varied Datasets
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(size_varied_datasets, size_apriori_times, width=0.4, label='Apriori Time (seconds)', color='b', align='center')
plt.bar(size_varied_datasets, size_brute_force_times, width=0.4, label='Brute-Force Estimate (seconds)', color='r', align='edge')

plt.yscale('log')  # Use logarithmic scale
plt.xlabel('Size Varied Datasets')
plt.ylabel('Time (seconds)')
plt.title('Size Varied Datasets: Apriori vs. Brute-Force Time')
plt.xticks(rotation=15)
plt.legend()
plt.grid(axis='y')

# Plotting for Group Varied Datasets
plt.subplot(1, 2, 2)
plt.bar(group_varied_datasets, group_apriori_times, width=0.4, label='Apriori Time (seconds)', color='b', align='center')
plt.bar(group_varied_datasets, group_brute_force_times, width=0.4, label='Brute-Force Estimate (seconds)', color='r', align='edge')

plt.yscale('log')  # Use logarithmic scale
plt.xlabel('Group Varied Datasets')
plt.ylabel('Time (seconds)')
plt.title('Group Varied Datasets: Apriori vs. Brute-Force Time')
plt.xticks(rotation=15)
plt.legend()
plt.grid(axis='y')

# Show the plots
plt.tight_layout()
plt.show()
