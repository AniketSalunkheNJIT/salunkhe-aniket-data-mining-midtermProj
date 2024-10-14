import pandas as pd
import itertools
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import time
import os
from typing import List, Tuple
from tabulate import tabulate


pd.set_option('display.max_colwidth', None)  
pd.set_option('display.max_columns', None)   
pd.set_option('display.width', 800)       
pd.set_option('display.float_format', '{:.4f}'.format)

# Function to load dataset
def load_dataset(dataset_choice: str) -> List[List[str]]:
    """
    Load dataset based on the user's choice and return transactions.
    """
    datasets = {
        '1': 'datasets/amazon_dataset.csv',
        '2': 'datasets/best_buy_dataset.csv',
        '3': 'datasets/kmart_dataset.csv',
        '4': 'datasets/nike_dataset.csv',
        '5': 'datasets/generic_dataset.csv'
    }

    dataset_file = datasets.get(dataset_choice)

    if not dataset_file:
        print(f"Error: Invalid dataset choice: {dataset_choice}")
        return []

    if not os.path.exists(dataset_file):
        print(f"Error: Dataset not found: {dataset_file}")
        return []
    
    try:
        df = pd.read_csv(dataset_file)
        transactions = df['Transaction'].apply(lambda x: [item.strip() for item in x.split(',')]).tolist()
        print(f"Successfully loaded dataset: {dataset_file}")
        return transactions
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

# Brute-force frequent itemsets generation
def brute_force_frequent_itemsets(transactions: List[List[str]], min_support: float) -> List[Tuple[Tuple[str], float]]:
    """
    Generate frequent itemsets using brute-force method and sort by support.
    """
    total_transactions = len(transactions)
    items = set(itertools.chain.from_iterable(transactions))
    itemsets = []
    transaction_list = list(map(set, transactions))
    for k in range(1, len(items) + 1):
        for combination in itertools.combinations(sorted(items), k):
            support_count = sum(1 for transaction in transaction_list if set(combination).issubset(transaction))
            support = support_count / total_transactions
            if support >= min_support:
                itemsets.append((combination, support))
    itemsets.sort(key=lambda x: (-x[1], x[0]))
    return itemsets

# Apriori frequent itemsets generation
def run_apriori(transactions: List[List[str]], min_support: float) -> pd.DataFrame:
    """
    Run Apriori algorithm to find frequent itemsets and sort by support.
    """
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: tuple(sorted(x)))
    frequent_itemsets = frequent_itemsets[['itemsets', 'support']]
    frequent_itemsets = frequent_itemsets.sort_values(by=['support', 'itemsets'], ascending=[False, True]).reset_index(drop=True)
    return frequent_itemsets

# FP-Growth frequent itemsets generation
def run_fp_growth(transactions: List[List[str]], min_support: float) -> pd.DataFrame:
    """
    Run FP-Growth algorithm to find frequent itemsets and sort by support.
    """
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: tuple(sorted(x)))
    frequent_itemsets = frequent_itemsets[['itemsets', 'support']]
    frequent_itemsets = frequent_itemsets.sort_values(by=['support', 'itemsets'], ascending=[False, True]).reset_index(drop=True)
    return frequent_itemsets

def print_association_rules(rules: pd.DataFrame):
    """
    Nicely formatted output of the association rules to match the example format.
    """
    if rules.empty:
        print("No association rules to display.")
        return
    
    for idx, row in rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        confidence = row['confidence'] * 100  # Convert to percentage
        support = row['support'] * 100        # Convert to percentage
        
        print(f"Rule {idx + 1}: {antecedents} -> {consequents}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Support: {support:.2f}%\n")

# Generate association rules for all algorithms using support and confidence
def generate_association_rules(frequent_itemsets: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    """
    Generate association rules from frequent itemsets using the specified confidence.
    """
    if frequent_itemsets.empty:
        print("No frequent itemsets found, skipping association rule generation.")
        return pd.DataFrame()
    # Generate association rules using support and confidence
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: tuple(sorted(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: tuple(sorted(x)))
    rules = rules.sort_values(by=['confidence', 'support'], ascending=[False, False]).reset_index(drop=True)
    rules = rules[['antecedents', 'consequents', 'support', 'confidence']]
    return rules

# Compare results from all algorithms
def compare_results(brute_force_results, apriori_results, fp_growth_results):
    """
    Compare frequent itemsets from all three algorithms.
    """
    brute_force_set = set((itemset, support) for itemset, support in brute_force_results)
    apriori_set = set()
    for _, row in apriori_results.iterrows():
        itemset = row['itemsets']
        support = row['support']
        apriori_set.add((itemset, support))
    
    fp_growth_set = set()
    for _, row in fp_growth_results.iterrows():
        itemset = row['itemsets']
        support = row['support']
        fp_growth_set.add((itemset, support))

    def sets_almost_equal(set1, set2, tol=1e-6):
        if len(set1) != len(set2):
            return False
        for item1 in set1:
            if not any(item1[0] == item2[0] and abs(item1[1] - item2[1]) < tol for item2 in set2):
                return False
        return True

    print("\n--- Comparison of Frequent Itemsets ---")
    print("Brute-Force and Apriori match:", sets_almost_equal(brute_force_set, apriori_set))
    print("Brute-Force and FP-Growth match:", sets_almost_equal(brute_force_set, fp_growth_set))
    print("Apriori and FP-Growth match:", sets_almost_equal(apriori_set, fp_growth_set))

# Compare association rules from all algorithms
def compare_association_rules(brute_rules, apriori_rules, fp_growth_rules):
    """
    Compare association rules from all three algorithms.
    """
    def rules_to_set(rules_df):
        rules_set = set()
        for _, row in rules_df.iterrows():
            antecedents = row['antecedents']
            consequents = row['consequents']
            support = row['support']
            confidence = row['confidence']
            rules_set.add((antecedents, consequents, support, confidence))
        return rules_set

    brute_set = rules_to_set(brute_rules) if not brute_rules.empty else set()
    apriori_set = rules_to_set(apriori_rules) if not apriori_rules.empty else set()
    fp_growth_set = rules_to_set(fp_growth_rules) if not fp_growth_rules.empty else set()

    def rules_almost_equal(set1, set2, tol=1e-6):
        if len(set1) != len(set2):
            return False
        for rule1 in set1:
            if not any(rule1[0] == rule2[0] and rule1[1] == rule2[1] and
                       abs(rule1[2] - rule2[2]) < tol and abs(rule1[3] - rule2[3]) < tol for rule2 in set2):
                return False
        return True

    print("\n--- Comparison of Association Rules ---")
    print("Brute-Force and Apriori rules match:", rules_almost_equal(brute_set, apriori_set))
    print("Brute-Force and FP-Growth rules match:", rules_almost_equal(brute_set, fp_growth_set))
    print("Apriori and FP-Growth rules match:", rules_almost_equal(apriori_set, fp_growth_set))

# Measure performance of a function
def measure_performance(func, *args) -> Tuple:
    """
    Measure the execution time of a given function.
    """
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    return result, end - start

# Validate user input for support and confidence
def get_valid_input(prompt: str, min_value: float = 0.01, max_value: float = 1.0, default_value: float = None) -> float:
    """
    Get valid input from the user for support and confidence values.
    """
    if default_value is not None:
        return default_value
    while True:
        try:
            value = float(input(prompt))
            if min_value <= value <= max_value:
                return value
            else:
                print(f"Error: Input must be between {min_value} and {max_value}. Please try again.")
        except ValueError:
            print("Error: Invalid input. Please enter a numeric value.")

# Main function
def main():
    print("Select dataset (1: Amazon, 2: Best Buy, 3: K-Mart, 4: Nike, 5: Generic): ")
    dataset_choice = input("Enter the number corresponding to your dataset: ")

    transactions = load_dataset(dataset_choice)

    if not transactions:
        return

    # Get user input for support and confidence
    min_support = get_valid_input("Enter minimum support (between 0.01 and 1): ", 0.01, 1.0)
    min_confidence = get_valid_input("Enter minimum confidence (between 0.01 and 1): ", 0.01, 1.0)

    # Run algorithms
    print("\nRunning Brute-Force method...")
    brute_results, brute_time = measure_performance(brute_force_frequent_itemsets, transactions, min_support)
    brute_force_df = pd.DataFrame(brute_results, columns=['itemsets', 'support'])
    brute_force_df['itemsets'] = brute_force_df['itemsets'].apply(lambda x: tuple(x))

    print("\nRunning Apriori method...")
    apriori_results, apriori_time = measure_performance(run_apriori, transactions, min_support)

    print("\nRunning FP-Growth method...")
    fp_growth_results, fp_time = measure_performance(run_fp_growth, transactions, min_support)

    # Generate association rules
    brute_association_rules = generate_association_rules(brute_force_df, min_confidence)
    apriori_association_rules = generate_association_rules(apriori_results, min_confidence)
    fp_growth_association_rules = generate_association_rules(fp_growth_results, min_confidence)

    print("\n--- Association Rules (Brute-Force) ---")
    print_association_rules(brute_association_rules)

    print("\n--- Association Rules (Apriori) ---")
    print_association_rules(apriori_association_rules)

    print("\n--- Association Rules (FP-Growth) ---")
    print_association_rules(fp_growth_association_rules)

    # Comparing the results
    compare_results(brute_results, apriori_results, fp_growth_results)
    compare_association_rules(brute_association_rules, apriori_association_rules, fp_growth_association_rules)

    print(f"\n--- Performance Comparison ---")
    print(f"Brute-Force Time: {brute_time:.4f} seconds")
    print(f"Apriori Time: {apriori_time:.4f} seconds")
    print(f"FP-Growth Time: {fp_time:.4f} seconds")

    print("\n--- Brute-Force Frequent Itemsets: ---")
    print(tabulate(brute_force_df, headers='keys', tablefmt='simple', showindex=False))

    print("\n--- Apriori Frequent Itemsets: ---")
    print(tabulate(apriori_results, headers='keys', tablefmt='simple', showindex=False))

    print("\n--- FP-Growth Frequent Itemsets: ---")
    print(tabulate(fp_growth_results, headers='keys', tablefmt='simple', showindex=False))



if __name__ == "__main__":
    main()
