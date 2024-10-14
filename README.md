# Data Mining Midterm Project

This project implements three algorithms—**Brute-Force**, **Apriori**, and **FP-Growth**—for discovering frequent itemsets and generating association rules from a transactional dataset. The project compares the performance of these algorithms based on their execution time and the number of rules generated.

## Table of Contents
- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Algorithms Implemented](#algorithms-implemented)
- [Requirements](#requirements)
- [Installation](#installation)

## Project Overview

In this project, I explore the effectiveness and efficiency of three common algorithms for association rule mining in data mining:
- Brute-Force
- Apriori
- FP-Growth

The project includes frequent itemset discovery, support and confidence calculations, association rule generation, and a performance comparison between the algorithms.

## Datasets

This project uses multiple datasets, including:
- `amazon_dataset.csv`
- `best_buy_dataset.csv`
- `kmart_dataset.csv`
- `nike_dataset.csv`
- `generic_dataset.csv`

Each dataset consists of transaction data where each transaction is a list of items purchased together.

## Algorithms Implemented

The following algorithms have been implemented:

1. **Brute-Force Algorithm**  
   The Brute-Force algorithm generates all possible itemsets and calculates support, making it computationally expensive but exhaustive in nature.

2. **Apriori Algorithm**  
   Apriori improves upon Brute-Force by pruning non-frequent itemsets early in the process, reducing the overall computation time.

3. **FP-Growth Algorithm**  
   FP-Growth uses an FP-tree data structure to efficiently mine frequent patterns without generating candidate itemsets explicitly, making it faster than Apriori and Brute-Force.

## Requirements

Before running the project, ensure you have the following installed:
- Python 3.x
- pip (Python package manager)

The required Python libraries can be installed via the `requirements.txt` file.

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/YourGitHubUsername/salunkhe-aniket-data-mining-midtermProj.git
  
2. Navigate to the project directory:

   ```bash
   cd salunkhe-aniket-data-mining-midtermProj
   
3. Install dependencies:

   ```bash
   pip install -r requirements.txt

4. Run script:
   
   ```bash
   python app.py

