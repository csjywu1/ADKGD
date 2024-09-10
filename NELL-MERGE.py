import pandas as pd

# Define file paths
train_path = r'C:\Users\Admin\Desktop\multirelational-poincare-master\multirelational-poincare-master\data\NELL-995-h100\train.txt'
test_path = r'C:\Users\Admin\Desktop\multirelational-poincare-master\multirelational-poincare-master\data\NELL-995-h100\test.txt'
valid_path = r'C:\Users\Admin\Desktop\multirelational-poincare-master\multirelational-poincare-master\data\NELL-995-h100\valid.txt'

# Read the files
try:
    train_data = pd.read_csv(train_path, sep='\t', header=None)
    test_data = pd.read_csv(test_path, sep='\t', header=None)
    valid_data = pd.read_csv(valid_path, sep='\t', header=None)
except FileNotFoundError as e:
    print(f"File not found: {e}")
    raise

# Combine all data
all_data = pd.concat([train_data, test_data, valid_data], ignore_index=True)
all_data.columns = ['head', 'relation', 'tail']

# Get unique entities and relations
entities = pd.unique(all_data[['head', 'tail']].values.ravel('K'))
relations = all_data['relation'].unique()

# Calculate mean in-degree
in_degrees = all_data['tail'].value_counts()
mean_in_degree = in_degrees.mean()

# Prepare the statistics
stats = {
    'Total Entities': len(entities),
    'Total Relations': len(relations),
    'Total Triples': len(all_data),
    'Mean In-Degree': mean_in_degree
}

# Create a DataFrame to store results
results = pd.DataFrame()

# Use pd.concat to add new results
results = pd.concat([results, pd.DataFrame([stats])], ignore_index=True)

# Display the results
print(results)
