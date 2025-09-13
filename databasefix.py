import pandas as pd

# Load the historical data
file_path = '/Users/bunny/Powerball_App/powerball_results_02.tsv'
df = pd.read_csv(file_path, sep='\t')

# Function to sort white ball numbers in ascending order
def sort_white_balls(row):
    white_balls = [row['Number 1'], row['Number 2'], row['Number 3'], row['Number 4'], row['Number 5']]
    white_balls_sorted = sorted(white_balls)  # Sort in ascending order
    return pd.Series(white_balls_sorted)

# Apply the sorting function to each row
df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']] = df.apply(sort_white_balls, axis=1)

# Save the updated dataframe back to the TSV file
df.to_csv(file_path, sep='\t', index=False)

print("White ball numbers have been sorted in ascending order and the database has been updated.")