import pandas as pd
import sqlite3

# Load the historical data from the TSV file
file_path = '/Users/bunny/Powerball_App/powerball_results_02.tsv'
df = pd.read_csv(file_path, sep='\t')

# Convert the 'Draw Date' column to datetime
df['Draw Date'] = pd.to_datetime(df['Draw Date'])

# Create an SQLite database and load the data
db_path = 'powerball.db'  # SQLite database file
conn = sqlite3.connect(db_path)

# Save the DataFrame to the SQLite database
df.to_sql('powerball_results', conn, if_exists='replace', index=False)

# Close the connection
conn.close()

print("Historical data has been loaded into SQLite database.")