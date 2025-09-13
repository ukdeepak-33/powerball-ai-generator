import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('powerball.db')
cursor = conn.cursor()

# Query the database
cursor.execute("SELECT * FROM powerball_results LIMIT 5")
rows = cursor.fetchall()

# Print the results
for row in rows:
    print(row)

# Close the connection
conn.close()