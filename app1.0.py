from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import random
from itertools import combinations
import math
import os
from collections import defaultdict

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load historical data
def load_historical_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist. Please provide a valid file path.")
    df = pd.read_csv(file_path, sep='\t')
    # Convert 'Draw Date' to datetime and then format to YYYY-MM-DD string
    df['Draw Date'] = pd.to_datetime(df['Draw Date']).dt.strftime('%Y-%m-%d')
    return df

# Get the last draw result
def get_last_draw(df):
    return df.iloc[-1]

# Check if the generated numbers match any historical draw
def check_exact_match(df, white_balls):
    for _, row in df.iterrows():
        historical_white_balls = row[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values
        if set(white_balls) == set(historical_white_balls):
            return True
    return False

# Generate random numbers for Powerball
def generate_powerball_numbers(df, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, high_low_balance=None):
    while True:
        # Generate white balls (5 numbers from the specified range, excluding excluded numbers)
        available_numbers = [num for num in range(white_ball_range[0], white_ball_range[1] + 1) if num not in excluded_numbers]
        
        # Ensure there are enough available numbers to pick from
        if len(available_numbers) < 5:
            # Fallback or raise an error if not enough numbers are available
            # For this example, we'll just continue trying, but in a real app, you might want to handle this more robustly
            print("Not enough available numbers for white balls after exclusions and range constraints.")
            continue
            
        white_balls = random.sample(available_numbers, 5)

        # Ensure two numbers are from Group A
        group_a_numbers = [num for num in white_balls if num in group_a]
        if len(group_a_numbers) < 2:
            continue

        # Generate Powerball (1 number from the specified range)
        powerball = random.randint(powerball_range[0], powerball_range[1])

        # Ensure the generated numbers do not match the last draw
        last_white_balls = df.iloc[-1][['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values
        # Compare sets of white balls, convert powerball to int just in case
        if set(white_balls) == set(last_white_balls) and powerball == int(df.iloc[-1]['Powerball']):
            continue

        # Ensure the generated numbers do not exactly match any previous 5 main numbers
        if check_exact_match(df, white_balls):
            continue

        # Check odd/even condition
        even_count = sum(1 for num in white_balls if num % 2 == 0)
        odd_count = 5 - even_count

        if odd_even_choice == "All Even" and even_count != 5:
            continue
        elif odd_even_choice == "All Odd" and odd_count != 5:
            continue
        elif odd_even_choice == "3 Even / 2 Odd" and (even_count != 3 or odd_count != 2):
            continue
        elif odd_even_choice == "3 Odd / 2 Even" and (odd_count != 3 or even_count != 2):
            continue
        elif odd_even_choice == "1 Even / 4 Odd" and (even_count != 1 or odd_count != 4):
            continue
        elif odd_even_choice == "1 Odd / 4 Even" and (odd_count != 1 or even_count != 4):
            continue

        # Combo condition is currently not implemented for filtering generation, only for display purposes in the original code.
        # If filtering generation by combo is desired, additional logic would be needed here.
        
        # Check high/low balance condition
        if high_low_balance is not None and len(high_low_balance) == 2:
            low_numbers = [num for num in white_balls if num <= 34]
            high_numbers = [num for num in white_balls if num >= 35]
            if len(low_numbers) < high_low_balance[0] or len(high_numbers) < high_low_balance[1]:
                continue
        elif high_low_balance is not None and len(high_low_balance) != 2:
            print("Warning: high_low_balance must contain exactly two numbers (e.g., '2 3'). Ignoring invalid input.")


        break

    return white_balls, powerball

# Check if generated numbers match historical data
def check_historical_match(df, white_balls, powerball):
    for _, row in df.iterrows():
        historical_white_balls = row[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values
        historical_powerball = row['Powerball']
        if set(white_balls) == set(historical_white_balls) and powerball == historical_powerball:
            return row['Draw Date'] # Draw Date is already a string YYYY-MM-DD
    return None

# Frequency analysis of white balls and Powerball
def frequency_analysis(df):
    # Frequency of white balls (1–69)
    white_balls = df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.flatten()
    white_ball_freq = pd.Series(white_balls).value_counts().reindex(range(1, 70), fill_value=0)

    # Frequency of Powerball (1–26)
    powerball_freq = df['Powerball'].value_counts().reindex(range(1, 27), fill_value=0)

    return white_ball_freq, powerball_freq

# Hot and cold numbers analysis
def hot_cold_numbers(df, last_draw_date_str):
    # Convert last_draw_date_str to datetime object for comparison
    last_draw_date = pd.to_datetime(last_draw_date_str)
    # Filter data for the last year
    one_year_ago = last_draw_date - pd.DateOffset(years=1)
    
    # Ensure 'Draw Date' is datetime for comparison, then filter
    df['Draw Date_dt'] = pd.to_datetime(df['Draw Date'])
    recent_data = df[df['Draw Date_dt'] >= one_year_ago]
    
    # Frequency of white balls in the last year
    white_balls = recent_data[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.flatten()
    white_ball_freq = pd.Series(white_balls).value_counts()

    # Hot numbers (14 most frequent) - sort by frequency descending
    hot_numbers = white_ball_freq.nlargest(14).sort_values(ascending=False)

    # Cold numbers (14 least frequent) - sort by frequency ascending
    cold_numbers = white_ball_freq.nsmallest(14).sort_values(ascending=True)

    return hot_numbers, cold_numbers

# Monthly white ball analysis
def monthly_white_ball_analysis(df, last_draw_date_str):
    # Convert last_draw_date_str to datetime object for comparison
    last_draw_date = pd.to_datetime(last_draw_date_str)
    # Filter data for the last 6 months
    six_months_ago = last_draw_date - pd.DateOffset(months=6)
    
    # Ensure 'Draw Date' is datetime for comparison, then filter
    df['Draw Date_dt'] = pd.to_datetime(df['Draw Date'])
    recent_data = df[df['Draw Date_dt'] >= six_months_ago].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Group by month and get the white balls
    recent_data['Month'] = recent_data['Draw Date_dt'].dt.to_period('M')
    
    # Aggregate white balls by month, flatten and convert to list of lists for display
    monthly_balls = recent_data.groupby('Month')[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].apply(
        lambda x: sorted(list(set(x.values.flatten()))) # Get unique sorted balls for each month
    ).to_dict()
    
    # Convert Period objects to string for dictionary keys
    monthly_balls_str_keys = {str(k): v for k, v in monthly_balls.items()}

    return monthly_balls_str_keys

# Sum of main balls analysis
def sum_of_main_balls(df):
    # Calculate the sum of the 5 white balls for each draw
    # Ensure 'Sum' column is calculated only once or re-calculated if data changes
    temp_df = df.copy() # Work on a copy to avoid modifying original df unintentionally
    temp_df['Sum'] = temp_df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].sum(axis=1)
    return temp_df[['Draw Date', 'Sum']]

# Find past results with a specific sum of white balls
def find_results_by_sum(df, target_sum):
    # Calculate the sum of the 5 white balls for each draw if 'Sum' column does not exist
    temp_df = df.copy() # Work on a copy
    if 'Sum' not in temp_df.columns:
        temp_df['Sum'] = temp_df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].sum(axis=1)
    
    # Filter rows where the sum matches the target sum
    results = temp_df[temp_df['Sum'] == target_sum]
    return results[['Draw Date', 'Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Sum']]

# Save generated numbers to a file (kept as is, not exposed via web route directly)
def save_generated_numbers(white_balls, powerball, file_path="generated_numbers.txt"):
    with open(file_path, "a") as file:
        file.write(f"White Balls: {white_balls}, Powerball: {powerball}\n")
    print(f"Numbers saved to {file_path}")

# Simulate multiple draws (kept as is, output to console/not directly to template)
def simulate_multiple_draws(df, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, num_draws=100):
    results = []
    for _ in range(num_draws):
        white_balls, powerball = generate_powerball_numbers(df, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers)
        results.append(white_balls + [powerball])
    
    # Flatten the results and count frequencies
    all_numbers = [num for draw in results for num in draw]
    freq = pd.Series(all_numbers).value_counts().sort_index()
    
    print(f"Frequency of Numbers in {num_draws} Draws:")
    print(freq)
    return freq # Return frequency for potential display later if needed

# Function to calculate combinations (n choose k)
def calculate_combinations(n, k):
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)

# Function to calculate winning probability
def winning_probability(white_ball_range, powerball_range):
    # Calculate total white ball combinations
    total_white_balls_in_range = white_ball_range[1] - white_ball_range[0] + 1
    white_ball_combinations = calculate_combinations(total_white_balls_in_range, 5)

    # Calculate total Powerball numbers
    total_powerballs_in_range = powerball_range[1] - powerball_range[0] + 1

    # Calculate total combinations
    total_combinations = white_ball_combinations * total_powerballs_in_range

    # Calculate probability as "1 in X" and as a percentage
    probability_1_in_x = f"1 in {total_combinations:,}" if total_combinations > 0 else "N/A"
    probability_percentage = f"{1 / total_combinations * 100:.10f}%" if total_combinations > 0 else "N/A"

    return probability_1_in_x, probability_percentage

# Function to calculate partial match probabilities
def partial_match_probabilities(white_ball_range, powerball_range):
    total_white_balls_in_range = white_ball_range[1] - white_ball_range[0] + 1
    total_powerballs_in_range = powerball_range[1] - powerball_range[0] + 1

    # Total possible Powerball combinations (denominator for odds)
    total_powerball_comb = calculate_combinations(total_white_balls_in_range, 5) * total_powerballs_in_range

    probabilities = {}

    # Match 5 White Balls + Powerball
    if total_powerball_comb > 0:
        probabilities["Match 5 White Balls + Powerball"] = f"{total_powerball_comb / (calculate_combinations(5, 5) * calculate_combinations(total_white_balls_in_range - 5, 0) * calculate_combinations(total_powerballs_in_range, 1)):,.0f} to 1"
    else:
        probabilities["Match 5 White Balls + Powerball"] = "N/A"

    # Match 5 White Balls only
    if total_powerball_comb > 0:
        probabilities["Match 5 White Balls Only"] = f"{total_powerball_comb / (calculate_combinations(5, 5) * calculate_combinations(total_white_balls_in_range - 5, 0) * calculate_combinations(total_powerballs_in_range - 1, 1)):,.0f} to 1"
    else:
        probabilities["Match 5 White Balls Only"] = "N/A"

    # Match 4 White Balls + Powerball
    if total_powerball_comb > 0:
        probabilities["Match 4 White Balls + Powerball"] = f"{total_powerball_comb / (calculate_combinations(5, 4) * calculate_combinations(total_white_balls_in_range - 5, 1) * calculate_combinations(total_powerballs_in_range, 1)):,.0f} to 1"
    else:
        probabilities["Match 4 White Balls + Powerball"] = "N/A"

    # Match 4 White Balls Only
    if total_powerball_comb > 0:
        probabilities["Match 4 White Balls Only"] = f"{total_powerball_comb / (calculate_combinations(5, 4) * calculate_combinations(total_white_balls_in_range - 5, 1) * calculate_combinations(total_powerballs_in_range - 1, 1)):,.0f} to 1"
    else:
        probabilities["Match 4 White Balls Only"] = "N/A"

    # Match 3 White Balls + Powerball
    if total_powerball_comb > 0:
        probabilities["Match 3 White Balls + Powerball"] = f"{total_powerball_comb / (calculate_combinations(5, 3) * calculate_combinations(total_white_balls_in_range - 5, 2) * calculate_combinations(total_powerballs_in_range, 1)):,.0f} to 1"
    else:
        probabilities["Match 3 White Balls + Powerball"] = "N/A"

    # Match 3 White Balls Only
    if total_powerball_comb > 0:
        probabilities["Match 3 White Balls Only"] = f"{total_powerball_comb / (calculate_combinations(5, 3) * calculate_combinations(total_white_balls_in_range - 5, 2) * calculate_combinations(total_powerballs_in_range - 1, 1)):,.0f} to 1"
    else:
        probabilities["Match 3 White Balls Only"] = "N/A"
    
    # Match 2 White Balls + Powerball (add this tier as it's common)
    if total_powerball_comb > 0:
        probabilities["Match 2 White Balls + Powerball"] = f"{total_powerball_comb / (calculate_combinations(5, 2) * calculate_combinations(total_white_balls_in_range - 5, 3) * calculate_combinations(total_powerballs_in_range, 1)):,.0f} to 1"
    else:
        probabilities["Match 2 White Balls + Powerball"] = "N/A"

    # Match 1 White Ball + Powerball (add this tier as it's common)
    if total_powerball_comb > 0:
        probabilities["Match 1 White Ball + Powerball"] = f"{total_powerball_comb / (calculate_combinations(5, 1) * calculate_combinations(total_white_balls_in_range - 5, 4) * calculate_combinations(total_powerballs_in_range, 1)):,.0f} to 1"
    else:
        probabilities["Match 1 White Ball + Powerball"] = "N/A"

    # Match Powerball Only (add this tier)
    if total_powerball_comb > 0:
        probabilities["Match Powerball Only"] = f"{total_powerball_comb / (calculate_combinations(5, 0) * calculate_combinations(total_white_balls_in_range - 5, 5) * calculate_combinations(total_powerballs_in_range, 1)):,.0f} to 1"
    else:
        probabilities["Match Powerball Only"] = "N/A"


    return probabilities


# Export analysis results to CSV (kept as is)
def export_analysis_results(df, file_path="analysis_results.csv"):
    # Ensure 'Draw Date' is formatted correctly before export if needed,
    # though it should already be string YYYY-MM-DD from load_historical_data
    df.to_csv(file_path, index=False)
    print(f"Analysis results saved to {file_path}")

# Find last draw dates for individual numbers
def find_last_draw_dates_for_numbers(df, white_balls, powerball):
    """
    Find the last draw date for each individual number (white balls and Powerball).
    Returns a dictionary with the last draw date for each number.
    """
    last_draw_dates = {}

    # Create a temporary DataFrame with Draw Date as datetime for easier sorting if not already
    temp_df = df.copy()
    temp_df['Draw Date_dt'] = pd.to_datetime(temp_df['Draw Date'])
    
    # Sort by date descending to find the most recent draw quickly
    sorted_df = temp_df.sort_values(by='Draw Date_dt', ascending=False)

    # Check last draw date for each white ball
    for number in white_balls:
        # Iterate through the historical data in reverse order (most recent first)
        for _, row in sorted_df.iterrows():
            historical_white_balls = row[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values
            if number in historical_white_balls:
                last_draw_dates[f"White Ball {number}"] = row['Draw Date'] # Use the formatted string date
                break  # Stop searching after finding the most recent draw date

    # Check last draw date for the Powerball
    for _, row in sorted_df.iterrows():
        if powerball == row['Powerball']:
            last_draw_dates[f"Powerball {powerball}"] = row['Draw Date'] # Use the formatted string date
            break  # Stop searching after finding the most recent draw date

    return last_draw_dates

# Modify 3 out of 5 white balls and the Powerball
def modify_combination(df, white_balls, powerball, white_ball_range, powerball_range, excluded_numbers):
    """
    Modify 3 out of 5 white balls and the Powerball in a previously drawn combination.
    """
    # Ensure white_balls is a mutable list
    white_balls = list(white_balls)

    # Randomly select 3 indices to modify
    indices_to_modify = random.sample(range(5), 3)
    
    # Generate new numbers for the selected indices
    for i in indices_to_modify:
        while True:
            new_number = random.randint(white_ball_range[0], white_ball_range[1])
            # Ensure new number is not in excluded list and is unique within the current white_balls set
            if new_number not in excluded_numbers and new_number not in white_balls:
                white_balls[i] = new_number
                break
    
    # Generate a new Powerball
    while True:
        new_powerball = random.randint(powerball_range[0], powerball_range[1])
        # Ensure new powerball is not in excluded list and is different from the original powerball
        if new_powerball not in excluded_numbers and new_powerball != powerball:
            powerball = new_powerball
            break
    
    # Convert all numbers to native Python integers
    white_balls = [int(num) for num in white_balls]
    powerball = int(powerball)
    
    return white_balls, powerball

# Find common pairs of Number 1 and Number 2
def find_common_pairs(df, top_n=10):
    """
    Find the most common pairs of Number 1 and Number 2 in the historical data.
    """
    pair_count = defaultdict(int)
    
    for _, row in df.iterrows():
        # Ensure numbers are sorted to treat (1,2) and (2,1) as the same pair
        nums = sorted([row['Number 1'], row['Number 2']])
        pair_count[tuple(nums)] += 1
    
    # Sort pairs by frequency and return the top N
    sorted_pairs = sorted(pair_count.items(), key=lambda x: x[1], reverse=True)
    return [pair[0] for pair in sorted_pairs[:top_n]]

# Filter common pairs by range
def filter_common_pairs_by_range(common_pairs, num_range):
    """
    Filter common pairs to only include pairs where both numbers fall within the specified range.
    """
    filtered_pairs = []
    if not num_range or len(num_range) != 2:
        print("Warning: Invalid num_range for filtering common pairs. Expected (min, max). Returning all common pairs.")
        return common_pairs # Return all if range is invalid or not provided
        
    min_val, max_val = num_range
    for pair in common_pairs:
        if min_val <= pair[0] <= max_val and min_val <= pair[1] <= max_val:
            filtered_pairs.append(pair)
    return filtered_pairs

# Generate numbers using common pairs
def generate_with_common_pairs(df, common_pairs, white_ball_range, excluded_numbers):
    """
    Generate a combination using a common pair as the first two numbers.
    """
    if not common_pairs:
        # Fallback if no common pairs are found after filtering
        print("No common pairs available after filtering. Generating random white balls.")
        available_numbers = [num for num in range(white_ball_range[0], white_ball_range[1] + 1) if num not in excluded_numbers]
        if len(available_numbers) < 5: # Not enough numbers for 5 white balls
             raise ValueError("Not enough numbers to generate 5 white balls after exclusions.")
        return random.sample(available_numbers, 5)


    # Randomly select a common pair
    num1, num2 = random.choice(common_pairs)
    
    # Generate the remaining three numbers
    # Ensure remaining numbers are not in excluded_numbers and are not num1 or num2
    available_numbers = [num for num in range(white_ball_range[0], white_ball_range[1] + 1) 
                         if num not in excluded_numbers and num not in [num1, num2]]
    
    if len(available_numbers) < 3:
        # Not enough numbers to pick 3 unique remaining numbers
        # This case is tricky and might need more sophisticated handling in a real app
        # For simplicity, if this happens, we'll try to generate a fully random set
        print(f"Warning: Not enough unique numbers to complete the combination with common pair ({num1}, {num2}). Falling back to full random generation.")
        return random.sample([n for n in range(white_ball_range[0], white_ball_range[1] + 1) if n not in excluded_numbers], 5)

    remaining_numbers = random.sample(available_numbers, 3)
    
    # Combine the numbers and sort them
    white_balls = sorted([num1, num2] + remaining_numbers)
    return white_balls

# Default file path
# IMPORTANT: Update this path to where your powerball_results_02.tsv file is located
file_path = '/Users/bunny/Powerball_App/powerball_results_02.tsv'

# Load historical data
try:
    df = load_historical_data(file_path)
except FileNotFoundError as e:
    print(e)
    # If the file is not found, exit or handle gracefully, e.g., serving a page with an error message
    # For a Flask app, it's better to show an error page or message to the user
    # For now, we'll let the app potentially crash if run directly without the file,
    # but in a production Flask app, you'd handle this more robustly.
    exit()

# Get the last draw result (after loading df)
last_draw = get_last_draw(df)

# Group A numbers
group_a = [3, 5, 6, 7, 9, 11, 15, 16, 18, 21, 23, 24, 27, 31, 32, 33, 36, 42, 44, 45, 48, 50, 51, 54, 55, 60, 66, 69]

# Default ranges and excluded numbers
white_ball_range = (1, 69)
powerball_range = (1, 26)
excluded_numbers = []


@app.route('/')
def index():
    last_draw = get_last_draw(df)
    last_draw_dict = last_draw.to_dict()
    # Format the 'Draw Date' for the last_draw_dict if it's not already
    last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')
    return render_template('index.html', last_draw=last_draw_dict)

@app.route('/generate', methods=['POST'])
def generate():
    odd_even_choice = request.form.get('odd_even_choice', 'Any') # Default to 'Any'
    combo_choice = request.form.get('combo_choice', 'No Combo')
    white_ball_min = int(request.form.get('white_ball_min', 1))
    white_ball_max = int(request.form.get('white_ball_max', 69))
    white_ball_range = (white_ball_min, white_ball_max)
    powerball_min = int(request.form.get('powerball_min', 1))
    powerball_max = int(request.form.get('powerball_max', 26))
    powerball_range = (powerball_min, powerball_max)
    excluded_numbers = [int(num.strip()) for num in request.form.get('excluded_numbers', '').split(",") if num.strip().isdigit()] if request.form.get('excluded_numbers') else []
    
    high_low_balance_str = request.form.get('high_low_balance', '')
    high_low_balance = None
    if high_low_balance_str:
        try:
            parts = [int(num.strip()) for num in high_low_balance_str.split() if num.strip().isdigit()]
            if len(parts) == 2:
                high_low_balance = tuple(parts)
            else:
                flash("High/Low Balance input must be two numbers separated by space (e.g., '2 3').", 'error')
        except ValueError:
            flash("Invalid High/Low Balance format. Please enter numbers separated by space.", 'error')


    # Generate the numbers
    try:
        white_balls, powerball = generate_powerball_numbers(df, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, high_low_balance)
        # Find the last draw date for each individual number
        last_draw_dates = find_last_draw_dates_for_numbers(df, white_balls, powerball)
    except ValueError as e:
        flash(str(e), 'error')
        return redirect(url_for('index'))

    # Fetch the last draw result and convert it to a dictionary
    last_draw = get_last_draw(df)
    last_draw_dict = last_draw.to_dict()
    last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')


    # Render the template with the generated numbers and last draw dates
    return render_template('index.html', 
                           white_balls=white_balls, 
                           powerball=powerball, 
                           last_draw=last_draw_dict, 
                           last_draw_dates=last_draw_dates)

@app.route('/generate_modified', methods=['POST'])
def generate_modified():
    # Get filter parameters from the form
    use_common_pairs = request.form.get('use_common_pairs') == 'on'
    num_range_str = request.form.get('num_range', '')
    num_range = None
    if num_range_str:
        try:
            parts = [int(num.strip()) for num in num_range_str.split() if num.strip().isdigit()]
            if len(parts) == 2:
                num_range = tuple(parts)
            else:
                flash("Filter Common Pairs by Range input must be two numbers separated by space (e.g., '1 20').", 'error')
        except ValueError:
            flash("Invalid Filter Common Pairs by Range format. Please enter numbers separated by space.", 'error')

    # Randomly select a previously drawn combination
    random_row = df.sample(1).iloc[0]
    white_balls = random_row[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.tolist()
    powerball = random_row['Powerball']

    # Modify the combination
    try:
        if use_common_pairs:
            common_pairs = find_common_pairs(df, top_n=20)  # Adjust top_n as needed
            if num_range:
                common_pairs = filter_common_pairs_by_range(common_pairs, num_range)
            
            if not common_pairs:
                flash("No common pairs found with the specified filter. Generating a random combination instead.", 'info')
                white_balls, powerball = generate_powerball_numbers(df, group_a, "Any", "No Combo", white_ball_range, powerball_range, excluded_numbers)
            else:
                white_balls = generate_with_common_pairs(df, common_pairs, white_ball_range, excluded_numbers)
                # Regenerate Powerball as it's not handled by generate_with_common_pairs
                powerball = random.randint(powerball_range[0], powerball_range[1])
        else:
            white_balls, powerball = modify_combination(df, white_balls, powerball, white_ball_range, powerball_range, excluded_numbers)
            
        # Ensure the modified combination has never been drawn before
        max_attempts = 100 # Add a safety break to prevent infinite loops
        attempts = 0
        while check_exact_match(df, white_balls) and attempts < max_attempts:
            if use_common_pairs:
                common_pairs_recheck = find_common_pairs(df, top_n=20)
                if num_range:
                    common_pairs_recheck = filter_common_pairs_by_range(common_pairs_recheck, num_range)
                if common_pairs_recheck:
                    white_balls = generate_with_common_pairs(df, common_pairs_recheck, white_ball_range, excluded_numbers)
                else:
                    # Fallback if no common pairs can be generated again
                    white_balls, powerball = generate_powerball_numbers(df, group_a, "Any", "No Combo", white_ball_range, powerball_range, excluded_numbers)
            else:
                # Need to get new random row to modify, otherwise it might get stuck
                random_row = df.sample(1).iloc[0]
                white_balls_base = random_row[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.tolist()
                powerball_base = random_row['Powerball']
                white_balls, powerball = modify_combination(df, white_balls_base, powerball_base, white_ball_range, powerball_range, excluded_numbers)
            attempts += 1
        
        if attempts == max_attempts:
            flash("Could not find a unique modified combination after many attempts. Please try again.", 'error')
            return redirect(url_for('index'))

        # Convert all numbers to native Python integers
        white_balls = [int(num) for num in white_balls]
        powerball = int(powerball)

        # Find the last draw date for each individual number
        last_draw_dates = find_last_draw_dates_for_numbers(df, white_balls, powerball)

        # Fetch the last draw result and convert it to a dictionary
        last_draw = get_last_draw(df)
        last_draw_dict = last_draw.to_dict()
        last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')


        # Render the template with the generated numbers and last draw dates
        return render_template('index.html', 
                            white_balls=white_balls, 
                            powerball=powerball, 
                            last_draw=last_draw_dict, 
                            last_draw_dates=last_draw_dates)
    except ValueError as e:
        flash(str(e), 'error')
        return redirect(url_for('index'))


@app.route('/frequency_analysis')
def frequency_analysis_route():
    white_ball_freq, powerball_freq = frequency_analysis(df)
    last_draw = get_last_draw(df)  # Fetch the last draw result
    last_draw_dict = last_draw.to_dict()
    last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')

    # Convert Series to dictionary and then to list of dictionaries for easier Jinja2 iteration
    white_ball_freq_list = [{'Number': int(k), 'Frequency': int(v)} for k, v in white_ball_freq.items()]
    powerball_freq_list = [{'Number': int(k), 'Frequency': int(v)} for k, v in powerball_freq.items()]

    return render_template('index.html', 
                           white_ball_freq=white_ball_freq_list, 
                           powerball_freq=powerball_freq_list, 
                           last_draw=last_draw_dict)

@app.route('/hot_cold_numbers')
def hot_cold_numbers_route():
    last_draw_date_str = last_draw['Draw Date'] # Already formatted as string
    hot_numbers, cold_numbers = hot_cold_numbers(df, last_draw_date_str)
    
    last_draw_dict = last_draw.to_dict()
    last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')

    # Convert Series to list of dictionaries
    hot_numbers_list = [{'Number': int(k), 'Frequency': int(v)} for k, v in hot_numbers.items()]
    cold_numbers_list = [{'Number': int(k), 'Frequency': int(v)} for k, v in cold_numbers.items()]

    return render_template('index.html', 
                           hot_numbers=hot_numbers_list, 
                           cold_numbers=cold_numbers_list, 
                           last_draw=last_draw_dict)

@app.route('/monthly_white_ball_analysis')
def monthly_white_ball_analysis_route():
    last_draw_date_str = last_draw['Draw Date'] # Already formatted as string
    monthly_balls = monthly_white_ball_analysis(df, last_draw_date_str)
    
    last_draw_dict = last_draw.to_dict()
    last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')

    # monthly_balls is already a dictionary with string keys and list values
    return render_template('index.html', 
                           monthly_balls=monthly_balls, 
                           last_draw=last_draw_dict)

@app.route('/sum_of_main_balls')
def sum_of_main_balls_route():
    sum_data = sum_of_main_balls(df)
    last_draw_dict = last_draw.to_dict()
    last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')

    # Ensure 'Draw Date' in sum_data is formatted as string before passing to_dict('records')
    sum_data_list = sum_data.copy()
    # No need to convert here if load_historical_data already formats it
    # sum_data_list['Draw Date'] = pd.to_datetime(sum_data_list['Draw Date']).dt.strftime('%Y-%m-%d')
    
    return render_template('index.html', 
                           sum_data=sum_data_list.to_dict('records'), 
                           last_draw=last_draw_dict)

@app.route('/find_results_by_sum', methods=['POST'])
def find_results_by_sum_route():
    target_sum_str = request.form.get('target_sum')
    results = []
    target_sum_display = None

    if target_sum_str and target_sum_str.isdigit():
        target_sum = int(target_sum_str)
        target_sum_display = target_sum # Keep for display in template
        results_df = find_results_by_sum(df, target_sum)
        # Ensure 'Draw Date' in results_df is formatted as string before passing
        # results_df['Draw Date'] = pd.to_datetime(results_df['Draw Date']).dt.strftime('%Y-%m-%d') # Already done by load_historical_data
        results = results_df.to_dict('records')
    else:
        flash("Please enter a valid number for Target Sum.", 'error')

    last_draw_dict = last_draw.to_dict()
    last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')


    return render_template('index.html', 
                           results=results, # results here is the list of dicts
                           last_draw=last_draw_dict,
                           target_sum=target_sum_display) # Pass the target sum back for display

@app.route('/simulate_multiple_draws', methods=['POST'])
def simulate_multiple_draws_route():
    num_draws_str = request.form.get('num_draws')
    if num_draws_str and num_draws_str.isdigit():
        num_draws = int(num_draws_str)
        simulated_freq = simulate_multiple_draws(df, group_a, "Any", "No Combo", white_ball_range, powerball_range, excluded_numbers, num_draws)
        
        last_draw_dict = last_draw.to_dict()
        last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')

        # Convert simulated_freq Series to list of dicts for rendering
        simulated_freq_list = [{'Number': int(k), 'Frequency': int(v)} for k, v in simulated_freq.items()]

        return render_template('index.html', 
                               simulated_freq=simulated_freq_list, 
                               num_simulations=num_draws,
                               last_draw=last_draw_dict)
    else:
        flash("Please enter a valid number for Number of Simulations.", 'error')
        return redirect(url_for('index'))


@app.route('/winning_probability')
def winning_probability_route():
    probability_1_in_x, probability_percentage = winning_probability(white_ball_range, powerball_range)
    last_draw_dict = last_draw.to_dict()
    last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')

    return render_template('index.html', 
                           probability_1_in_x=probability_1_in_x, 
                           probability_percentage=probability_percentage, 
                           last_draw=last_draw_dict)

@app.route('/partial_match_probabilities')
def partial_match_probabilities_route():
    probabilities = partial_match_probabilities(white_ball_range, powerball_range)
    last_draw_dict = last_draw.to_dict()
    last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')

    return render_template('index.html', 
                           probabilities=probabilities, 
                           last_draw=last_draw_dict)

@app.route('/export_analysis_results')
def export_analysis_results_route():
    # Calling export_analysis_results with the DataFrame will save the current state of df.
    # If specific analysis results are meant to be exported, they should be passed to this function.
    # For now, it exports the entire loaded historical data.
    export_analysis_results(df) 
    flash("Analysis results exported to analysis_results.csv", 'info')
    return redirect(url_for('index'))

# New route for finding results by first white ball number
@app.route('/find_results_by_first_white_ball', methods=['POST'])
def find_results_by_first_white_ball():
    white_ball_number_str = request.form.get('white_ball_number')
    results_dict = []
    white_ball_number_display = None
    sort_by_year_flag = request.form.get('sort_by_year') == 'on'

    if white_ball_number_str and white_ball_number_str.isdigit():
        white_ball_number = int(white_ball_number_str)
        white_ball_number_display = white_ball_number # Keep for display in template
        
        # Filter the dataframe to find rows where the first white ball number matches the entered number
        results = df[df['Number 1'] == white_ball_number].copy() # Work on a copy

        # Sort the results by year (extract year from 'Draw Date' and sort)
        if sort_by_year_flag:
            # Draw Date is already YYYY-MM-DD string, convert to datetime for year extraction and sorting
            results['Year'] = pd.to_datetime(results['Draw Date']).dt.year
            results = results.sort_values(by='Year')
        
        # Convert the results to a dictionary for rendering in the template
        results_dict = results.to_dict('records')
    else:
        flash("Please enter a valid number for First White Ball Number.", 'error')

    # Fetch the last draw result and convert it to a dictionary
    last_draw = get_last_draw(df)
    last_draw_dict = last_draw.to_dict()
    last_draw_dict['Draw Date'] = pd.to_datetime(last_draw_dict['Draw Date']).strftime('%Y-%m-%d')

    
    # Render the template with the results and last draw
    return render_template('index.html', 
                           results_by_first_white_ball=results_dict, 
                           last_draw=last_draw_dict,
                           white_ball_number=white_ball_number_display, # Pass back for display
                           sort_by_year=sort_by_year_flag) # Pass back for checkbox state

if __name__ == '__main__':
    app.run(debug=True)

