from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import random
from itertools import combinations
import math
import os
from collections import defaultdict, Counter
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Default file path (adjust as necessary)
# It's highly recommended to place 'powerball_results_02.tsv' in the same directory as app.py
# or provide an absolute path if it's elsewhere.
file_path = 'powerball_results_02.tsv'

# Load historical data
def load_historical_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist. Please provide a valid file path.")
    df = pd.read_csv(file_path, sep='\t')
    df['Draw Date'] = pd.to_datetime(df['Draw Date'])
    
    # Calculate sum of white balls if not already present
    if 'Sum' not in df.columns:
        df['Sum'] = df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].sum(axis=1)
        
    return df

# Get the last draw result
def get_last_draw(df):
    if not df.empty:
        return df.iloc[-1]
    return pd.Series() # Return an empty Series if DataFrame is empty

# Global DataFrame and last draw
df = pd.DataFrame() # Initialize as empty to prevent errors if file not found
last_draw = pd.Series() # Initialize as empty Series

# Initialize data on app startup
try:
    df = load_historical_data(file_path)
    last_draw = get_last_draw(df)
except FileNotFoundError as e:
    print(f"Error loading historical data: {e}")
    flash(f"Error: Historical data file not found at '{file_path}'. Please ensure the file exists and the path is correct.", 'error')
except Exception as e:
    print(f"An unexpected error occurred while loading data: {e}")
    flash(f"An unexpected error occurred during data loading: {e}", 'error')


# Pre-calculate historical combinations for quick lookup
def get_historical_combinations_set(df, include_powerball=True):
    combinations_set = set()
    if df.empty:
        return combinations_set
    for _, row in df.iterrows():
        white_balls = sorted(row[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].tolist())
        if include_powerball:
            combo_str = f"{','.join(map(str, white_balls))}|{row['Powerball']}"
        else:
            combo_str = f"{','.join(map(str, white_balls))}"
        combinations_set.add(combo_str)
    return combinations_set

# Only attempt to populate these if df is not empty
if not df.empty:
    historical_white_ball_pb_combos = get_historical_combinations_set(df, include_powerball=True)
    historical_white_ball_combos = get_historical_combinations_set(df, include_powerball=False)
else:
    historical_white_ball_pb_combos = set()
    historical_white_ball_combos = set()

# Check if the generated numbers match any historical draw
def check_exact_match(white_balls, powerball, historical_combos_set):
    sorted_white_balls = sorted(white_balls)
    combo_str = f"{','.join(map(str, sorted_white_balls))}|{powerball}"
    return combo_str in historical_combos_set

# Check if the generated numbers (white balls only) match any historical draw
def check_exact_match_white_balls_only(white_balls, historical_white_ball_combos_set):
    sorted_white_balls = sorted(white_balls)
    combo_str = f"{','.join(map(str, sorted_white_balls))}"
    return combo_str in historical_white_ball_combos_set

# Generate random numbers for Powerball
def generate_powerball_numbers(df, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, high_low_balance=None, strategy="random", avoid_repeats_years=0):
    
    if df.empty:
        flash("Historical data not loaded, cannot generate numbers with complex rules.", 'error')
        return None, None

    # Filter historical combinations based on avoid_repeats_years
    recent_historical_white_ball_pb_combos = set()
    if avoid_repeats_years > 0:
        threshold_date = df['Draw Date'].max() - pd.DateOffset(years=avoid_repeats_years)
        recent_historical_combos_df = df[df['Draw Date'] >= threshold_date]
        recent_historical_white_ball_pb_combos = get_historical_combinations_set(recent_historical_combos_df, include_powerball=True)
    
    # Use the global historical_white_ball_pb_combos for all-time check if avoid_repeats_years is 0
    current_historical_pb_combos = historical_white_ball_pb_combos if avoid_repeats_years == 0 else recent_historical_white_ball_pb_combos

    max_attempts = 2000 # Increased limit for more restrictive criteria

    for attempt in range(max_attempts):
        
        white_balls = []
        powerball = 0
        
        available_white_for_gen = [num for num in range(white_ball_range[0], white_ball_range[1] + 1) if num not in excluded_numbers]
        available_powerball_for_gen = [num for num in range(powerball_range[0], powerball_range[1] + 1)]

        if len(available_white_for_gen) < 5:
            flash("Not enough available white balls with current exclusions/range.", 'error')
            return None, None
        if len(available_powerball_for_gen) < 1:
            flash("Not enough available Powerball numbers with current range.", 'error')
            return None, None

        if strategy == "trend_based":
            # Prioritize hot numbers, avoid cold
            hot_nums_white, cold_nums_white = hot_cold_numbers(df, df['Draw Date'].max())
            hot_white_list = hot_nums_white.index.tolist()
            
            hot_pb, cold_pb = hot_cold_powerball_numbers(df, df['Draw Date'].max())
            hot_pb_list = hot_pb.index.tolist()

            temp_white_balls = []
            while len(temp_white_balls) < 5:
                # 70% chance to pick from hot numbers, 30% from general available
                if len(hot_white_list) > 0 and random.random() < 0.7:
                    num = random.choice(hot_white_list)
                else:
                    num = random.choice(available_white_for_gen)
                
                if num not in temp_white_balls:
                    temp_white_balls.append(num)
            white_balls = sorted(temp_white_balls)
            
            if len(hot_pb_list) > 0 and random.random() < 0.7:
                powerball = random.choice(hot_pb_list)
            else:
                powerball = random.choice(available_powerball_for_gen)

        elif strategy == "low_hit_pattern":
            # Prioritize cold/due numbers
            hot_nums_white, cold_nums_white = hot_cold_numbers(df, df['Draw Date'].max())
            cold_white_list = cold_nums_white.index.tolist()
            
            hot_pb, cold_pb = hot_cold_powerball_numbers(df, df['Draw Date'].max())
            cold_pb_list = cold_pb.index.tolist()

            temp_white_balls = []
            while len(temp_white_balls) < 5:
                # 70% chance to pick from cold numbers, 30% from general available
                if len(cold_white_list) > 0 and random.random() < 0.7:
                    num = random.choice(cold_white_list)
                else:
                    num = random.choice(available_white_for_gen)
                
                if num not in temp_white_balls:
                    temp_white_balls.append(num)
            white_balls = sorted(temp_white_balls)
            
            if len(cold_pb_list) > 0 and random.random() < 0.7:
                powerball = random.choice(cold_pb_list)
            else:
                powerball = random.choice(available_powerball_for_gen)

        else: # "random" or "balanced_strategy" (random with checks)
            white_balls = random.sample(available_white_for_gen, 5)
            powerball = random.choice(available_powerball_for_gen)

        # Check for exact repeat (if avoiding repeats is enabled or always for all-time unique)
        if check_exact_match(white_balls, powerball, current_historical_pb_combos):
            continue # Try again if it's a repeat

        # Ensure group A numbers (always apply if the list is not empty)
        if group_a:
            group_a_numbers_in_white = [num for num in white_balls if num in group_a]
            if len(group_a_numbers_in_white) < 2: # At least 2 numbers from Group A
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
        
        # Check high/low balance condition
        if high_low_balance is not None and len(high_low_balance) == 2:
            low_numbers = [num for num in white_balls if num <= 34]
            high_numbers = [num for num in white_balls if num >= 35]
            if not (len(low_numbers) >= high_low_balance[0] and len(high_numbers) >= high_low_balance[1]):
                continue
        
        # Combo condition (simplified for now, more complex statistical rarity needs extensive historical analysis)
        # This part is more conceptual in Python without a pre-computed rarity database
        # For this implementation, it just ensures *some* combo exists if selected.
        if combo_choice != "No Combo":
            combo_size = 0
            if combo_choice == "2-combo":
                combo_size = 2
            elif combo_choice == "3-combo":
                combo_size = 3
            
            if combo_size > 0:
                # Check if *any* combo of `combo_size` exists in historical data (not necessarily rare)
                # This is a very loose check and doesn't guarantee a "rare" combo as per the feature request
                # To truly implement "combo rarity", you'd need pre-computed frequencies of all possible pairs/triplets
                # and generate based on those statistical outliers.
                found_historical_sub_combo = False
                for sub_combo in combinations(white_balls, combo_size):
                    # Check if this exact sub_combo (sorted) exists in historical_white_ball_combos
                    # This is still a challenging check for *all* sub_combos without a dedicated pre-computed set
                    # For now, it simply ensures the number of combinations matches the type requested.
                    pass # Placeholder
        
        # If all checks pass, we found a valid combination
        return sorted(white_balls), powerball

    flash("Could not generate numbers matching all criteria after many attempts. Try loosening your filters.", 'error')
    return None, None

# Function to calculate combinations (n choose k)
def calculate_combinations(n, k):
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)

# Check if generated numbers match historical data (used for displaying last drawn date)
def check_historical_match(df, white_balls, powerball):
    if df.empty:
        return None
    sorted_gen_white = sorted(white_balls)
    for _, row in df.iterrows():
        historical_white_balls = sorted(row[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.tolist())
        historical_powerball = row['Powerball']
        if sorted_gen_white == historical_white_balls and powerball == historical_powerball:
            return row['Draw Date']
    return None

# Hit & Miss Tracker functions
def calculate_draws_ago(df, number, is_powerball=False):
    if df.empty:
        return -1
    col_names = ['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5'] if not is_powerball else ['Powerball']
    
    # Iterate in reverse order to find the most recent draw
    for i, row in df.iloc[::-1].iterrows():
        if is_powerball:
            if row['Powerball'] == number:
                return len(df) - 1 - i
        else:
            if number in row[col_names].values:
                return len(df) - 1 - i
    return -1 # Not found

def calculate_hit_frequency_in_last_x_draws(df, number, num_draws, is_powerball=False):
    if df.empty or num_draws <= 0:
        return 0.0
    
    recent_data = df.tail(num_draws)
    count = 0
    if is_powerball:
        count = recent_data[recent_data['Powerball'] == number].shape[0]
    else:
        for _, row in recent_data.iterrows():
            if number in row[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values:
                count += 1
    return count / num_draws * 100 if num_draws > 0 else 0.0

def get_hot_cold_due_numbers(df, last_draw_date, num_draws_for_analysis=100):
    if df.empty:
        return {}, {}
        
    # White balls
    # Calculate overall average frequency for white balls
    all_white_balls_drawn = df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.flatten()
    white_ball_counts_all_time = pd.Series(all_white_balls_drawn).value_counts().reindex(range(1, 70), fill_value=0)
    avg_white_freq_all_time = white_ball_counts_all_time.mean()

    white_ball_status = {}
    for num in range(1, 70):
        draws_ago = calculate_draws_ago(df, num, is_powerball=False)
        frequency_last_x = calculate_hit_frequency_in_last_x_draws(df, num, num_draws_for_analysis, is_powerball=False)
        
        status = "Warm" # Default status
        if draws_ago == -1: # Number has never been drawn
            status = "Cold" # Consider never drawn as very cold
        else:
            # Determine status based on frequency and recency
            # These thresholds are heuristic and can be tuned
            if frequency_last_x >= (avg_white_freq_all_time / len(df) * num_draws_for_analysis * 1.2): # Higher than average in recent draws
                status = "Hot"
            elif frequency_last_x <= (avg_white_freq_all_time / len(df) * num_draws_for_analysis * 0.8): # Lower than average in recent draws
                status = "Cold"

            if draws_ago > 20: # If not seen for many draws, it's 'due'
                status = "Due"
            
            # Refine Hot/Cold: If it's very hot, it overrides due/cold
            if frequency_last_x >= (avg_white_freq_all_time / len(df) * num_draws_for_analysis * 1.5) and draws_ago < 5:
                status = "Hot"
            
            # If it's very cold and has not been drawn for a long time
            if frequency_last_x == 0 and draws_ago > 50:
                status = "Cold"

        white_ball_status[num] = {
            'draws_ago': draws_ago,
            'frequency_last_x': round(frequency_last_x, 2),
            'status': status
        }
    
    # Powerball
    all_powerballs_drawn = df['Powerball'].values.flatten()
    powerball_counts_all_time = pd.Series(all_powerballs_drawn).value_counts().reindex(range(1, 27), fill_value=0)
    avg_powerball_freq_all_time = powerball_counts_all_time.mean()

    powerball_status = {}
    for num in range(1, 27):
        draws_ago = calculate_draws_ago(df, num, is_powerball=True)
        frequency_last_x = calculate_hit_frequency_in_last_x_draws(df, num, num_draws_for_analysis, is_powerball=True)
        
        status = "Warm"
        if draws_ago == -1:
            status = "Cold"
        else:
            if frequency_last_x >= (avg_powerball_freq_all_time / len(df) * num_draws_for_analysis * 1.2):
                status = "Hot"
            elif frequency_last_x <= (avg_powerball_freq_all_time / len(df) * num_draws_for_analysis * 0.8):
                status = "Cold"

            if draws_ago > 10: # Due if not seen for a while (Powerball has smaller range)
                status = "Due"

            if frequency_last_x >= (avg_powerball_freq_all_time / len(df) * num_draws_for_analysis * 1.5) and draws_ago < 3:
                status = "Hot"
            
            if frequency_last_x == 0 and draws_ago > 20:
                status = "Cold"

        powerball_status[num] = {
            'draws_ago': draws_ago,
            'frequency_last_x': round(frequency_last_x, 2),
            'status': status
        }
            
    return white_ball_status, powerball_status

# Frequency analysis of white balls and Powerball (all time)
def frequency_analysis(df):
    if df.empty:
        return pd.Series(), pd.Series()
    white_balls = df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.flatten()
    white_ball_freq = pd.Series(white_balls).value_counts().reindex(range(1, 70), fill_value=0)

    powerball_freq = df['Powerball'].value_counts().reindex(range(1, 27), fill_value=0)

    return white_ball_freq, powerball_freq

# Hot and cold numbers analysis (for a specific period, e.g., last year)
def hot_cold_numbers(df, last_draw_date):
    if df.empty:
        return pd.Series(), pd.Series()
    one_year_ago = last_draw_date - pd.DateOffset(years=1)
    recent_data = df[df['Draw Date'] >= one_year_ago]

    white_balls = recent_data[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.flatten()
    white_ball_freq = pd.Series(white_balls).value_counts().sort_values(ascending=False)

    # Ensure all numbers are represented in cold_numbers, filling with 0 if not drawn
    all_white_numbers_range = pd.Series(range(1, 70))
    current_freq = white_ball_freq.reindex(all_white_numbers_range, fill_value=0)

    hot_numbers = current_freq.nlargest(14)
    cold_numbers = current_freq.nsmallest(14) # numbers that appeared least often in the last year

    return hot_numbers, cold_numbers

# Hot and cold Powerball numbers analysis (for a specific period, e.g., last year)
def hot_cold_powerball_numbers(df, last_draw_date):
    if df.empty:
        return pd.Series(), pd.Series()
    one_year_ago = last_draw_date - pd.DateOffset(years=1)
    recent_data = df[df['Draw Date'] >= one_year_ago]
    
    powerball_freq = recent_data['Powerball'].value_counts().sort_values(ascending=False)

    all_pb_numbers_range = pd.Series(range(1, 27))
    current_freq = powerball_freq.reindex(all_pb_numbers_range, fill_value=0)

    hot_numbers = current_freq.nlargest(5) # Top 5 hot powerballs
    cold_numbers = current_freq.nsmallest(5) # Bottom 5 cold powerballs

    return hot_numbers, cold_numbers

# Monthly white ball analysis
def monthly_white_ball_analysis(df, last_draw_date):
    if df.empty:
        return {}
    six_months_ago = last_draw_date - pd.DateOffset(months=6)
    recent_data = df[df['Draw Date'] >= six_months_ago].copy() # Use .copy() to avoid SettingWithCopyWarning

    recent_data['Month'] = recent_data['Draw Date'].dt.to_period('M')
    monthly_balls_series = recent_data.groupby('Month')[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].apply(lambda x: x.values.flatten().tolist())
    
    # Convert PeriodIndex to string for JSON serialization
    monthly_balls_dict = {str(month): balls for month, balls in monthly_balls_series.items()}
    return monthly_balls_dict

# Sum of main balls analysis
def sum_of_main_balls(df):
    if df.empty:
        return pd.DataFrame()
    return df[['Draw Date', 'Sum']]

# Find past results with a specific sum of white balls
def find_results_by_sum(df, target_sum):
    if df.empty:
        return pd.DataFrame()
    results = df[df['Sum'] == target_sum]
    return results[['Draw Date', 'Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball', 'Sum']]

# Simulate multiple draws and track frequencies
def simulate_multiple_draws_enhanced(df, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, num_draws=100, strategy="random", avoid_repeats_years=0):
    if df.empty:
        flash("Historical data not loaded, cannot run simulations.", 'error')
        return {}, {}, {}
    
    all_generated_white_balls = []
    all_generated_powerballs = []
    generated_draws_list = [] # To store full white ball sets for pair analysis

    # Attempt to generate numbers
    for _ in range(num_draws):
        white_balls, powerball = generate_powerball_numbers(df, group_a, odd_even_choice, combo_choice, white_ball_range, powerball_range, excluded_numbers, strategy=strategy, avoid_repeats_years=avoid_repeats_years)
        if white_balls and powerball:
            all_generated_white_balls.extend(white_balls)
            all_generated_powerballs.append(powerball)
            generated_draws_list.append(tuple(sorted(white_balls))) # Store as sorted tuple for consistency

    white_ball_freq = Counter(all_generated_white_balls)
    powerball_freq = Counter(all_generated_powerballs)

    pair_freq = Counter()
    for draw in generated_draws_list:
        for p in combinations(draw, 2):
            pair_freq[tuple(sorted(p))] += 1 # Ensure pairs are sorted for consistent counting

    return dict(white_ball_freq), dict(powerball_freq), dict(pair_freq)


# Winning probability and partial match probabilities (existing functions)
def winning_probability(white_ball_range, powerball_range):
    total_white_balls = white_ball_range[1] - white_ball_range[0] + 1
    white_ball_combinations = calculate_combinations(total_white_balls, 5)

    total_powerballs = powerball_range[1] - powerball_range[0] + 1
    total_combinations = white_ball_combinations * total_powerballs

    probability_1_in_x = f"1 in {total_combinations:,}"
    probability_percentage = f"{1 / total_combinations * 100:.10f}%" if total_combinations > 0 else "0%"

    return probability_1_in_x, probability_percentage

def partial_match_probabilities(white_ball_range, powerball_range):
    total_white_balls = white_ball_range[1] - white_ball_range[0] + 1
    total_powerballs = powerball_range[1] - powerball_range[0] + 1
    
    # Total possible combinations (Denominator for probabilities)
    total_combinations = calculate_combinations(total_white_balls, 5) * total_powerballs
    if total_combinations == 0:
        return {} # Return empty if no combinations are possible

    probabilities = {}

    # Match 5 White Balls + Powerball
    probabilities["Match 5 White Balls + Powerball"] = f"1 in {total_combinations:,}"

    # Match 5 White Balls (No Powerball)
    comb_5_white_0_pb = calculate_combinations(5, 5) * calculate_combinations(total_white_balls - 5, 0) * calculate_combinations(total_powerballs - 1, 1)
    if comb_5_white_0_pb > 0:
        probabilities["Match 5 White Balls (No Powerball)"] = f"1 in {total_combinations / comb_5_white_0_pb:,.0f}"

    # Match 4 White Balls + Powerball
    comb_4_white_1_pb = calculate_combinations(5, 4) * calculate_combinations(total_white_balls - 5, 1) * calculate_combinations(1, 1)
    if comb_4_white_1_pb > 0:
        probabilities["Match 4 White Balls + Powerball"] = f"1 in {total_combinations / comb_4_white_1_pb:,.0f}"

    # Match 4 White Balls (No Powerball)
    comb_4_white_0_pb = calculate_combinations(5, 4) * calculate_combinations(total_white_balls - 5, 1) * calculate_combinations(total_powerballs - 1, 1)
    if comb_4_white_0_pb > 0:
        probabilities["Match 4 White Balls (No Powerball)"] = f"1 in {total_combinations / comb_4_white_0_pb:,.0f}"
    
    # Match 3 White Balls + Powerball
    comb_3_white_1_pb = calculate_combinations(5, 3) * calculate_combinations(total_white_balls - 5, 2) * calculate_combinations(1, 1)
    if comb_3_white_1_pb > 0:
        probabilities["Match 3 White Balls + Powerball"] = f"1 in {total_combinations / comb_3_white_1_pb:,.0f}"

    # Match 3 White Balls (No Powerball)
    comb_3_white_0_pb = calculate_combinations(5, 3) * calculate_combinations(total_white_balls - 5, 2) * calculate_combinations(total_powerballs - 1, 1)
    if comb_3_white_0_pb > 0:
        probabilities["Match 3 White Balls (No Powerball)"] = f"1 in {total_combinations / comb_3_white_0_pb:,.0f}"

    # Match 2 White Balls + Powerball
    comb_2_white_1_pb = calculate_combinations(5, 2) * calculate_combinations(total_white_balls - 5, 3) * calculate_combinations(1, 1)
    if comb_2_white_1_pb > 0:
        probabilities["Match 2 White Balls + Powerball"] = f"1 in {total_combinations / comb_2_white_1_pb:,.0f}"

    # Match 1 White Ball + Powerball
    comb_1_white_1_pb = calculate_combinations(5, 1) * calculate_combinations(total_white_balls - 5, 4) * calculate_combinations(1, 1)
    if comb_1_white_1_pb > 0:
        probabilities["Match 1 White Ball + Powerball"] = f"1 in {total_combinations / comb_1_white_1_pb:,.0f}"

    # Match 0 White Balls + Powerball
    comb_0_white_1_pb = calculate_combinations(5, 0) * calculate_combinations(total_white_balls - 5, 5) * calculate_combinations(1, 1)
    if comb_0_white_1_pb > 0:
        probabilities["Match 0 White Balls + Powerball"] = f"1 in {total_combinations / comb_0_white_1_pb:,.0f}"


    return probabilities

# Pair and Triplet Frequency Analysis
def get_pair_and_triplet_frequencies(df, period="all_time"):
    if df.empty:
        return {'top_pairs': [], 'top_triplets': [], 'top_consecutive_pairs': []}

    if period == "last_12_months":
        one_year_ago = df['Draw Date'].max() - pd.DateOffset(years=1)
        data = df[df['Draw Date'] >= one_year_ago]
    else: # "all_time"
        data = df

    all_pairs = []
    all_triplets = []
    all_consecutive_pairs = []

    for _, row in data.iterrows():
        white_balls = sorted(row[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].tolist())
        
        # Pairs
        for p in combinations(white_balls, 2):
            all_pairs.append(tuple(sorted(p)))
            if p[1] == p[0] + 1: # Check for consecutive pairs
                all_consecutive_pairs.append(tuple(sorted(p)))

        # Triplets
        for t in combinations(white_balls, 3):
            all_triplets.append(tuple(sorted(t)))

    pair_freq = Counter(all_pairs)
    triplet_freq = Counter(all_triplets)
    consecutive_pair_freq = Counter(all_consecutive_pairs)

    top_10_pairs = sorted(pair_freq.items(), key=lambda item: item[1], reverse=True)[:10]
    top_10_triplets = sorted(triplet_freq.items(), key=lambda item: item[1], reverse=True)[:10]
    top_10_consecutive_pairs = sorted(consecutive_pair_freq.items(), key=lambda item: item[1], reverse=True)[:10]

    return {
        'top_pairs': top_10_pairs,
        'top_triplets': top_10_triplets,
        'top_consecutive_pairs': top_10_consecutive_pairs
    }

# Match Checker
def check_user_numbers(df, user_white_balls, user_powerball):
    if df.empty:
        return []
    matches = []
    user_white_balls_set = set(user_white_balls)

    for _, row in df.iterrows():
        historical_white_balls = set(row[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].values.tolist())
        historical_powerball = row['Powerball']
        draw_date = row['Draw Date']

        matched_white_count = len(user_white_balls_set.intersection(historical_white_balls))
        powerball_match = (user_powerball == historical_powerball)

        if matched_white_count > 0 or powerball_match:
            match_details = {
                'draw_date': draw_date.strftime('%Y-%m-%d'),
                'matched_white_balls_count': matched_white_count,
                'powerball_match': powerball_match,
                'historical_draw': row[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5', 'Powerball']].values.tolist()
            }
            matches.append(match_details)
    return matches

# Default ranges and excluded numbers
white_ball_range = (1, 69)
powerball_range = (1, 26)
excluded_numbers = []
group_a = [3, 5, 6, 7, 9, 11, 15, 16, 18, 21, 23, 24, 27, 31, 32, 33, 36, 42, 44, 45, 48, 50, 51, 54, 55, 60, 66, 69]


@app.route('/')
def index():
    global last_draw
    last_draw_dict = last_draw.to_dict() if not last_draw.empty else {}
    return render_template('index.html', last_draw=last_draw_dict)

@app.route('/generate', methods=['POST'])
def generate():
    odd_even_choice = request.form.get('odd_even_choice', 'Any')
    combo_choice = request.form.get('combo_choice', 'No Combo')
    white_ball_min = int(request.form.get('white_ball_min', 1))
    white_ball_max = int(request.form.get('white_ball_max', 69))
    white_ball_range_gen = (white_ball_min, white_ball_max) # Renamed to avoid conflict with global
    powerball_min = int(request.form.get('powerball_min', 1))
    powerball_max = int(request.form.get('powerball_max', 26))
    powerball_range_gen = (powerball_min, powerball_max) # Renamed to avoid conflict with global
    excluded_numbers_raw = request.form.get('excluded_numbers', '')
    excluded_numbers_gen = [int(num.strip()) for num in excluded_numbers_raw.split(",")] if excluded_numbers_raw else []
    high_low_balance_raw = request.form.get('high_low_balance', '')
    high_low_balance = tuple(map(int, high_low_balance_raw.split())) if high_low_balance_raw else None
    strategy = request.form.get('strategy', 'random')
    avoid_repeats_years = int(request.form.get('avoid_repeats_years', 0))

    # Generate the numbers
    white_balls, powerball = generate_powerball_numbers(df, group_a, odd_even_choice, combo_choice, white_ball_range_gen, powerball_range_gen, excluded_numbers_gen, high_low_balance, strategy, avoid_repeats_years)

    last_draw_dates = {}
    if white_balls and powerball and not df.empty:
        for number in white_balls:
            last_date_series = df.loc[df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].apply(lambda row: number in row.values, axis=1), 'Draw Date']
            if not last_date_series.empty:
                last_draw_dates[f"White Ball {number}"] = last_date_series.max().strftime('%Y-%m-%d')
            else:
                last_draw_dates[f"White Ball {number}"] = "Never Drawn"

        last_pb_date_series = df.loc[df['Powerball'] == powerball, 'Draw Date']
        if not last_pb_date_series.empty:
            last_draw_dates[f"Powerball {powerball}"] = last_pb_date_series.max().strftime('%Y-%m-%d')
        else:
            last_draw_dates[f"Powerball {powerball}"] = "Never Drawn"


        # Check if the generated numbers have been drawn before
        if check_exact_match(white_balls, powerball, historical_white_ball_pb_combos):
            exact_match_date = check_historical_match(df, white_balls, powerball)
            if exact_match_date:
                flash(f"Combination {white_balls} + {powerball} was previously drawn on: {exact_match_date.strftime('%Y-%m-%d')}", 'info')
            else: # Should not happen if check_exact_match is true
                flash(f"Combination {white_balls} + {powerball} was previously drawn.", 'info')
        else:
            flash("Combination is NEW and has never been drawn before!", 'success')
    elif white_balls is None or powerball is None:
        pass # Flash messages are already handled by generate_powerball_numbers
    else:
        flash("Cannot generate numbers: Historical data is empty or invalid.", 'error')


    last_draw_dict = last_draw.to_dict() if not last_draw.empty else {}
    return render_template('index.html', 
                           white_balls=white_balls, 
                           powerball=powerball, 
                           last_draw=last_draw_dict, 
                           last_draw_dates=last_draw_dates,
                           odd_even_choice_selected=odd_even_choice, # Pass back for UI persistence
                           combo_choice_selected=combo_choice,
                           white_ball_min_selected=white_ball_min,
                           white_ball_max_selected=white_ball_max,
                           powerball_min_selected=powerball_min,
                           powerball_max_selected=powerball_max,
                           excluded_numbers_selected=excluded_numbers_raw,
                           high_low_balance_selected=high_low_balance_raw,
                           strategy_selected=strategy,
                           avoid_repeats_years_selected=avoid_repeats_years,
                           active_tab='generate_numbers_section') # For keeping accordion open

@app.route('/frequency_analysis')
def frequency_analysis_route():
    white_ball_freq, powerball_freq = frequency_analysis(df)
    last_draw_dict = last_draw.to_dict() if not last_draw.empty else {}
    return render_template('index.html', 
                           white_ball_freq=white_ball_freq.to_dict(), 
                           powerball_freq=powerball_freq.to_dict(), 
                           last_draw=last_draw_dict,
                           active_tab='analysis_tools') # For keeping accordion open - Renamed to this for clarity on FE

@app.route('/hot_cold_numbers_detailed')
def hot_cold_numbers_detailed_route():
    white_ball_status = {}
    powerball_status = {}
    if not df.empty and not last_draw.empty:
        white_ball_status, powerball_status = get_hot_cold_due_numbers(df, last_draw['Draw Date'])
    else:
        flash("Historical data not loaded, cannot perform Hit & Miss analysis.", 'error')

    last_draw_dict = last_draw.to_dict() if not last_draw.empty else {}
    return render_template('index.html', 
                           white_ball_status=white_ball_status, 
                           powerball_status=powerball_status, 
                           last_draw=last_draw_dict,
                           active_tab='hit_miss_tracker') # For keeping accordion open

@app.route('/pair_triplet_analysis')
def pair_triplet_analysis_route():
    all_time_analysis = {}
    last_12_months_analysis = {}
    if not df.empty:
        all_time_analysis = get_pair_and_triplet_frequencies(df, period="all_time")
        last_12_months_analysis = get_pair_and_triplet_frequencies(df, period="last_12_months")
    else:
        flash("Historical data not loaded, cannot perform Pair & Triplet analysis.", 'error')
    
    last_draw_dict = last_draw.to_dict() if not last_draw.empty else {}
    return render_template('index.html', 
                           all_time_pair_triplet_analysis=all_time_analysis,
                           last_12_months_pair_triplet_analysis=last_12_months_analysis,
                           last_draw=last_draw_dict,
                           active_tab='pair_triplet_frequency')

@app.route('/draw_sum_visualizer')
def draw_sum_visualizer_route():
    sum_data_for_chart = []
    if not df.empty and 'Sum' in df.columns:
        # Ensure sum_data_for_chart is always a list of native Python integers
        sum_data_for_chart = [int(s) for s in df['Sum'].tolist()]
    else:
        flash("Historical data not loaded or 'Sum' column missing, cannot visualize draw sums.", 'error')

    last_draw_dict = last_draw.to_dict() if not last_draw.empty else {}
    return render_template('index.html', 
                           sum_data_for_chart=sum_data_for_chart, 
                           last_draw=last_draw_dict,
                           active_tab='draw_sum_visualizer')


@app.route('/find_results_by_sum', methods=['POST'])
def find_results_by_sum_route():
    target_sum = int(request.form.get('target_sum'))
    results_by_sum_df = find_results_by_sum(df, target_sum)
    results_by_sum_list = results_by_sum_df.to_dict('records') if not results_by_sum_df.empty else []

    last_draw_dict = last_draw.to_dict() if not last_draw.empty else {}
    return render_template('index.html', 
                           results_by_sum=results_by_sum_list, 
                           last_draw=last_draw_dict,
                           target_sum_searched=target_sum,
                           active_tab='draw_sum_visualizer')

@app.route('/simulate_multiple_draws', methods=['POST'])
def simulate_multiple_draws_route():
    num_draws = int(request.form.get('num_draws'))
    odd_even_choice = request.form.get('odd_even_choice_sim', 'Any')
    combo_choice = request.form.get('combo_choice_sim', 'No Combo')
    white_ball_min = int(request.form.get('white_ball_min_sim', 1))
    white_ball_max = int(request.form.get('white_ball_max_sim', 69))
    white_ball_range_sim = (white_ball_min, white_ball_max)
    powerball_min = int(request.form.get('powerball_min_sim', 1))
    powerball_max = int(request.form.get('powerball_max_sim', 26))
    powerball_range_sim = (powerball_min, powerball_max)
    excluded_numbers_raw_sim = request.form.get('excluded_numbers_sim', '')
    excluded_numbers_sim = [int(num.strip()) for num in excluded_numbers_raw_sim.split(",")] if excluded_numbers_raw_sim else []
    strategy = request.form.get('strategy_sim', 'random')
    avoid_repeats_years = int(request.form.get('avoid_repeats_years_sim', 0))

    if df.empty:
        flash("Historical data not loaded, cannot run simulations.", 'error')
        return redirect(url_for('index'))

    white_ball_freq_sim, powerball_freq_sim, pair_freq_sim = simulate_multiple_draws_enhanced(
        df, group_a, odd_even_choice, combo_choice, white_ball_range_sim, powerball_range_sim, 
        excluded_numbers_sim, num_draws, strategy, avoid_repeats_years
    )
    
    last_draw_dict = last_draw.to_dict() if not last_draw.empty else {}
    return render_template('index.html', 
                           sim_white_ball_freq=white_ball_freq_sim,
                           sim_powerball_freq=powerball_freq_sim,
                           sim_pair_freq=sorted(pair_freq_sim.items(), key=lambda x: x[1], reverse=True)[:20], # Top 20 pairs
                           num_draws_simulated=num_draws,
                           last_draw=last_draw_dict,
                           active_tab='draw_simulation_enhancements')

@app.route('/match_checker', methods=['POST'])
def match_checker_route():
    user_white_balls_str = request.form.get('user_white_balls', '')
    user_powerball_str = request.form.get('user_powerball', '')

    user_white_balls = []
    try:
        if user_white_balls_str:
            user_white_balls = sorted([int(num.strip()) for num in user_white_balls_str.split(',')])
            if len(user_white_balls) != 5:
                flash("Please enter exactly 5 white ball numbers.", 'error')
                return redirect(url_for('index'))
            if not all(1 <= num <= 69 for num in user_white_balls):
                flash("White ball numbers must be between 1 and 69.", 'error')
                return redirect(url_for('index'))
        else:
            flash("Please enter your white ball numbers.", 'error')
            return redirect(url_for('index'))
    except ValueError:
        flash("Invalid white ball numbers. Please use comma-separated integers (e.g., 5,12,30,45,60).", 'error')
        return redirect(url_for('index'))
    
    user_powerball = None
    try:
        if user_powerball_str:
            user_powerball = int(user_powerball_str.strip())
            if not (1 <= user_powerball <= 26):
                flash("Powerball number must be between 1 and 26.", 'error')
                return redirect(url_for('index'))
        else:
            flash("Please enter your Powerball number.", 'error')
            return redirect(url_for('index'))
    except ValueError:
        flash("Invalid Powerball number. Please enter an integer.", 'error')
        return redirect(url_for('index'))

    if df.empty:
        flash("Historical data not loaded, cannot check matches.", 'error')
        return redirect(url_for('index'))

    matches = check_user_numbers(df, user_white_balls, user_powerball)
    
    last_draw_dict = last_draw.to_dict() if not last_draw.empty else {}
    return render_template('index.html', 
                           user_checked_white_balls=user_white_balls,
                           user_checked_powerball=user_powerball,
                           match_results=matches,
                           last_draw=last_draw_dict,
                           active_tab='match_checker')

# Existing routes (modified to pass active_tab for accordion control)
@app.route('/generate_modified', methods=['POST'])
def generate_modified():
    # This route is simplified. In a full implementation, "modified combination"
    # would involve more sophisticated logic like altering one number from a historical draw
    # or using common pairs/triplets in a specific way.
    # For now, it just generates a new random combo, ensuring it's not a historical repeat.

    if df.empty:
        flash("Historical data not loaded, cannot generate modified numbers.", 'error')
        return redirect(url_for('index'))
    
    new_white_balls, new_powerball = generate_powerball_numbers(
        df, group_a, "Any", "No Combo", white_ball_range, powerball_range, excluded_numbers, 
        strategy="random", avoid_repeats_years=0 # Always try to avoid all-time repeats here
    )
    
    last_draw_dates = {}
    if new_white_balls and new_powerball:
        for number in new_white_balls:
            last_date_series = df.loc[df[['Number 1', 'Number 2', 'Number 3', 'Number 4', 'Number 5']].apply(lambda row: number in row.values, axis=1), 'Draw Date']
            if not last_date_series.empty:
                last_draw_dates[f"White Ball {number}"] = last_date_series.max().strftime('%Y-%m-%d')
            else:
                last_draw_dates[f"White Ball {number}"] = "Never Drawn"

        last_pb_date_series = df.loc[df['Powerball'] == new_powerball, 'Draw Date']
        if not last_pb_date_series.empty:
            last_draw_dates[f"Powerball {new_powerball}"] = last_pb_date_series.max().strftime('%Y-%m-%d')
        else:
            last_draw_dates[f"Powerball {new_powerball}"] = "Never Drawn"

        if check_exact_match(new_white_balls, new_powerball, historical_white_ball_pb_combos):
            flash("Generated modified combination was already drawn. Try again.", 'info')
        else:
            flash("Generated NEW modified combination!", 'success')
    else:
        flash("Failed to generate a modified combination.", 'error')


    last_draw_dict = last_draw.to_dict() if not last_draw.empty else {}
    return render_template('index.html', 
                           white_balls=new_white_balls, # Reusing white_balls and powerball variables
                           powerball=new_powerball, 
                           last_draw=last_draw_dict, 
                           last_draw_dates=last_draw_dates,
                           active_tab='generate_modified')


@app.route('/find_results_by_first_white_ball', methods=['POST'])
def find_results_by_first_white_ball():
    white_ball_number = int(request.form.get('white_ball_number'))
    
    if df.empty:
        flash("Historical data not loaded, cannot find results.", 'error')
        return redirect(url_for('index'))

    # Filter the dataframe to find rows where the first white ball number matches the entered number
    results = df[df['Number 1'] == white_ball_number]
    
    # Sort the results by year (extract year from 'Draw Date' and sort)
    sort_by_year = request.form.get('sort_by_year') == 'on'
    if sort_by_year:
        results['Year'] = results['Draw Date'].dt.year
        results = results.sort_values(by='Year')
    
    results_dict = results.to_dict('records')
    
    last_draw_dict = last_draw.to_dict() if not last_draw.empty else {}
    
    return render_template('index.html', 
                           results_by_first_white_ball=results_dict, 
                           last_draw=last_draw_dict,
                           white_ball_number=white_ball_number,
                           sort_by_year=sort_by_year,
                           active_tab='find_results_by_first_white_ball')

@app.route('/winning_probability')
def winning_probability_route():
    probability_1_in_x, probability_percentage = winning_probability(white_ball_range, powerball_range)
    last_draw_dict = last_draw.to_dict() if not last_draw.empty else {}
    return render_template('index.html', 
                           probability_1_in_x=probability_1_in_x, 
                           probability_percentage=probability_percentage, 
                           last_draw=last_draw_dict,
                           active_tab='probability_calculations')

@app.route('/partial_match_probabilities')
def partial_match_probabilities_route():
    probabilities = partial_match_probabilities(white_ball_range, powerball_range)
    last_draw_dict = last_draw.to_dict() if not last_draw.empty else {}
    return render_template('index.html', 
                           probabilities=probabilities, 
                           last_draw=last_draw_dict,
                           active_tab='probability_calculations')

@app.route('/export_analysis_results')
def export_analysis_results_route():
    # Placeholder for export logic, as direct file saving isn't user-facing in this environment
    # In a real app, you'd generate a CSV/Excel file and send it as a response
    flash("Export functionality would generate a file for download in a deployed application. (Not directly available in this live demo.)", 'info')
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)

