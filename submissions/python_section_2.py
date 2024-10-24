import pandas as pd
import numpy as np
import datetime
# Load dataset
df = pd.read_csv('datasets\dataset-2.csv')
def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Extract unique IDs from the dataset
    unique_ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    # Create a DataFrame to store the distances with all values initialized to infinity
    distance_matrix = pd.DataFrame(np.inf, index=unique_ids, columns=unique_ids)
    
    # Fill the diagonal with zeros since the distance from any point to itself is zero
    np.fill_diagonal(distance_matrix.values, 0)
    
    # Populate the matrix with the given distances, ensuring symmetry
    for _, row in df.iterrows():
        id_start, id_end, distance = row['id_start'], row['id_end'], row['distance']
        distance_matrix.loc[id_start, id_end] = distance
        distance_matrix.loc[id_end, id_start] = distance

    # Use the Floyd-Warshall algorithm to find the shortest path between all pairs of nodes
    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.loc[i, j] > distance_matrix.loc[i, k] + distance_matrix.loc[k, j]:
                    distance_matrix.loc[i, j] = distance_matrix.loc[i, k] + distance_matrix.loc[k, j]

    return distance_matrix

def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Unroll the matrix into a long-format DataFrame with columns 'id_start', 'id_end', and 'distance'
    unrolled_df = df.stack().reset_index()
    unrolled_df.columns = ['id_start', 'id_end', 'distance']
    
    # Exclude rows where 'id_start' is equal to 'id_end' (diagonal values)
    unrolled_df = unrolled_df[unrolled_df['id_start'] != unrolled_df['id_end']]
    
    return unrolled_df

def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Filter the DataFrame to get distances for the given reference_id in 'id_start'
    reference_distances = df[df['id_start'] == reference_id]['distance']
    
    # Calculate the average distance for the reference_id
    average_distance = reference_distances.mean()
    
    # Define the 10% threshold range
    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1
    
    # Find 'id_start' values whose average distance lies within the 10% threshold range
    ids_within_threshold = df.groupby('id_start')['distance'].mean()
    ids_within_threshold = ids_within_threshold[(ids_within_threshold >= lower_bound) &
                                                (ids_within_threshold <= upper_bound)].index.tolist()

    # Sort the list of IDs
    ids_within_threshold.sort()
    
    return ids_within_threshold

def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type and add them as new columns
    for vehicle_type, rate in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate
    
    return df

def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # List of days from Monday to Sunday
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Create a list to hold the new rows
    new_rows = []
    
    # Iterate over each row in the input DataFrame
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        base_distances = {vehicle: row[vehicle] for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']}
        
        # Iterate over each day of the week
        for day in days_of_week:
            # Determine if it's a weekend
            is_weekend = day in ['Saturday', 'Sunday']
            
            # Iterate over each hour in the day to create time ranges
            for hour in range(24):
                # Define start and end time for each hour
                start_time = datetime.time(hour=hour, minute=0, second=0)
                end_time = datetime.time(hour=(hour + 1) % 24, minute=0, second=0)

                # Determine the discount factor
                if is_weekend:
                    discount_factor = 0.7
                else:
                    if start_time < datetime.time(10, 0):
                        discount_factor = 0.8
                    elif start_time < datetime.time(18, 0):
                        discount_factor = 1.2
                    else:
                        discount_factor = 0.8

                # Apply the discount factor to each vehicle type
                adjusted_rates = {vehicle: base_distances[vehicle] * discount_factor for vehicle in base_distances}

                # Create a new row with the additional time-related columns
                new_row = {
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': day,
                    'end_day': day,
                    'start_time': start_time,
                    'end_time': end_time,
                    **adjusted_rates
                }
                
                # Add the new row to the list
                new_rows.append(new_row)
    
    # Convert the list of new rows into a DataFrame
    expanded_df = pd.DataFrame(new_rows)
    
    return expanded_df



# Apply the functions
distance_matrix = calculate_distance_matrix(df)
unrolled_distance_df = unroll_distance_matrix(distance_matrix)
ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_distance_df, reference_id=1001400)
toll_rate_df = calculate_toll_rate(unrolled_distance_df)
time_based_toll_rate_df = calculate_time_based_toll_rates(toll_rate_df)

# Output results
print(distance_matrix)
print(unrolled_distance_df)
print(ids_within_threshold)
print(toll_rate_df)
print(time_based_toll_rate_df)
