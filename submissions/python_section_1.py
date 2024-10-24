from typing import Dict, List

import pandas as pd

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    length = len(lst)

    for i in range(0, length, n):
        temp = []
        for j in range(min(n, length - i)):
            temp.insert(0, lst[i + j])
        result.extend(temp)

    return result

# Taking input from the user
user_input = input("Enter a list of integers separated by spaces: ")
n = int(input("Enter the group size n: "))

# Converting the input string to a list of integers
lst = list(map(int, user_input.split()))

# Calling the function and printing the result
result = reverse_by_n_elements(lst, n)
print("The list after reversing every group of n elements is:", result)

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.

    
    """
    result = {}
    for word in lst:
        length = len(word)
        if length not in result:
            result[length] = []
        result[length].append(word)

    # Sorting the dictionary by the key (length of strings)
    return dict(sorted(result.items()))

# Taking input from the user
user_input = input("Enter a list of strings separated by spaces: ")

# Splitting the input string into a list of strings
lst = user_input.split()

# Calling the function and printing the result
result = group_by_length(lst)
print("The grouped dictionary by string length is:", result)

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    """
    def flatten(current, parent_key='', sep='.'):
        items = []
        for k, v in current.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    items.extend(flatten(item, f"{new_key}[{i}]", sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    return flatten(nested_dict, sep=sep)

# Taking input from the user
user_input = input("Enter a nested dictionary (in Python dict format): ")
nested_dict = eval(user_input)  # Using eval to convert string input to dictionary

# Calling the function and printing the result
result = flatten_dict(nested_dict)
print("The flattened dictionary is:", result)

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    """
    def backtrack(path, remaining, used, results):
        if len(path) == len(nums):
            results.append(path[:])
            return

        for i in range(len(remaining)):
            if used[i] or (i > 0 and remaining[i] == remaining[i - 1] and not used[i - 1]):
                continue
            used[i] = True
            path.append(remaining[i])
            backtrack(path, remaining, used, results)
            path.pop()
            used[i] = False

    nums.sort()  # Sort the input to handle duplicates
    results = []
    used = [False] * len(nums)
    backtrack([], nums, used, results)

    return results

# Taking input from the user
user_input = input("Enter a list of integers separated by spaces: ")
nums = list(map(int, user_input.split()))

# Calling the function and printing the result
result = unique_permutations(nums)
print("The unique permutations are:", result)

import re

def find_all_dates(text: str) -> List[str]:
    """
    Finds all dates in the specified formats from the given text.
    """
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b' # yyyy.mm.dd
    ]
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text))
    return dates

# Taking input from the user
user_input = input("Enter a text containing dates: ")

# Calling the function and printing the result
result = find_all_dates(user_input)
print("The dates found in the text are:", result)


import polyline
from typing import Tuple
import math

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the Haversine distance between two latitude/longitude points in meters.
    """
    R = 6371000  # Radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    """
    # Decode the polyline string into a list of coordinates
    coordinates = polyline.decode(polyline_str)

    # Create a DataFrame from the coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    # Calculate the distance between consecutive points
    distances = [0.0]
    for i in range(1, len(coordinates)):
        lat1, lon1 = coordinates[i - 1]
        lat2, lon2 = coordinates[i]
        distances.append(haversine_distance(lat1, lon1, lat2, lon2))

    df['distance'] = distances
    return df



def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the matrix by 90 degrees clockwise and transform elements.
    """
    n = len(matrix)
    rotated = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]

    transformed = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated[i]) - rotated[i][j]
            col_sum = sum(row[j] for row in rotated) - rotated[i][j]
            transformed[i][j] = row_sum + col_sum

    return transformed

# Taking input from the user
user_input = input("matrix  =")
matrix = eval(user_input)

# Calling the function and printing the result
result = rotate_and_multiply_matrix(matrix)
print("The transformed matrix is:", result)

def check_time_completeness(df: pd.DataFrame) -> pd.Series:
    """
    Verifies whether each (id, id_2) pair in the dataset covers a full 24-hour period for all 7 days of the week.

    Args:
        df (pd.DataFrame): DataFrame containing columns id, id_2, startDay, startTime, endDay, endTime.

    Returns:
        pd.Series: A boolean series with multi-index (id, id_2), indicating if each pair has complete coverage.
    """
    # Drop rows with missing essential values
    df = df.dropna(subset=['id', 'id_2', 'startDay', 'startTime', 'endDay', 'endTime'])

    # Convert startDay and endDay to integers, handling non-finite values gracefully
    df['startDay'] = pd.to_numeric(df['startDay'], errors='coerce').fillna(0).astype(int)
    df['endDay'] = pd.to_numeric(df['endDay'], errors='coerce').fillna(0).astype(int)

    # Convert startTime and endTime to timedelta format
    df['startTime'] = pd.to_timedelta(df['startTime'], errors='coerce').fillna(pd.Timedelta(seconds=0))
    df['endTime'] = pd.to_timedelta(df['endTime'], errors='coerce').fillna(pd.Timedelta(seconds=0))

    # Create a multi-index based on id and id_2
    df.set_index(['id', 'id_2'], inplace=True)

    # Generate a reference set of all 7 days of the week and 24 hours of the day
    all_days = set(range(1, 8))  # Assuming startDay and endDay are represented by numbers 1 (Monday) to 7 (Sunday)
    full_day = pd.Timedelta('1 day')

    # Function to check coverage for each group
    def is_complete(group):
        days_covered = set()
        for _, row in group.iterrows():
            # Collect all days covered from startDay to endDay
            days_covered.update(range(row['startDay'], row['endDay'] + 1))
            # Check if time covers full 24 hours for each covered day
            if row['endTime'] - row['startTime'] < full_day:
                return False
        # Ensure all days of the week are covered
        return days_covered == all_days

    # Group by multi-index and check completeness for each (id, id_2) pair
    completeness = df.groupby(level=['id', 'id_2']).apply(is_complete)

    return completeness

# Example usage
if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('dataset-1.csv')
    # Check completeness
    result = check_time_completeness(df)
    print("Time Completeness Check:")
    print(result)
