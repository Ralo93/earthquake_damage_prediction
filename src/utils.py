import sys
sys.path.append('..')

def sum_up(numbers):
    if not numbers:
        raise ValueError("The list is empty!")
    return sum(numbers)