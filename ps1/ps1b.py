###########################
# 6.0002 Problem Set 1b: Space Change
# Name: B********* ****
# Collaborators: none
# Time: 1hr 45min 3/2/22
# Author: charz, cdenise

#================================
# Part B: Golden Eggs
#================================

# Problem 1
def dp_make_weight(egg_weights, target_weight, memo = {}):
    """
    Find number of eggs to bring back, using the smallest number of eggs. Assumes there is
    an infinite supply of eggs of each weight, and there is always an egg of value 1.
    
    Parameters:
    egg_weights - tuple of integers, available egg weights sorted from smallest to largest value (1 = d1 < d2 < ... < dk)
    target_weight - int, amount of weight we want to find eggs to fit
    memo - dictionary, OPTIONAL parameter for memoization (you may not need to use this parameter depending on your implementation)
    
    Returns: int, smallest number of eggs needed to make target weight
    """
    # TODO: Your code here
    # if its in the memo, return the result from the memo
    # print(egg_weights, target_weight, "a")
    # base case: target weight 0
    if target_weight in memo:
        result = memo[target_weight]
    elif target_weight == 0:
        result = 0
    elif target_weight - egg_weights[-1] < 0:
        result = dp_make_weight(egg_weights[:-1], target_weight, memo)
    else:
        # check left branch (taking the largest egg weight)
        # print(egg_weights, target_weight)
        left_result = 1 + dp_make_weight(egg_weights, target_weight - max(egg_weights), memo)
        # print(left_result, "L")
        # check right branch (skipping the largest egg weight in favor of smaller one)
        try:
            right_result = dp_make_weight(egg_weights[:-1], target_weight, memo)
        except Exception:
            right_result = left_result + 1  # funny little workaround.
        # print(right_result, "R")
        # compare somewhere
        if left_result < right_result:
            result = left_result
        else:
            result = right_result

    # memo will store the best possible result w/ given remaining after its calculated
    memo[target_weight] = result
    # print(result, target_weight, "C")
    return result

# EXAMPLE TESTING CODE, feel free to add more if you'd like
if __name__ == '__main__':
    egg_weights = (1, 5, 10, 25)
    n = 99
    print("Egg weights = (1, 5, 10, 25)")
    print("n = 99")
    print("Expected ouput: 9 (3 * 25 + 2 * 10 + 4 * 1 = 99)")
    print("Actual output:", dp_make_weight(egg_weights, n))
    print()
