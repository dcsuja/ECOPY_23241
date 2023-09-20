#ab9oac

def evens_from_list(input_list):
    return [number for number in input_list if number % 2 == 0]

my_list = [1, 2, 3, 4, 5, 6]
result = evens_from_list(my_list)
print(result)

def every_element_is_odd(input_list):
    for number in input_list:
        if number % 2 == 0:
            return False
    return True

my_list = [1, 3, 5, 7, 8]
result = every_element_is_odd(my_list)
print(result)


def kth_largest_in_list(input_list, kth_largest):
    sorted_list = sorted(input_list, reverse=True)

    if 1 <= kth_largest <= len(sorted_list):
        return sorted_list[kth_largest - 1]
    else:
        raise ValueError("Not valid k value")


my_list = [5, 4, 9, 3, 1, 6]
k = 2
result = kth_largest_in_list(my_list, k)
print(result)


def cumavg_list(input_list):
    cumulative_sum = 0
    cumulative_average_list = []

    for i, num in enumerate(input_list, start=1):
        cumulative_sum += num
        cumulative_average = cumulative_sum / i
        cumulative_average_list.append(cumulative_average)

    return cumulative_average_list


my_list = [3, 5, 7, 8, 11]
result = cumavg_list(my_list)
print(result)


def element_wise_multiplication(input_list1, input_list2):
    if len(input_list1) != len(input_list2):
        raise ValueError("Lengths of lists are different.")

    result = []

    for i in range(len(input_list1)):
        product = input_list1[i] * input_list2[i]
        result.append(product)

    return result


list1 = [1, 2, 3, 4, 5]
list2 = [6, 7, 8, 9, 10]
result = element_wise_multiplication(list1, list2)
print(result)

def merge_lists(*lists):
    merged_list = []
    for sublist in lists:
        merged_list.extend(sublist)
    return merged_list

my_list1 = [1, 2, 3]
my_list2 = [4, 5]

result = merge_lists(my_list1, my_list2)
print(result)


def squared_odds(input_list):
    squared_odds_list = []

    for number in input_list:
        if number % 2 != 0:
            squared_odds_list.append(number ** 2)

    return squared_odds_list


my_list = [1, 2, 3, 4, 5, 6, 7]
result = squared_odds(my_list)
print(result)

def reverse_sort_by_key(input_dict):
    sorted_dict = dict(sorted(input_dict.items(), key=lambda item: item[0], reverse=True))
    return sorted_dict

my_dict = {'a': 4, 'b': 6, 'c': 8}
result = reverse_sort_by_key(my_dict)
print(result)


def sort_list_by_divisibility(input_list):
    result_dict = {
        'by_two': [],
        'by_five': [],
        'by_two_and_five': [],
        'by_none': []
    }

    for number in input_list:
        if number % 2 == 0 and number % 5 == 0:
            result_dict['by_two_and_five'].append(number)
        elif number % 2 == 0:
            result_dict['by_two'].append(number)
        elif number % 5 == 0:
            result_dict['by_five'].append(number)
        else:
            result_dict['by_none'].append(number)

    return result_dict


my_list = [3, 4, 5, 6, 7, 8, 10, 15]
result = sort_list_by_divisibility(my_list)
print(result)
