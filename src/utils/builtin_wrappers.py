#ab9oac

lista1 = [17, 18, 3, 14, 'a', 'alma']
print(lista1)

elemek = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
lista2 = list(elemek)
print(lista2)

lista2[1]

import random
random_elem = random.choice(lista2)
print(random_elem)

max(lista2)

lista2.index(max(lista2))

def contains_value(input_list, element):
    return element in input_list

result = contains_value(lista2, 3)

if result:
    print("The element is in the list.")
else:
    print("The element is not in the list.")

def number_of_elements_in_list(input_list):
    return len(input_list)

count = number_of_elements_in_list(lista2)

print(f"The number of elements in the list is: {count}")


def remove_every_element_from_list(input_list):
    input_list.clear


result = remove_every_element_from_list(lista2)

print(result)


def reverse_list(input_list):
    return input_list[::-1]


reversed_list = reverse_list(lista2)

print(reversed_list)

def odds_from_list(input_list) :
    return [x for x in input_list if x % 2 !=0]

odds_list = odds_from_list(lista2)

print(odds_list)

def number_of_odds_in_list(input_list) :
    return len([x for x in input_list if x % 2 !=0])

count = number_of_odds_in_list(lista2)

print(count)


def contains_odd(input_list):
    return [x for x in input_list if x % 2 != 0]


result = contains_odd(lista2)

if result:
    print("There is an odd integer in the list.")

else:
    print("There are no odd integers in the list.")

def second_largest_in_list(input_list) :
    return sorted_elements[-2]

elements = set(lista2)

sorted_elements = sorted(elements)

second_largest = second_largest_in_list(lista2)

print(second_largest)

def sum_of_elements_in_list(input_list):
    return sum(input_list)

total = sum_of_elements_in_list(lista2)

print(total_sum)


def cumsum_list(input_list):
    cum_sum = 0
    cumulative_list = []
    for item in input_list:
        cum_sum += item
        cumulative_list.append(cum_sum)
    return cumulative_list

cumulative_result = cumsum_list(lista2)
print(cumulative_result)


def element_wise_sum(input_list1, input_list2):
    if len(input_list1) != len(input_list2):
        raise ValueError("Length of list is different")

    result_list = []
    for i in range(len(input_list1)):
        result_list.append(input_list1[i] + input_list2[i])

    return result_list


result = element_wise_sum(lista1, lista2)
print(result)

def subset_of_list(input_list, start_index, end_index):
    if start_index < 0 or end_index >= len(input_list):
        raise ValueError("Invalid indexes")

    subset = input_list[start_index:end_index + 1]
    return subset

subset = subset_of_list(lista2, 2, 5)
print(subset)

def every_nth(input_list, step_size):
    if step_size <= 0:
        raise ValueError("Step-size must be positive.")

    result = input_list[::step_size]
    return result

result = every_nth(lista2, 3)
print(result)


def only_unique_in_list(input_list):
    unique_elements = set()

    for item in input_list:
        if item in unique_elements:
            return False
        unique_elements.add(item)

    return True


result1 = only_unique_in_list(lista1)
result2 = only_unique_in_list(lista2)

print(result1)
print(result2)


def keep_unique(input_list):
    unique_elements = []

    for item in input_list:
        if item not in unique_elements:
            unique_elements.append(item)

    return unique_elements


unique_elements = keep_unique(lista2)
print(unique_elements)

def swap(input_list, first_index, second_index):
    if (
        first_index < 0
        or second_index < 0
        or first_index >= len(input_list)
        or second_index >= len(input_list)
    ):
        raise ValueError("Invalid indexes")

    input_list[first_index], input_list[second_index] = input_list[second_index], input_list[first_index]
    return input_list

swap(lista2, 1, 3)
print(lista2)

def remove_element_by_value(input_list, value_to_remove):
    input_list = [item for item in input_list if item != value_to_remove]
    return input_list

value_to_remove = 2
new_list = remove_element_by_value(lista2, value_to_remove)
print(new_list)

def remove_element_by_index(input_list, index):
    if index < 0 or index >= len(input_list):
        raise ValueError("Érvénytelen index")

    del input_list[index]
    return input_list

index_to_remove = 2
new_list = remove_element_by_index(lista2, index_to_remove)
print(new_list)

def multiply_every_element(input_list, multiplier):
    result_list = [item * multiplier for item in input_list]
    return result_list

multiplier = 2
new_list = multiply_every_element(lista2, multiplier)
print(new_list)

my_dict = {'a': 9, 'b': [12, 'c']}
print(my_dict)

my_dict['a']

d_value = my_dict.get('d', Null)
print(d_value)

def remove_key(input_dict, key):
    if key in input_dict:
        del input_dict[key]
    return input_dict

key_to_remove = 'a'
new_dict = remove_key(my_dict, key_to_remove)
print(new_dict)

def sort_by_key(input_dict):
    sorted_dict = dict(sorted(input_dict.items(), key=lambda x: x[0]))
    return sorted_dict

sorted_dict = sort_by_key(my_dict)
print(sorted_dict)

def sum_in_dict(input_dict):
    total_sum = sum(input_dict.values())
    return float(total_sum)

total = sum_in_dict(my_dict)
print(total)

def merge_two_dicts(input_dict1, input_dict2):
    merged_dict = {**input_dict1, **input_dict2}
    return merged_dict

def merge_dicts(*dicts):
    merged_dict = {}
    for d in dicts:
        merged_dict.update(d)
    return merged_dict


def sort_list_by_parity(input_list):
    even_numbers = [num for num in input_list if num % 2 == 0]
    odd_numbers = [num for num in input_list if num % 2 != 0]
    result_dict = {'even': even_numbers, 'odd': odd_numbers}
    return result_dict


from typing import Dict


def mean_by_key_value(input_dict: Dict[str, list]) -> Dict[str, float]:
    result_dict = {}  # Új dictionary az átlagokkal

    for key, values in input_dict.items():
        if values:
            mean_value = sum(values) / len(values)  # Az értékek átlaga
            result_dict[key] = mean_value

    return result_dict


def count_frequency(input_list):
    frequency_dict = {}

    for item in input_list:
        if item in frequency_dict:
            frequency_dict[item] += 1
        else:
            frequency_dict[item] = 1

    return frequency_dict
