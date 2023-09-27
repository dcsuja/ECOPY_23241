import random

print(random.random())

print(random.randint(0,100))

random.seed(42)
print(random.randint(0,100))


def random_from_list(input_list):
    if len(input_list) == 0:
        raise ValueError("Input list is empty")

    random_index = random.randint(0, len(input_list) - 1)

    return input_list[random_index]


def random_sublist_from_list(input_list, number_of_elements):
    return random.sample(input_list, number_of_elements)


def random_from_string(input_string):
    return random.choice(input_string)


def hundred_small_random():
    random_list = [random.random() for _ in range(100)]

    return random_list

def hundred_large_random():
    random_list = [random.randint(10, 1000) for _ in range(100)]

    return random_list

def five_random_number_div_three():
    a = random.randrange(9, 1000, 3)
    b = random.randrange(9, 1000, 3)
    c = random.randrange(9, 1000, 3)
    d = random.randrange(9, 1000, 3)
    e = random.randrange(9, 1000, 3)
    random_5_elements = [a, b, c, d, e]
    return random_5_elements


def random_reorder(input_list):
    shuffled_list = random.sample(input_list, len(input_list))
    return shuffled_list

def uniform_one_to_five():
    return random.uniform(1, 6)
