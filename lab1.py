import math
import random
import statistics

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

khi_square_dict: dict[int, float] = {
    1: 3.8,
    2: 6.0,
    3: 7.8,
    4: 9.5,
    5: 11.1,
    6: 12.6,
    7: 14.1,
    8: 15.5,
    9: 16.9,
    10: 18.3,
    11: 19.7,
    12: 21.0,
    13: 22.4,
    14: 23.7,
    15: 25.0,
    16: 26.3,
    17: 27.6,
    18: 28.9,
    19: 30.1,
    20: 31.4,
    21: 32.7,
    22: 33.9,
    23: 35.2,
    24: 36.4,
    25: 37.7,
    26: 38.9,
    27: 40.1,
    28: 41.3,
    29: 42.6,
    30: 43.8
}


def exponential_distribution(lyambda: float) -> tuple[list, list]:
    x_list = []
    y_list = []
    for index in range(10000):
        rand_value = random.random()
        x_value = -np.log(rand_value) / lyambda
        y_value = np.exp(-lyambda*x_value)
        x_list.append(x_value)
        y_list.append(y_value)
    return x_list, y_list


def normal_distribution(alpha: float, sygma: float) -> tuple[list, list]:
    x_list = []
    y_list = []
    for index in range(10000):
        mu = -6
        for ind in range(12):
            mu += random.random()
        x_value = sygma * mu + alpha
        y_value = 1 / (sygma * math.sqrt(2 * math.pi)) * math.exp(-math.pow(x_value - alpha, 2)/(2 * math.pow(sygma, 2)))
        x_list.append(x_value)
        y_list.append(y_value)
    return x_list, y_list


def uniform_distribution(a: float, c: float) -> list:
    x_list = []
    z = a * random.random() % c
    for index in range(10000):
        z = a * z % c
        x_list.append(z/c)
    return x_list


def expand_by_intervals(x_list: list, count: int) -> list:
    interval_range = (max(x_list) - min(x_list)) / count
    intervals_dict: dict[tuple, int] = {}
    counter = min(x_list)
    for index in range(count):
        intervals_dict[(counter, counter + interval_range)] = 0
        counter = counter + interval_range
    return intervals_dict


def count_elements_for_intervals(x_list: list, intervals_dict: dict) -> dict:
    for x_value in x_list:
        for interval in intervals_dict.keys():
            if interval[0] < x_value and interval[1] >= x_value:
                intervals_dict[interval] += 1
    return intervals_dict


def expected_for_interval(x_start: float, x_end: float, distribution: str, **kwargs) -> float:
    exp_value = 0.0
    if distribution == "exponential":
        lyambda = kwargs['lyambda']
        exp_value = np.exp(-lyambda * x_start) - np.exp(-lyambda * x_end)
    elif distribution == "normal":
        sygma = kwargs['sygma']
        alpha = kwargs['alpha']

        def formule(x):
            return 1/(sygma * np.sqrt(2 * np.pi)) * np.exp(-math.pow(x - alpha, 2) / (2 * math.pow(sygma, 2)))
        result, message = integrate.quad(formule, x_start, x_end)
        exp_value = result
    elif distribution == "uniform":
        x_list = kwargs['x_list']
        exp_value = (x_end - x_start) / (max(x_list) - min(x_list))
    else:
        SystemExit
    return exp_value


def khi_values_for_intervals(expected_values_list: list,
                             found_values_dict: dict) -> tuple[float, float]:
    found_khi_value = 0.0
    for index, range_value in enumerate(found_values_dict.keys()):
        expected_value = expected_values_list[index] * 10000
        found_khi_value += math.pow(found_values_dict[range_value] - expected_value, 2) / expected_value
    return khi_square_dict[len(found_values_dict.keys()) - 1], found_khi_value


def draw_distribution_graph(x_list: list, distribution: str, space=0) -> None:
    plt.title(f"{distribution} distribution")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([min(x_list) - space, max(x_list) + space])
    bins = np.arange(-100, 100, 0.5)
    plt.hist(x_list, bins=bins, alpha=0.5)
    plt.show()


def draw_function_graph(x_list: list, y_list: list, distribution: str) -> None:
    plt.title(f"{distribution} function")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x_list, y_list)
    plt.show()


def print_results(distribution: str, x_list: list, khi_values: tuple, counted_intervals: dict) -> None:
    print(f"Results for the {distribution} distribution.")
    print(f"Mean value for the 'x' list: {statistics.mean(x_list)}.")
    print(f"Dyspers value for the 'x' list: {statistics.pvariance(x_list)}.")
    for interval_range in counted_intervals.keys():
        print(f"In the [{interval_range[0]} - {interval_range[1]}] there is {counted_intervals[interval_range]} elements.")
    print(f"Expected X^2 value is: {khi_values[0]}")
    print(f"Found X^2 value is: {khi_values[1]}")


def exponential_observing(distribution: str, intervals_amount: int, lyambda: float):
    x_list, y_list = exponential_distribution(lyambda)
    intervals_dict: dict = expand_by_intervals(x_list, intervals_amount)
    expected_values_list = []
    for interval in intervals_dict.keys():
        expected_values_list.append(expected_for_interval(interval[0], interval[1], distribution, lyambda=lyambda))
    elements_in_intervals_dict = count_elements_for_intervals(x_list, intervals_dict)
    khi_values = khi_values_for_intervals(expected_values_list, elements_in_intervals_dict)
    print_results(distribution, x_list, khi_values, elements_in_intervals_dict)
    draw_distribution_graph(x_list, distribution)
    draw_function_graph(x_list, y_list, distribution)


def normal_observing(distribution: str, intervals_amount: int, alpha: float, sygma: float):
    x_list, y_list = normal_distribution(alpha, sygma)
    intervals_dict: dict = expand_by_intervals(x_list, intervals_amount)
    expected_values_list = []
    for interval in intervals_dict.keys():
        expected_values_list.append(expected_for_interval(interval[0], interval[1], distribution, alpha=alpha, sygma=sygma))
    elements_in_intervals_dict = count_elements_for_intervals(x_list, intervals_dict)
    khi_values = khi_values_for_intervals(expected_values_list, elements_in_intervals_dict)
    print_results(distribution, x_list, khi_values, elements_in_intervals_dict)
    draw_distribution_graph(x_list, distribution)
    draw_function_graph(x_list, y_list, distribution)


def uniform_observing(distribution: str, intervals_amount: int, a: float, c: float):
    x_list = uniform_distribution(a, c)
    intervals_dict: dict = expand_by_intervals(x_list, intervals_amount)
    expected_values_list = []
    for interval in intervals_dict.keys():
        expected_values_list.append(expected_for_interval(interval[0], interval[1], distribution, x_list=x_list))
    elements_in_intervals_dict = count_elements_for_intervals(x_list, intervals_dict)
    khi_values = khi_values_for_intervals(expected_values_list, elements_in_intervals_dict)
    print_results(distribution, x_list, khi_values, elements_in_intervals_dict)
    draw_distribution_graph(x_list, distribution, 0.1)


if __name__ == "__main__":
    exponential_observing("exponential", 20, 0.5)
    normal_observing("normal", 20, 0.5, 2)
    uniform_observing("uniform", 20, math.pow(5, 13), math.pow(2, 31))
