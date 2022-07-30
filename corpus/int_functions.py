def fibonacci(i: int):
    if i <= 0:
        return 0
    if i == 1:
        return 1
    return fibonacci(i - 1) + fibonacci(i - 2)


def factorial(i: int):
    if i <= 0:
        return 1
    return factorial(i - 1) * i
