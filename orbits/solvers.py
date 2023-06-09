import logging


def newtons_method(f, fprime, x0, eps=1e-6):
    x = x0
    for i in range(100):
        delta = f(x) / fprime(x)
        x = x - delta
        if abs(delta) < eps:
            return x
    raise ValueError("Could not converge in 100 iterations, probably a problem")


def mean(a, b):
    return (a + b) / 2


def sign(x):
    return -1 if x < 0 else 1 if x > 0 else 0


def bracket_zero(f, floor, ceil, eps=1e-6, maxiter=100):
    x = mean(floor, ceil)
    if sign(f(floor)) * sign(f(ceil)) != -1:
        raise ValueError("There must be one zero crossing in the range")
    i = 0
    while abs(f(x)) > eps:
        logging.debug("bracket_zero is bounded by %f, %f", floor, ceil)
        if sign(f(x)) == sign(f(ceil)):
            floor, ceil = floor, mean(floor, ceil)
        else:
            floor, ceil = mean(floor, ceil), ceil
        x = mean(floor, ceil)
        i += 1
        if i > maxiter:
            raise ValueError("Could not converge in the limited iterations")
    return x
