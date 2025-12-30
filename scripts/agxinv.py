import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize

def piecewise_function(x):
    # Constants
    c = 20/33

    # Pre-calculating complex terms for the second piece
    # Constant term inside the second denominator
    inner_const = ( (140608 * (13/33)**0.25 * np.sqrt(2)) / 35937 ) - 1
    scaling_factor = 35937 * inner_const * (33/13)**0.25 / 2197

    # Define the two parts of the function
    def part1(x):
        numerator = 2 * (x - c)
        denominator = (1 - (476063 * (x - c)**3 / 8000))**(1/3)
        return (numerator / denominator) + 0.5

    def part2(x):
        numerator = 2 * (x - c)
        # Power is 13/4 (3.25) and outer power is 4/13 (~0.307)
        term = scaling_factor * (x - c)**(13/4) + 1
        denominator = term**(4/13)
        return (numerator / denominator) + 0.5

    # Apply piecewise logic
    return np.piecewise(x,
                        [(x >= 0) & (x < c), (x >= c) & (x <= 1)],
                        [part1, part2])

# 1. Define the model function
def complex_model(x, a4, a3, a2, a1, a0, c1, c0):
    poly = a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0
    # Add a tiny epsilon to prevent log(0) or log(negative)
    poly = np.clip(poly, 1e-9, 1 - 1e-9)
    return c1 * np.log2((1/poly) - 1) + c0

# 2. Your data (replace with actual arrays)
x_data = np.linspace(0.0, 1.0, num=65536)
y_data = piecewise_function(x_data)
def objective(params):
    y_pred = complex_model(y_data, *params)
    return np.sum((y_pred - x_data)**2)

constraints = [
    {'type': 'eq', 'fun': lambda p: complex_model(0.0, *p) - 0.0},
    {'type': 'eq', 'fun': lambda p: complex_model(1.0, *p) - 1.0}
]

initial_guesses = [0.0, -0.34838081, -0.13518383, -0.43438567,  0.99778071,  0.11707198,  0.70691182]

result = minimize(objective, initial_guesses, constraints=constraints)
params = result.x

print(f"Optimal coefficients: {params}")
print(f"a_4={params[0]}")
print(f"a_3={params[1]}")
print(f"a_2={params[2]}")
print(f"a_1={params[3]}")
print(f"a_0={params[4]}")
print(f"c_1={params[5]}")
print(f"c_0={params[6]}")