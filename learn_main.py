import model
import numpy as np

start = np.array[
    0.20,   # share of lowest ability (permanently unemployed)
    0.70,   # share of mid ability
    0.10,   # share of high ability
    1,      # prod fn constant
    0.6,    # lower tier job share of output
    0.02    # discount rate
    ]

firm = [[
    30,     # workers of type 'new' assigned to role 1
    30,     # workers of type 'low' assigned to role 1
    0,      # workers of type 'high' assigned to role 1
    0,      # workers of type 'neg' assigned to role 1
    10],     # workers of type 'pos' assigned to role 1
    [20,     # workers of type 'new' assigned to role 2
    0,      # workers of type 'low' assigned to role 2
    0,      # workers of type 'high' assigned to role 2
    0,      # workers of type 'neg' assigned to role 2
    0]      # workers of type 'pos' assigned to role 2
    ]

print(np.sum(start[0:3]))
print(firm)
model1 = model.Model(*start[0:])
output = model1.prod(firm)

