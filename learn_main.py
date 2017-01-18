import model
import numpy as np
import sys
import matplotlib.pyplot as plt

start = (
    0.20,   # share of lowest ability (permanently unemployed)
    0.70,   # share of mid ability
    0.10,   # share of high ability
    1,      # prod fn constant
    0.5,    # lower tier job share of output
    0.02    # discount rate
)

firm = (
    30,  # workers of type 'new' assigned to role 1
    30,  # workers of type 'low' assigned to role 1
    0,   # workers of type 'high' assigned to role 1
    0,   # workers of type 'neg' assigned to role 1
    10,  # workers of type 'pos' assigned to role 1
    20,  # workers of type 'new' assigned to role 2
    0,   # workers of type 'low' assigned to role 2
    0,   # workers of type 'high' assigned to role 2
    0,   # workers of type 'neg' assigned to role 2
    0    # workers of type 'pos' assigned to role 2
)

print(np.sum(start[0:3]))
print(firm)
model1 = model.Model(*start)
output = model1.prod(*firm)
model2 = model.Model()

grid2 = np.array([[39, 0, 0, 0, 0, 390, 0, 0, 0, 0],
                 [40, 0, 0, 0, 0, 390, 0, 0, 0, 0],
                 [0, 40, 0, 0, 0, 0, 0, 39, 0, 0],
                 [1, 40, 0, 0, 0, 0, 0, 39, 0, 0]])

grid4 = np.array([[20, 20, 0, 0, 0, 200, 0, 20, 0, 0],
                 [20, 20, 0, 0, 0, 200, 0, 21, 0, 0],
                 [0, 20, 0, 0, 0, 0, 0, 20, 0, 0],
                 [0, 20, 0, 0, 0, 0, 0, 21, 0, 0]])

grid3 = np.array([[20, 20, 0, 0, 0, 200, 0, 0, 0, 0],
                 [20, 21, 0, 0, 0, 200, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 10, 0, 0, 0, 0],
                 [0, 10, 0, 0, 0, 0, 0, 10, 0, 0]])

grid = 100*np.ones(10)
test = model1.prod(*grid3.T)
print(test)
print(test[1]-test[0])
print(test[3]-test[2])

'''
want to show
1. how uncertainty influences staffing decisions
    a. marginal product across feasible input space
    b. effect uncertain roles have on marginal product of certain workers
2. equilibrium behavior
3. comparative statics


staffing algo
1. fill certain
2. fill neg type
3. fill pos type
4. fill new
'''




x = np.zeros(10)
xhi = np.zeros(10)
highval = -1;
for i in range(0, 10**10):
    number2 = str(i).zfill(10)
    for c in range(0, 9):
        x[c] = number2[c]
    f1 = 10 * x
    f2 = 100 - f1
    target = model1.prod(*f1) + model1.prod(*f2)
    if target > highval:
        xhi = np.copy(x)
        highval = np.copy(target)






# solve
def newf(alloc):
    reverse =