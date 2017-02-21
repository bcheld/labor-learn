"""
Filename: learn_main.py

Author: Brian Held

A script to initialize various analyses of the learning model outlined in Ch. 2 of my thesis

"""
import model
import numpy as np
#import matplotlib.pyplot as plt

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

# 1. Show whether 1 or multiple firms exist in simple case
# 2. Show path over time of workers at single firm


"""
Code for when allocate function is vectorized:

wkrs = np.repeat(100,5)
output = model1.allocate(*wkrs)[1]
firm1 = np.random.randint(1,99,size=(10,5))
firm2 = np.subtract(100, firm1)
f1, p1 = model1.allocate(*firm1.T)
"""

print(np.sum(start[0:3]))
print(firm)
model1 = model.Model(*start)
output = model1.prod(*firm)
model2 = model.Model()


wkrs = np.repeat(100,5)
wkrs = np.append([wkrs],[np.repeat(99,5)],axis=0)
output = model1.allocate(*wkrs[0])[1]
output = np.append([output], [output], axis=0)

for x in range(100):
    firm1 = np.random.randint(1,99,size=5)
    firm2 = np.subtract(100, firm1)
    f1, p1 = model1.allocate(*firm1)
    f2, p2 = model1.allocate(*firm2)
    wkrs = np.append(wkrs, [firm1], axis=0)
    output = np.append(output, [p1 + p2], axis=0)


'''
print(np.sum(start[0:3]))
print(firm)
model1 = model.Model(*start)
output = model1.prod(*firm)
model2 = model.Model()
'''
a,b,c = model1.simt(100,0,0,0,0,10)

import pandas as pd
df = pd.DataFrame(wkrs)
df['output'] = output

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

# one extreme  1    8   39   93   67 - new and high go to one  firm
# 663, 498, 623, 973, 444

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


'''


# itertools?
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

'''