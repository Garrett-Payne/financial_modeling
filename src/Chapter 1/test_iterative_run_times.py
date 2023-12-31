'''This code snippet is intended to compare the differences in timing performance using a native 
for loop, using numpy, and using numexpr'''

# based on pages 22 + 23 of Python for Finance

import math
import timeit
import numpy as np
import numexpr as ne
ne.set_num_threads(1)
f = '3 * log(a) + cos(a) ** 2'
loops = 2500000
a = range(1, loops) 
b = np.arange(1,loops)
def f(x):
    return 3 * math.log(x) + math.cos(x) ** 2


## The Code below will run the 3 different methods of iterating and print out the time it takes to complete 2500000 iterations
if __name__ == "__main__":


    # use basic for loop
    setup_code1 = '''from __main__ import f,a'''
    test_code1 = 'r = [f(x) for x in a]'
    print("Native for loop time in seconds:")
    print(timeit.timeit(stmt=test_code1,setup=setup_code1,number=1))
    print("\n")

    # use numpy array:
    setup_code2 = '''from __main__ import f,b
import numpy as np'''
    test_code2 ='r = 3 * np.log(b) + np.cos(b) ** 2'
    print("Numpy array time in seconds:")
    print(timeit.timeit(stmt=test_code2,setup=setup_code2,number=1))
    print("\n")

    # use numexpr (set with 1 thread):
    setup_code3 = '''from __main__ import b
import numexpr as ne
ne.set_num_threads(1)'''
    test_code3 = '''g = '3 * log(b) + cos(b) ** 2' 
r= ne.evaluate(g)'''
    print("Numexpr (1 thread) time in seconds:")
    print(timeit.timeit(stmt=test_code3,setup=setup_code3,number=1))
    print("\n")


    # use numexpr (set with 4 threads):
    setup_code4 = '''from __main__ import b
import numexpr as ne
ne.set_num_threads(4)'''
    test_code4 = '''g = '3 * log(b) + cos(b) ** 2' 
r= ne.evaluate(g)'''
    print("Numexpr (4 thread) time in seconds:")
    print(timeit.timeit(stmt=test_code4,setup=setup_code4,number=1))
    print("\n")