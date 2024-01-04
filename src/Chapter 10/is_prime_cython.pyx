# defining a basic function that is pythonic
def is_prime_cy1(I):
    if I%2==0: return False
    for i in range(3, int(I ** 0.5) + 1, 2):
        if I%i==0: return False
    return True

# making a new function that defines data types
def is_prime_cy2(long I):
    cdef long i
    if I%2==0: return False
    for i in range(3, int(I ** 0.5) + 1, 2):
        if I%i==0: return False 
    return True