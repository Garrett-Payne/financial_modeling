def fib_rec_cy2(int n):
    if n<2: 
        return n
    else:
        return fib_rec_cy2(n - 1) + fib_rec_cy2(n - 2)
    
def fib_it_cy2(int n):
    cdef long i 
    cdef long x=0,y=1 
    for i in range(1, n + 1):
        x,y=y,x+y 
    return x