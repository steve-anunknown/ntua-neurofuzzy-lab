import numpy as np 

def gradient_descent(f, df, x_curr, a, eps=0.001, limit=1000):
    value = f(*x_curr)
    delta = df(*x_curr)
    x_next = x_curr - a * delta 
    nvalue = f(*x_next)
    iterations = 0
    while abs(value - nvalue) > eps and iterations < limit:
        value = nvalue
        delta = df(*x_next)
        x_next = x_next - a * delta 
        nvalue = f(*x_next)
        iterations += 1
    return nvalue, x_next, iterations

def newtons_method(f, df, ddf, x_curr, a, eps=0.001, limit=1000):
    x_next = x_curr - df(*x_curr) / ddf(*x_curr)
    value = f(*x_curr)
    nvalue = f(*x_next)
    iterations = 0
    while abs(value - nvalue) > eps and iterations < limit:
        value = nvalue
        x_next += -df(*x_next) / ddf(*x_next)
        nvalue = f(*x_next)
        iterations += 1
    return nvalue, x_next, iterations

def momentum_method(f, df, x_curr, a1, a2, eps=0.001, limit=1000):
    v_curr = np.array([0, 0])
    v_next = a1 * v_curr - a2 * df(*x_curr)
    x_next = x_curr + v_next
    iterations = 0;
    while abs(f(*x_curr) - f(*x_next)) > eps and iterations < limit:
        v_curr = v_next
        x_curr = x_next
        v_next = a1 * v_curr - a2 * df(*x_curr)
        x_next = x_curr + v_next
        iterations += 1;
    return f(*x_next), x_next, iterations

