from math import sqrt

def quadratic_equation(a, b, c):
    D = b**2-4*a*c
    if D>0:
        square_root = sqrt(D)
        x1 = (-b+square_root)/(2*a)
        x2 = (-b-square_root)/(2*a)
        return x1, x2