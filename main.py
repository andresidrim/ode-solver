# Authors: André Sidrim, Fernanda Panzera

import sympy as sp
import os

def zeroInputFirstDegreeODE():
    t = sp.symbols('t')
    y = sp.Function('y')(t)
    
    ode = sp.Eq(y.diff(t) + 3 * y, 0) # -> Equation to be solved
    print("The differential equation is:")
    sp.pprint(ode)
    print()
    
    general_solution = sp.dsolve(ode)
    print("The general solution to the ODE is:")
    sp.pprint(general_solution)
    print()
    
    y0 = 2
    
    C1 = sp.symbols('C1')
    
    initial_condition_eq = general_solution.rhs.subs(t, 0) - y0
    print("Setting initial condition:")
    sp.pprint(initial_condition_eq)
    print()
    
    constant_solution = sp.solve(initial_condition_eq, C1)
    
    if constant_solution:
        C1value = constant_solution[0]
        particular_solution = general_solution.subs(C1, C1value)
        print("C1 =", C1value)
        print("\nThe particular solution to the ODE with the given initial condition is:")
        sp.pprint(particular_solution)
    else:
        print("Error: Unable to find a solution for the constant.")

def zeroInputSecondDegreeODE():
    t = sp.symbols('t')
    y = sp.Function('y')(t)
    
    ode = sp.Eq(y.diff(t, t) + 2 * y.diff(t) + y, 0) # -> Equation to be solved
    print("The differential equation is:")
    sp.pprint(ode)
    print()
    
    general_solution = sp.dsolve(ode)
    print("The general solution to the ODE is:")
    sp.pprint(general_solution)
    print()
    
    y0 = 1
    dy0 = 0
    
    constants = list(general_solution.rhs.free_symbols)
    C1, C2 = constants[0], constants[1]
    
    eq1 = general_solution.rhs.subs(t, 0) - y0
    eq2 = sp.diff(general_solution.rhs, t).subs(t, 0) - dy0
    print("Setting initial conditions:")
    sp.pprint(eq1)
    sp.pprint(eq2)
    print()
    
    constants_solution = sp.solve((eq1, eq2), (C1, C2))
    
    if constants_solution:
        C1value = constants_solution[C1]
        C2value = constants_solution[C2]
        particular_solution = general_solution.subs(constants_solution)
        print("C1 =", C1value.evalf())  
        print("C2 =", C2value.evalf())  
        print("\nThe particular solution to the ODE with the given initial conditions is:")
        sp.pprint(particular_solution)
    else:
        print("Error: Unable to find a solution for the constants.")


def zeroStateFirstDegreeODE():
    t, s = sp.symbols('t s')
    Y = sp.Function('Y')(s)
    y = sp.Function('y')(t)
    
    f = t  
    
    F = sp.laplace_transform(f, t, s, noconds=True)
    print("Laplace transform of the forcing function F(s):")
    sp.pprint(F)
    print()
    
    odeLaplace = sp.Eq(s * Y - 0 + 3 * Y, F)   # -> Equation to be solved
    print("Laplace transform of the differential equation:")
    sp.pprint(odeLaplace)
    print()
    
    YSol = sp.solve(odeLaplace, Y)[0]
    print("Solving for Y(s):")
    sp.pprint(YSol)
    print()
    
    ySol = sp.inverse_laplace_transform(YSol, s, t)
    print("Inverse Laplace transform to find y(t):")
    sp.pprint(ySol)
    print("\nThe zero state solution to the ODE is:")
    sp.pprint(ySol)

def zeroStateSecondDegreeODE():
    t, s = sp.symbols('t s')
    Y = sp.Function('Y')(s)
    y = sp.Function('y')(t)
    
    f = t  
    
    F = sp.laplace_transform(f, t, s, noconds=True)
    print("Laplace transform of the forcing function F(s):")
    sp.pprint(F)
    print()
    
    odeLaplace = sp.Eq(s**2 * Y - s*0 - 0 + 2 * s * Y + Y, F)   # -> Equation to be solved
    print("Laplace transform of the differential equation:")
    sp.pprint(odeLaplace)
    print()
    
    YSol = sp.solve(odeLaplace, Y)[0]
    print("Solving for Y(s):")
    sp.pprint(YSol)
    print()
    
    ySol = sp.inverse_laplace_transform(YSol, s, t)
    print("Inverse Laplace transform to find y(t):")
    sp.pprint(ySol)
    print("\nThe zero state solution to the ODE is:")
    sp.pprint(ySol)

def main():
    while True:
        print('(1) Zero Input First Degree ODE')
        print('(2) Zero Input Second Degree ODE')
        print('(3) Zero State First Degree ODE')
        print('(4) Zero State Second Degree ODE')
        usrInput = input()

        os.system('cls')

        if int(usrInput) in [1, 2, 3, 4]:
            break

    if int(usrInput) == 1:
        zeroInputFirstDegreeODE()
    elif int(usrInput) == 2:
        zeroInputSecondDegreeODE()
    elif int(usrInput) == 3:
        zeroStateFirstDegreeODE()
    elif int(usrInput) == 4:
        zeroStateSecondDegreeODE()

if __name__ == "__main__":
    main()