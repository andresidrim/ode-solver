import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import os

sp.init_printing(pretty_print=True)

def first_order_ode(t, y, a, b):
    dydt = a * y + b * t
    return dydt

def solveFirstOrderODE(a, b, y0):
    # Define the time span
    t_span = [0, 1]

    # Solve the first-order ODE
    sol = solve_ivp(first_order_ode, t_span, [y0], args=(a, b))

    # Extract the solution
    t = sol.t
    y = sol.y[0]

    # Print the solution
    # print("Solution (numerical):")
    # print("t:", t)
    # print("y:", y)

    # Define the first-order ODE symbolically
    t_sym = sp.symbols('t')
    y_sym = sp.Function('y')(t_sym)
    equation = sp.Eq(y_sym.diff(t_sym) - a * y_sym - b * t_sym, 0)

    # Solve the ODE symbolically with initial condition
    C1 = sp.symbols('C1')
    solution = sp.dsolve(equation, y_sym, ics={y_sym.subs(t_sym, 0): y0})

    # # Extract the value of C1
    # C1_value = sp.solve(solution.rhs.subs(t_sym, 0) - y0, C1)

    # Print the general solution
    print("\nGeneral solution (symbolic):")
    print(solution)

    # Print the value of C1
    print("Value of C1:", y[0])
        
def second_order_ode(t, y, a, b, c, d):
    dydt = y[1]
    dzdt = -(b * y[1] + c * y[0] + d) / a
    return [dydt, dzdt]

def solveSecondOrderODE(a, b, c, d, y0, y_prime_0):
    # Initial conditions
    y0 = [y0, y_prime_0] 

    # Time span
    t_span = [0, 10]

    # Solve the ODE
    sol = solve_ivp(second_order_ode, t_span, y0, args=(a, b, c, d))

    # Extract the solution
    t = sol.t
    y = sol.y[0]

    # Print the values of C1 and C2 (which correspond to y(0) and y'(0))
    print("Values of C1 and C2 (from numerical solution):")
    print("C1 (y(0)): ", y0[0])
    print("C2 (y'(0)): ", y0[1])

    # Define the second-order ODE symbolically
    t_sym = sp.symbols('t')
    y_sym = sp.Function('y')(t_sym)
    equation = sp.Eq(a * y_sym.diff(t_sym, t_sym) + b * y_sym.diff(t_sym) + c * y_sym + d, 0)

    # Solve the ODE symbolically
    solution = sp.dsolve(equation)

    # Print the general solution
    print("\nGeneral solution (symbolic):")
    print(solution)

    # Identify the constants in the solution
    constants = sp.symbols('C1 C2')

    # Extract the constants from the solution
    constants_values = [solution.subs(constant, sp.symbols(str(constant))) for constant in constants]

    # Print the values of C1 and C2
    print("\nValues of C1 and C2 (from symbolic solution):")
    for i, constant in enumerate(constants):
        print(f"{constant}: {constants_values[i]}")


def zeroInputFirstDegreeODE(a, b, y0):
    t = sp.symbols('t')
    y = sp.Function('y')(t)
    
    ode = sp.Eq(a * y.diff(t) + b * y, 0)  # Equation to be solved
    print("The differential equation is:")
    sp.pprint(ode)
    print()
    
    general_solution = sp.dsolve(ode)
    print("The general solution to the ODE is:")
    sp.pprint(general_solution)
    print()
    
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

def zeroInputSecondDegreeODE(a, b, c, y0, dy0):
    t = sp.symbols('t')
    y = sp.Function('y')(t)
    
    ode = sp.Eq(a * y.diff(t, t) + b * y.diff(t) + c * y, 0)  # Equation to be solved
    print("The differential equation is:")
    sp.pprint(ode)
    print()
    
    general_solution = sp.dsolve(ode)
    print("The general solution to the ODE is:")
    sp.pprint(general_solution)
    print()
    
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

def zeroStateFirstDegreeODE(a, b, c, forcing_function):
    t, s = sp.symbols('t s')
    Y = sp.Function('Y')(s)
    y = sp.Function('y')(t)
    
    F = sp.laplace_transform(forcing_function, t, s, noconds=True)
    print("Laplace transform of the forcing function F(s):")
    sp.pprint(F)
    print()
    
    odeLaplace = sp.Eq(a * s * Y - a * 0 + b * Y + c, F)   # Equation to be solved
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

def zeroStateSecondDegreeODE(a, b, c, d, forcing_function):
    t, s = sp.symbols('t s')
    Y = sp.Function('Y')(s)
    y = sp.Function('y')(t)
    
    F = sp.laplace_transform(forcing_function, t, s, noconds=True)
    print("Laplace transform of the forcing function F(s):")
    sp.pprint(F)
    print()
    
    odeLaplace = sp.Eq(a * s**2 * Y - a * s * 0 - a * 0 + b * s * Y + c * Y + d, F)   # Equation to be solved
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
        print('(5) General First Degree ODE')
        print('(6) General Second Degree ODE')
        usrInput = input()

        os.system('cls')

        if int(usrInput) in [1, 2, 3, 4, 5, 6]:
            break

    if int(usrInput) == 1:
        a = float(input("Enter the coefficient a: "))
        b = float(input("Enter the coefficient b: "))
        y0 = float(input("Enter the initial condition y(0): "))
        zeroInputFirstDegreeODE(a, b, y0)
    elif int(usrInput) == 2:
        a = float(input("Enter the coefficient a: "))
        b = float(input("Enter the coefficient b: "))
        c = float(input("Enter the coefficient c: "))
        y0 = float(input("Enter the initial condition y(0): "))
        dy0 = float(input("Enter the initial condition y'(0): "))
        zeroInputSecondDegreeODE(a, b, c, y0, dy0)
    elif int(usrInput) == 3:
        a = float(input("Enter the coefficient a: "))
        b = float(input("Enter the coefficient b: "))
        c = float(input("Enter the coefficient c: "))
        forcing_function = input("Enter the forcing function (in terms of t): ")
        zeroStateFirstDegreeODE(a, b, c, forcing_function)
    elif int(usrInput) == 4:
        a = float(input("Enter the coefficient a: "))
        b = float(input("Enter the coefficient b: "))
        c = float(input("Enter the coefficient c: "))
        d = float(input("Enter the coefficient d: "))
        forcing_function = input("Enter the forcing function (in terms of t): ")
        zeroStateSecondDegreeODE(a, b, c, d, forcing_function)
    elif int(usrInput) == 5:
        a = float(input("Enter the coefficient a: "))
        b = float(input("Enter the coefficient b: "))
        y0 = float(input("Enter the initial condition y(0): "))
        solveFirstOrderODE(a, b, y0)
    elif int(usrInput) == 6:
        a = float(input("Enter the coefficient a: "))
        b = float(input("Enter the coefficient b: "))
        c = float(input("Enter the coefficient c: "))
        d = float(input("Enter the coefficient d: "))
        y0 = float(input("Enter the initial condition y(0): "))
        dy0 = float(input("Enter the initial condition y'(0): "))
        solveSecondOrderODE(a, b, c, d, y0, dy0)

if __name__ == "__main__":
    main()