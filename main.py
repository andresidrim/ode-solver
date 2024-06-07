import sympy as sp
import os

def zeroInputFirstDegreeODE():
    t = sp.symbols('t')
    y = sp.Function('y')(t)
    
    # Define the ODE
    ode = sp.Eq(y.diff(t) + 3 * y, 0)
    print("The differential equation is:")
    sp.pprint(ode)
    print()
    
    # Solve the ODE to get the general solution
    general_solution = sp.dsolve(ode)
    print("The general solution to the ODE is:")
    sp.pprint(general_solution)
    print()
    
    # Initial condition
    y0 = 2
    
    # Extract the constant symbol C1
    C1 = sp.symbols('C1')
    
    # Substitute t=0 and y(0)=y0 into the general solution and solve for C1
    initial_condition_eq = general_solution.rhs.subs(t, 0) - y0
    print("Setting initial condition:")
    sp.pprint(initial_condition_eq)
    print()
    
    constant_solution = sp.solve(initial_condition_eq, C1)
    
    if constant_solution:
        C1_val = constant_solution[0]
        particular_solution = general_solution.subs(C1, C1_val)
        print("C1 =", C1_val)
        print("\nThe particular solution to the ODE with the given initial condition is:")
        sp.pprint(particular_solution)
    else:
        print("Error: Unable to find a solution for the constant.")

def zeroInputSecondDegreeODE():
    t = sp.symbols('t')
    y = sp.Function('y')(t)
    
    # Define the ODE
    ode = sp.Eq(y.diff(t, t) + 2 * y.diff(t) + y, 0)
    print("The differential equation is:")
    sp.pprint(ode)
    print()
    
    # Solve the ODE to get the general solution
    general_solution = sp.dsolve(ode)
    print("The general solution to the ODE is:")
    sp.pprint(general_solution)
    print()
    
    # Initial conditions
    y0 = 1
    dy0 = 0
    
    # Find the constant symbols (C1, C2) and solve for them using the initial conditions
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
        C1_val = constants_solution[C1]
        C2_val = constants_solution[C2]
        particular_solution = general_solution.subs(constants_solution)
        print("C1 =", C1_val.evalf())  # Evaluate C1 for a specific value
        print("C2 =", C2_val.evalf())  # Evaluate C2 for a specific value
        print("\nThe particular solution to the ODE with the given initial conditions is:")
        sp.pprint(particular_solution)
    else:
        print("Error: Unable to find a solution for the constants.")


def zeroStateFirstDegreeODE():
    t, s = sp.symbols('t s')
    Y = sp.Function('Y')(s)
    y = sp.Function('y')(t)
    
    # Define the forcing function
    f = t  # For example, let's take f(t) = t
    
    # Define the Laplace transform of the forcing function
    F = sp.laplace_transform(f, t, s, noconds=True)
    print("Laplace transform of the forcing function F(s):")
    sp.pprint(F)
    print()
    
    # Define the Laplace transform of the ODE
    ode_laplace = sp.Eq(s * Y - 0 + 3 * Y, F)  # Assuming y(0) = 0 for zero state response
    print("Laplace transform of the differential equation:")
    sp.pprint(ode_laplace)
    print()
    
    # Solve for Y(s)
    Y_sol = sp.solve(ode_laplace, Y)[0]
    print("Solving for Y(s):")
    sp.pprint(Y_sol)
    print()
    
    # Take the inverse Laplace transform to find y(t)
    y_sol = sp.inverse_laplace_transform(Y_sol, s, t)
    print("Inverse Laplace transform to find y(t):")
    sp.pprint(y_sol)
    print("\nThe zero state solution to the ODE is:")
    sp.pprint(y_sol)

def zeroStateSecondDegreeODE():
    t, s = sp.symbols('t s')
    Y = sp.Function('Y')(s)
    y = sp.Function('y')(t)
    
    # Define the forcing function
    f = t  # For example, let's take f(t) = t
    
    # Define the Laplace transform of the forcing function
    F = sp.laplace_transform(f, t, s, noconds=True)
    print("Laplace transform of the forcing function F(s):")
    sp.pprint(F)
    print()
    
    # Define the Laplace transform of the ODE
    ode_laplace = sp.Eq(s**2 * Y - s*0 - 0 + 2 * s * Y + Y, F)  # Assuming y(0) = 0 and y'(0) = 0 for zero state response
    print("Laplace transform of the differential equation:")
    sp.pprint(ode_laplace)
    print()
    
    # Solve for Y(s)
    Y_sol = sp.solve(ode_laplace, Y)[0]
    print("Solving for Y(s):")
    sp.pprint(Y_sol)
    print()
    
    # Take the inverse Laplace transform to find y(t)
    y_sol = sp.inverse_laplace_transform(Y_sol, s, t)
    print("Inverse Laplace transform to find y(t):")
    sp.pprint(y_sol)
    print("\nThe zero state solution to the ODE is:")
    sp.pprint(y_sol)

def main():
    while True:
        print('(1) First Degree ODE')
        print('(2) Second Degree ODE')
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