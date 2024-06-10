import sympy as sp
from scipy.integrate import solve_ivp

def model(t, y, a, b):
    dydt = -(b/a) * y
    return dydt

def solveODE(a, b, y0):
    initial_conditions = [y0]
    time_span = [0, 0]  
    
    sol = solve_ivp(lambda t, y: model(t, y, a, b), time_span, initial_conditions, dense_output=True)

    t = sol.t
    y = sol.y[0]
    
    tSymbol = sp.symbols('t')
    ySymbol = sp.Function('y')(tSymbol)
    equation = sp.Eq(a * ySymbol.diff(tSymbol) + b * ySymbol, 0)
    
    general_solution = sp.dsolve(equation)
    C1 = sp.symbols('C1')
    
    particular_solution = general_solution.subs(C1, y0)
    
    print("Solução numérica:")
    for ti, yi in zip(t, y):
        print(f"t = {ti:.2f}, y(t) = {yi:.2f}")

    print("\nSolução geral (simbólica):")
    sp.pprint(general_solution)

    print("\nSolução particular com condição inicial:")
    sp.pprint(particular_solution)

def solveZeroState(a, b, y0):   
    x = sp.symbols('x')
    y = sp.Function('y')
    
    ode = a*sp.diff(y(x), x) + b*y(x)
    
    sol = sp.dsolve(ode, y(x))
    
    gen_sol = sol.rhs
    
    C1 = sp.symbols('C1')
    
    particular_sol = gen_sol.subs(sp.symbols('C1'), C1)
    
    eq = sp.Eq(particular_sol.subs(x, 0), y0)
    
    constant = sp.solve(eq, C1)[0]

    print("C = ", constant)
    
    particular_sol = gen_sol.subs(C1, constant)

    print("Solução geral:")
    sp.pprint(sol)

    print("\nSolução particular com condição inicial:")
    sp.pprint(particular_sol)

def solveZeroInput(a, b, y0):
    t = sp.symbols('t')
    y = sp.Function('y')(t)
    
    ode = sp.Eq(a * y.diff(t) + b * y, 0)  
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
