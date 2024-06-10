import sympy as sp
from scipy.integrate import solve_ivp

def solveZeroState(a, b, c, y0, dy0):
    x = sp.symbols('x')
    y = sp.Function('y')
    
    ode = a*sp.diff(y(x), x, x) + b*sp.diff(y(x), x) + c*y(x)
    
    sol = sp.dsolve(ode, y(x))
    
    gen_sol = sol.rhs
    
    C1, C2 = sp.symbols('C1 C2')
    
    particular_sol = gen_sol.subs(sp.symbols('C1'), C1).subs(sp.symbols('C2'), C2)
    
    eq1 = sp.Eq(particular_sol.subs(x, 0), y0)
    eq2 = sp.Eq(sp.diff(particular_sol, x).subs(x, 0), dy0)
    
    constants = sp.solve((eq1, eq2), (C1, C2))

    print("C1 = ", constants[C1])
    print("C2 = ", constants[C2])
    
    particular_sol = gen_sol.subs(constants)

    print("Solução geral:")
    sp.pprint(sol)

    print("\nSolução particular com condições iniciais:")
    sp.pprint(particular_sol)

def model(t, y, a, b, c):
    dydt = y[1]
    dzdt = -(b * y[1] + c * y[0]) / a
    return [dydt, dzdt]

def solveODE(a, b, c, y0, yd0):
    initial_conditions = [y0, yd0]
    time_span = [0, 0]

    sol = solve_ivp(model, time_span, initial_conditions, args=(a, b, c), dense_output=True)

    t = sol.t
    y = sol.y[0]

    tSymbol = sp.symbols('t')
    ySymbol = sp.Function('y')(tSymbol)
    equation = sp.Eq(a * ySymbol.diff(tSymbol, tSymbol) + b * ySymbol.diff(tSymbol) + c * ySymbol, 0)

    general_solution = sp.dsolve(equation)
    constants = sp.symbols('C1 C2')

    eq1 = general_solution.rhs.subs(tSymbol, 0) - y0
    eq2 = sp.diff(general_solution.rhs, tSymbol).subs(tSymbol, 0) - yd0

    constants_solution = sp.solve((eq1, eq2), constants)
    particular_solution = general_solution.subs(constants_solution)

    print("Numerical solution values:")
    for ti, yi in zip(t, y):
        print(f"t = {ti:.2f}, y(t) = {yi:.2f}")

    print("\nGeneral solution (symbolic):")
    sp.pprint(general_solution)

    print("\nValues of C1 and C2 (from symbolic solution):")
    for constant, value in constants_solution.items():
        print(f"{constant} = {value.evalf()}")

    print("\nParticular solution with constants:")
    sp.pprint(particular_solution)
    
def solveZeroInput(a, b, c, y0, dy0):
    t = sp.symbols('t')
    y = sp.Function('y')(t)
    
    ode = sp.Eq(a * y.diff(t, t) + b * y.diff(t) + c * y, 0)  
    print("The differential equation is:")
    sp.pprint(ode)
    print()
    
    general_solution = sp.dsolve(ode)
    print("The general solution to the ODE is:")
    sp.pprint(general_solution)
    print()
    
    constants_solution = sp.solve((general_solution.rhs.subs(t, 0) - y0, sp.diff(general_solution.rhs, t).subs(t, 0) - dy0), dict=True)
    
    if constants_solution:
        constants_solution = constants_solution[0]
        particular_solution = general_solution.subs(constants_solution)
        print("The particular solution to the ODE with the given initial conditions is:")
        sp.pprint(particular_solution)
    else:
        print("Error: Unable to find a solution for the constants.")
