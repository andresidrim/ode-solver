# Authors: André Sidrim, Fernanda Panzera

import os

import secondOrderODE as secondOrder
import firstOrderODE as firstOrder

def main():
    while True:
        print("Is your ODE:\n")
        print("(1) First Order ODE")
        print("(2) Second Order ODE")

        firstChoice = input()

        os.system('cls')

        if int(firstChoice) in [1, 2]:
            break

    while True:
        print("(1) Solve ODE")
        print("(2) Solve Zero Input")
        print("(3) Solve Zero State")

        secondChoice = input()

        os.system('cls')

        if int(secondChoice) in [1, 2, 3]:
            break

    if int(firstChoice) == 1:
        if int(secondChoice) == 1:
            a = int(input("Enter the coefficient a: "))
            b = int(input("Enter the coefficient b: "))
            y0 = int(input("Enter the initial condition y(0): "))
            firstOrder.solveODE(a, b, y0)
        
        elif int(secondChoice) == 2:
            a = int(input("Enter the coefficient a: "))
            b = int(input("Enter the coefficient b: "))
            y0 = int(input("Enter the initial condition y(0): "))
            firstOrder.solveZeroInput(a, b, y0)

        elif int(secondChoice) == 3:
            a = int(input("Enter the coefficient a: "))
            b = int(input("Enter the coefficient b: "))
            y0 = int(input("Enter the initial condition y(0): "))
            firstOrder.solveZeroState(a, b, y0)

    elif int(firstChoice) == 2:
        if int(secondChoice) == 1:
            a = int(input("Enter the coefficient a: "))
            b = int(input("Enter the coefficient b: "))
            c = int(input("Enter the coefficient c: "))
            y0 = int(input("Enter the initial condition y(0): "))
            dy0 = int(input("Enter the initial condition y'(0): "))
            secondOrder.solveODE(a, b, c, y0, dy0)
        
        elif int(secondChoice) == 2:
            a = int(input("Enter the coefficient a: "))
            b = int(input("Enter the coefficient b: "))
            c = int(input("Enter the coefficient c: "))
            y0 = int(input("Enter the initial condition y(0): "))
            dy0 = int(input("Enter the initial condition y'(0): "))
            secondOrder.solveZeroInput(a, b, c, y0, dy0)

        elif int(secondChoice) == 3:
            a = int(input("Enter the coefficient a: "))
            b = int(input("Enter the coefficient b: "))
            c = int(input("Enter the coefficient c: "))
            y0 = int(input("Enter the initial condition y(0): "))
            dy0 = int(input("Enter the initial condition y'(0): "))
            secondOrder.solveZeroState(a, b, c, y0, dy0)

if __name__ == "__main__":
    main()