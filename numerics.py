"""

Essential functions of numerical methods
__version__: 1.05.24
@author: Anish Sarkar
Dependencies: Numpy, Sympy (optional)

"""

import numpy as np
from typing import Callable

def polynomial(x: float, *args) -> Callable:
    """Generates an n-th order polynomial depending
    on argument list length and evaluates it at x

    Args:
        x (int/float): Value at which polynomial is to be evaluated
        *args (list): Coefficients of the form [a0, a1, a2, a3, ...]
    Returns:
        float: Evaluated polynomial of the form
        a0 + a1*x + a2*x^2 + a3*x^3 + ...
    """
    args,=args
    func: float = 0.0
    for i,coeff in enumerate(args):
        func+=coeff*x**i
        
    return func

def true_err(new: float, old: float) -> float:
    """Calculates true error

    Args:
        new (float/int): New value
        old (float/int): Previous value

    Returns:
        float: True error
    """
    return new - old

def rel_err(new: float, old: float, percent: bool = False) -> float:
    """Calculates relative error

    Args:
        new (float/int): New value
        old (float/int): Previous value
        percent (bool, optional): Returns percentage error. Defaults to False.

    Returns:
        float: Relative error
    """
    return (new - old)*100/new if percent else (new - old)/new

def abs_rel_err(new: float, old: float, percent: bool = False) -> float:
    """Calculates absolute relative error

    Args:
        new (float/int): New value
        old (float/int): Previous value
        percent (bool, optional): Returns percentage error. Defaults to False.

    Returns:
        float: Absolute relative error
    """
    return np.abs((new - old)/new)*100 if percent else np.abs((new - old)/new)

def precision_err(n: int) -> float:
    """Calculates allowed error according to number 
    of precise significant digits required

    Args:
        n (int): Number of significant digits

    Returns:
        float: Estimated error tollerance percentage
    """
    return 0.5*10**(2-n)

def significance(eps: float, percent: bool = True) -> int:
    """Calculates number of correct signficant digits
    according to supplied value of error eps

    Args:
        eps (float): Tollerance
        percent (bool, optional): Tollerance argument passes is percent or not. Defaults to True.

    Returns:
        int: Number of significant digits
    """
    eps: float = np.abs(eps) if percent else np.abs(eps)*100
    
    return np.floor(2-np.log10(2*eps))

def derivative(func: Callable, x: float, h: float = 0.1) -> float:
    """Calculates the derivative for a given point and 
    step size using first principle

    Args:
        func (callable): Function to be differentiated
        x (float/int): Point at which function is differentiated
        h (float): Differnce in x. Defaults to 0.1.

    Returns:
        float: Differeniadted value
    """
    return (func(x+h) - func(x))/h

def precise_derivative(func: Callable, x: float, h: float = 0.1, adjust: float = 10, eps: float = 1.e-6, seed: float = 1.e8, percent: bool = False) -> tuple:
    """Calculates derivative value at x  from 
    first principle for a step size h 
    which gets scaled by 1/h till tollerance 
    eps is reached

    Args:
        func (callable): Function to be differentiated
        x (int/float): Point at which differentiatio is to be done
        h (float, optional): Initial step size. Defaults to 0.1.
        adjust (int/float, optional): Step size adjustment. Defaults to 10.
        eps (float, optional): Tollerance. Defaults to 1.e-6.
        seed (int/float, optional): Guess differrentiated value. Defaults to 1.e8.
        percent (bool, optional): Percantage used for tollerance if True. Defaults to False.

    Returns:
        tuple: Differentiated value, number of itterations taken
    """
    df_old = err = seed
    count: int = 0
    while (err > eps):
        count+=1
        df: float = derivative(func, x, h)
        err: float = abs_rel_err(df, df_old, percent)
        df_old: float = df
        h/=adjust
        
    return df, count

def taylor_series(func: Callable, n: int, x0: float = 0) -> Callable:
    """Evaluates taylor series upto n terms about x0

    Args:
        func (sympy.Expression): Function to be expanded
        n (int): number of terms
        x0 (float, optional): Value about which series is to be expanded. Defaults to 0.

    Returns:
        sympy.Expression: Taylor series expression upto n terms
    """
    import sympy as sy
    
    x: object = sy.symbols("x")
    result: float = 0
    count: int = 0
    i: int = 0
    while(count < n):
        dy = func.subs(x,x0)*(x-x0)**i/np.math.factorial(i)
        result+=dy
        func = sy.diff(func,x)
        if dy != 0:
            count+=1
        i+=1
        
    return result

def divided_diff(x: list[float], y: list[float], *args) -> float:
    """Calculates divided difference value recursively

    Args:
        x (list/numpy/ndarray): x data points
        y (list/numpy.ndarray): y data points

    Returns:
        float: Divided Difference value
    """
    args,=args
    if len(args) == 2:
        return (y[args[1]] - y[args[0]])/(x[args[1]] - x[args[0]])
        
    diff: float = (divided_diff(x, y, args[:-1]) - divided_diff(x, y, args[1:]))/\
        (x[args[0]]-x[args[-1]])
        
    return diff

def poly_interpolation(xdata: list[float], ydata: list[float], x: float, deg: int, shift: int = 0) -> np.ndarray:
    """Interpolates at a point from supplied data using nth degree polynomial

    Args:
        xdata (list/numpy.ndarray): List of x points
        ydata (list/numpy.ndarray): List of y points
        x (float/int): Point at which interpolation is to be done
        deg (int): Degree of polynomial
        shift (int): Shifts the index of data point by n

    Returns:
        numpy.ndarray: Array of coefficient values
    """
    if not isinstance(xdata, np.ndarray):
        xdata: np.ndarray = np.array(xdata)
    if not isinstance(ydata, np.ndarray):
        ydata: np.ndarray = np.array(ydata)
    if not isinstance(xdata, np.ndarray):
        x: np.ndarray = np.array(x)
        
    index = np.max(np.where(xdata<=x)) - shift
    X: np.ndarray = np.zeros((deg+1, deg+1))
    B: np.ndarray = np.zeros(deg+1)
    power: np.ndarray = np.arange(0,deg+1)

    for i,_ in enumerate(X):
        X[i] = xdata[index+i]**power
        B[i] = ydata[index+i]
        
    soln: np.ndarray = np.linalg.solve(X, B)
    
    return soln

def linear_spline_interpolation(xdata: list[float], ydata: list[float], x: float) -> np.ndarray:
    """Function tthat uses linear spline interpolation to interpolate supplied data

    Args:
        xdata (list/np.ndarray): List of x points
        ydata (list/np.ndarray): List of y points
        x (int/float): Point to be interpolated at

    Returns:
        np.ndarray: Array of coefficinets at each spline
    """
    if not (isinstance(xdata, np.ndarray)):
        xdata: np.ndarray = np.array(xdata)
    if not (isinstance(ydata, np.ndarray)):
        ydata: np.ndarray = np.array(ydata)
    if not (isinstance(xdata, np.ndarray)):
        x: np.ndarray = np.array(x)
    
    coeff: list[float] = []
    
    for index,_ in enumerate(xdata[:len(xdata)-1]):
        X: np.ndarray = np.array([[1, xdata[index]],
                    [1, xdata[index+1]]])
        B: np.ndarray = np.array([ydata[index],
                    ydata[index+1]])
        soln: np.ndarray = np.linalg.solve(X, B)
        coeff.append(soln)
    
    return np.array(coeff)

def quadratic_spline_interpolation(xdata: list[float], ydata: list[float], x: float) -> np.ndarray:
    """Function tthat uses quadratic spline interpolation to interpolate supplied data

    Args:
        xdata (list/np.ndarray): List of x points
        ydata (list/np.ndarray): List of y points
        x (int/float): Point to be interpolated at

    Returns:
        np.ndarray: Array of coefficinets at each spline
    """
    if not (isinstance(xdata, np.ndarray)):
        xdata: np.ndarray = np.array(xdata)
    if not (isinstance(ydata, np.ndarray)):
        ydata: np.ndarray = np.array(ydata)
    if not (isinstance(xdata, np.ndarray)):
        x: np.ndarray = np.array(x)
        
    lenx: int = len(xdata)-1
    
    X: np.ndarray = np.zeros((3*lenx, 3*lenx))
    B: np.ndarray = np.zeros(3*lenx)
    X[0][0] = xdata[0]*xdata[0]
    X[0][1] = xdata[0]
    X[0][2] = 1
    X[2*lenx-1][3*lenx-3] = xdata[lenx]*xdata[lenx]
    X[2*lenx-1][3*lenx-2] = xdata[lenx]
    X[2*lenx-1][3*lenx-1] = 1
    X[len(X)-1][0] = 1
    for i in range(lenx-1):
        xval = xdata[i+1]
        X[2*i+1][3*i] = xval*xval
        X[2*i+1][3*i+1] = xval
        X[2*i+1][3*i+2] = 1
        X[2*i+2][3*i+3] = xval*xval
        X[2*i+2][3*i+4] = xval
        X[2*i+2][3*i+5] = 1
    for i in range(lenx-1):
        xval = xdata[i+1]
        X[2*lenx+i][3*i] = 2*xval
        X[2*lenx+i][3*i+1] = 1
        X[2*lenx+i][3*i+3] = -2*xval
        X[2*lenx+i][3*i+4] = -1
        
    B[0] = ydata[0]
    B[2*lenx-1] = ydata[lenx]
    
    for i in range(lenx-1):
        B[2*i+1] = ydata[i+1]
        B[2*i+2] = ydata[i+1]  
    
    coeff: np.ndarray = np.linalg.solve(X,B)
    coeff = np.reshape(coeff, (int(len(coeff)/3),3))
    
    return coeff

def nddp(xdata: list[float], ydata: list[float], x: float, order: int, shift: int = 0) -> float:
    """Interpolation function using Newton Divided Difference Polynomial

    Args:
        xdata (list/numpy.ndarray): x data points
        ydata (list/numpy.ndarray): y data points
        x (int/float): Interpolation point
        order (int): Degree of polynomial
        shift (int): Shifts the index of data point by n. Defaults to 0.

    Returns:
        float: Interpolated value
    """
    if not isinstance(xdata, np.ndarray):
        xdata: np.ndarray = np.array(xdata)
    if not isinstance(ydata, np.ndarray):
        ydata: np.ndarray = np.array(ydata)
    if not isinstance(xdata, np.ndarray):
        x: np.ndarray = np.array(x)
    
    index: int = np.max(np.where(xdata<=x)) - shift
    val: float = ydata[index]
    for i in range(order):
        arr: np.ndarray = np.arange(index, index+i+2)
        val+=divided_diff(xdata, ydata, arr[::-1].tolist())*\
            np.prod(x-xdata[index:index+i+1])
    
    return val

def euler_first(func: Callable, h: float, x_stop: float, y_start: float, x_start: float) -> np.ndarray:
    """First order Euler method solver

    Args:
        func (callable): dy/dx = func
        h (float): Step size
        x_stop (int/float): x stopping point
        y_start (int/float): y boundary value
        x_start (int/float): x boundary value

    Returns:
        numpy.ndarray: Returns array of y values till x_stop
    """
    x: np.ndarray = np.arange(x_start, x_stop+h, h)
    y: np.ndarray = np.zeros(len(x))
    y[0] = y_start
    for i,val in enumerate(x[1:]):
        y[i+1] = y[i] + func(val, y[i])*h
    
    return y

def rk_2(func: Callable, h: float, x_stop: float, y_start: float, x_start: float, method: str = 'heun') -> np.ndarray:
    """2nd order Range-Kutta integrator

    Args:
        func (callable): Function to be solved
        h (float): Step size
        x_stop (int/float): x topping point
        y_start (int/float): y boundary value/inital value
        x_start (int/float): x boundary value/initial value
        method (string): heun/midpoint/ralston

    Returns:
        numpy.ndarray: Integrated values
    """
    x: np.ndarray = np.arange(x_start, x_stop+h, h)
    y: np.ndarray = np.zeros(len(x))
    y[0] = y_start
    if method == "heun":
        a1, a2, b1, c1 = 0.5, 0.5, 1, 1
    elif method == "midpoint":
        a1, a2, b1, c1 = 0, 1, 0.5, 0.5
    elif method == "ralston":
        a1, a2, b1, c1 = 1/3, 2/3, 3/4, 3/4
    for i,val in enumerate(x[1:]):
        k1: float = func(val, y[i])
        k2: float = func(val+b1*h, y[i]+c1*k1*h)
        y[i+1] = y[i] + (a1*k1 + a2*k2)*h
            
    return y

def rk_4(func: Callable, h: float, x_stop: float, y_start: float, x_start: float) -> np.ndarray:
    """4th order Range-Kutta integrator

    Args:
        func (callable): Function to be solved
        h (float): Step size
        x_stop (int/float): x topping point
        y_start (int/float): y boundary value/inital value
        x_start (int/float): x boundary value/initial value
        method (string): heun/midpoint/ralston

    Returns:
        numpy.ndarray: Integrated values
    """
    x: np.ndarray = np.arange(x_start, x_stop+h, h)
    y: np.ndarray = np.zeros(len(x))
    y[0] = y_start
    for i,val in enumerate(x[1:]):
        k1: float = func(val, y[i])
        k2: float = func(val+0.5*h, y[i]+0.5*k1*h)
        k3: float = func(val+0.5*h, y[i]+0.5*k2*h)
        k4: float = func(val+h, y[i]+k3*h)
        y[i+1] = y[i] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)*h
            
    return y

def trapezoidal(func: Callable, N: int, x_start: float, x_stop: float) -> float:
    """Trapezoidal integrator

    Args:
        func (callable): Function to be integrated
        N (int): Number of divisions
        x_start (int/float): Integration starting boundary
        x_stop (int/float): Integration ending boundary

    Returns:
        float: Integrated result
    """
    h: float = (x_stop-x_start)/N
    x: np.ndarray = np.arange(x_start+h, x_stop, h)
    result: float = 0.5*h*(func(x_start) + func(x_stop) + 2*np.sum(func(x)))
            
    return result

def simpsons1_3(func: Callable, N: int, x_start: float, x_stop: float) -> float:
    """Simpsons 1/3 integrator

    Args:
        func (callable): Function to be integrated
        N (int): Number of divisions
        x_start (int/float): Integration starting boundary
        x_stop (int/float): Integration ending boundary

    Returns:
        float: Integrated result
    """
    h: float = (x_stop-x_start)/N
    x: np.ndarray = np.arange(x_start+h, x_stop, h)
    result: float = (1/3)*h*(func(x_start) + func(x_stop) + 
            4*np.sum(func(x[::2])) + 2*np.sum(func(x[1::2])))
            
    return result

def simpsons3_8(func: Callable, N: int, x_start: float, x_stop: float) -> float:
    """Simpsons 3/8 integrator

    Args:
        func (callable): Function to be integrated
        N (int): Number of divisions
        x_start (int/float): Integration starting boundary
        x_stop (int/float): Integration ending boundary

    Returns:
        float: Integrated result
    """
    h: float = (x_stop-x_start)/N
    x: np.ndarray = np.arange(x_start+h, x_stop, h)
    result: float = (3/8)*h*(func(x_start) + func(x_stop) + 
            3*np.sum(func(x[::3])) + 3*np.sum(func(x[1::3])) + 
            2*np.sum(func(x[2::3])))
            
    return result

def poly_regression(x: list[float], y: list[float], deg: int) -> np.ndarray:
    """Polynomial regression function

    Args:
        x (list/numpy.ndarray): List of x data points
        y (list/numpy.ndarray): list of y data points
        deg (int): Degree of polynomial

    Returns:
        numpy.ndarray: Array of parameters [a0, a1, a2, ...]
        of the form a0 + a1*x + a2*x^2 + ...
    """
    if not isinstance(x, np.ndarray):
        x: np.ndarray = np.array(x)
    if not isinstance(y, np.ndarray):
        y: np.ndarray = np.array(y)
    X: np.ndarray = np.zeros((deg+1,deg+1))
    B: np.ndarray = np.zeros(deg+1)
    for i,_ in enumerate(X):
        for j,_ in enumerate(X):
            X[i][j] = np.sum(x**(j+i))
    for i,_ in enumerate(B):
        B[i] = np.sum(y*x**i)
    
    result: np.ndarray = np.linalg.solve(X,B)
    
    return result

def exp_regression(x: list[float], y: list[float]) -> tuple:
    """Exponential regression function

    Args:
        x (list/numpy.ndarray): List of x data points
        y (list/numpy.ndarray): list of y data points

    Returns:
        tuple: Tuple of parameters [a0, a1]
        of the form a0*e^(x*a1)
    """
    if not isinstance(x, np.ndarray):
        x: np.ndarray = np.array(x)
    if not isinstance(y, np.ndarray):
        y: np.ndarray = np.array(y)
    a0, a1 = poly_regression(x, np.log(y), 1)
    
    return np.exp(a0),a1

def pow_regression(x: list[float], y: list[float]) -> tuple:
    """Power law regression function

    Args:
        x (list/numpy.ndarray): List of x data points
        y (list/numpy.ndarray): list of y data points

    Returns:
        tuple: Tuple of parameters [a0, a1]
        of the form a0*x^a1
    """
    if not isinstance(x, np.ndarray):
        x: np.ndarray = np.array(x)
    if not isinstance(y, np.ndarray):
        y: np.ndarray = np.array(y)
    a0, a1 = poly_regression(np.log(x), np.log(y), 1)
    
    return np.exp(a0),a1

def bisection(func: Callable, left: float, right: float, eps_x: float = 0, eps_f: float = 0) -> tuple:
    """Bisection root finding algorithm

    Args:
        func (callable): Function whose root is to be found
        left (int/float): Left boundary value of interval
        right (int/float): Right boundary value of
        eps_x (float, optional): Tollerance on x. Defaults to 0.
        eps_f (float, optional): Tollerance on function. Defaults to 0.

    Raises:
        Exception: Function may have 0 or even number of roots

    Returns:
        tuple: x value, Function value at x, number of itterations taken
    """
    if func(left)*func(right) < 0:
        x: np.ndarray = np.array([left, right], dtype=np.float64)
        c_old: float = np.mean(x)
        old: float = func(c_old)
        if (np.abs(old)<eps_f):
            root: float = c_old
            
            return root, func(root), 0
        err: float = 100
        count: int = 0
        while(np.abs(func(c_old))>eps_f or err>eps_x):
            count+=1
            x = np.insert(x, 1, c_old)
            if np.prod(func(x[:2])) < 0:
                x = x[:2]
            else:
                x = x[1:]
            c_new: float = np.mean(x)
            err = abs_rel_err(c_new, c_old)
            c_old = c_new
            
        return c_new, func(c_new), count
    raise Exception("Function does not have any root in the given interval or has even number of roots")
        
def newton_raphson(func: Callable, dfunc: Callable, guess: float, eps_x: float = 0, eps_f: float = 0) -> tuple:
    """Newton Raphson root finding algorithm

    Args:
        func (callable): Function whose root is to be found
        dfunc (callable): Differentiated function
        guess (int/float): Initial guess
        eps_x (float, optional): Tollerance on x. Defaults to 0.
        eps_f (float, optional): Tollerance on function. Defaults to 0

    Returns:
        tuple: x value, Function value at x, number of itterations taken
    """
    x_old: float = guess
    if np.abs(func(x_old)) < eps_f:
        return x_old, func(x_old), 0
    err: float = 100
    count: int = 0
    while(np.abs(func(x_old))>eps_f or err>eps_x):
        count+=1
        x_new = x_old - func(x_old)/dfunc(x_old)
        err = abs_rel_err(x_new, x_old)
        x_old = x_new
        
    return x_new, func(x_new), count

def secant(func: Callable, guess_1: float, guess_2: float, eps_x: float = 0, eps_f: float = 0) -> tuple:
    """Secant root finding algorithm

    Args:
        func (callable): Function whose root is to be found
        guess_1 (int/float): Initial guess 1
        guess_2 (int/float): Initial guess 2
        eps_x (float, optional): Tollerance on x. Defaults to 0.
        eps_f (float, optional): Tollerance on function. Defaults to 0

    Returns:
        tuple: x value, Function value at x, number of itterations taken
    """
    x_1: float = guess_1
    x_2old: float = guess_2
    if (np.abs(func(x_1)) < eps_f):
        return x_1, func(x_1), 0
    elif (np.abs(func(x_2old)) < eps_f):
        return x_2old, func(x_2old), 0
    err: float = 100
    count: int = 0
    while(np.abs(func(x_2old))>eps_f or err>eps_x):
        count+=1
        x_2new = x_2old - func(x_2old)*(x_2old-x_1)/(func(x_2old)-func(x_1))
        err = abs_rel_err(x_2new, x_2old)
        x_1 =x_2old
        x_2old = x_2new
        
    return x_2new, func(x_2new), count

if __name__=="__main__":
    print("Executed when invoked directly")