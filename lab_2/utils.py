import math
import numpy as np
from scipy.optimize import minimize

def array(f, numval, numdh):
    """Создать N-мерный массив.
    
    param: f - функция, которая приминает N аргументов.
    param: numval - диапазоны значений параметров функции. Список
    param: numdh - шаги для параметров. Список
    
    """
    def rec_for(f, numdim, numdh, current_l, l_i, arr):
        """Рекурсивный цикл.
        
        param: f - функция, которая приминает N аргументов.
        param: numdim - размерность выходной матрицы. Список
        param: numdh - шаги для параметров. Список
        param: current_l - текущая глубина рекурсии.
        param: l_i - промежуточный список индексов. Список
        param: arr - матрица, с которой мы работаем. np.array
        
        """
        for i in range(numdim[current_l]):
            l_i.append(i)
            if current_l < len(numdim) - 1:
                rec_for(f, numdim, numdh, current_l + 1, l_i, arr)
            else:
                args = (np.array(l_i) * np.array(numdh))
                arr[tuple(l_i)] = f(*args)
            l_i.pop()
        return arr
    numdim = [int(numval[i] / numdh[i]) + 1 for i in range(len(numdh))]
    arr = np.zeros(numdim)
    arr = rec_for(f, numdim, numdh, 0, [], arr)
    
    # Надо отобразить так x - j, y - i (для графиков), поэтому используем transpose
    
    arr = np.transpose(arr)
    return arr

def TDMA(a, b, c, f):
    """Метод прогонки.
    
    param: a - левая поддиагональ. 
    param: b - правая поддиагональ.
    param: c - центр.
    param: f - правая часть.
    """
    
    alpha = [0]
    beta = [0]
    n = len(f)
    x = [0] * n

    for i in range(n - 1):
        alpha.append(-b[i] / (a[i] * alpha[i] + c[i]))
        beta.append((f[i] - a[i] * beta[i]) / (a[i] * alpha[i] + c[i]))

    x[n - 1] = (f[n - 1] - a[n - 1] * beta[n - 1]) / (c[n - 1] + a[n - 1] * alpha[n - 1])

    for i in reversed(range(n - 1)):
        x[i] = alpha[i + 1] * x[i + 1] + beta[i + 1]

    return x

def integral(arr, dh):
    val = 0.
    for i in range(0, arr.shape[0] - 1):
        val += arr[i] + arr[i + 1]
        
    return val * dh / 2.

def integral_2(matr, dh, dt):
    val = 0.
    buf = np.zeros(matr.shape[0])
    for j in range(0, matr.shape[0]):
        buf[j] = integral(matr[j,:], dh)
        
    val = integral(buf, dt)
    
    return val

#----------------------------------------------------------------------------------------------------------------------

def criterion_1(model):
    val = 0.
    dh, dt = model.dh, model.dt
    
    # Вычисление нормы разности
    
    if len(model.f_arr) == 1:
        val = 10000000.
    else:
        matr = (model.f_arr[-1] - model.f_arr[-2]) ** 2
        val = abs(integral_2(matr, dh, dt))
        
    return val

def criterion_2(model):
    val_1, val_2 = 0., 0.
    val = 0.
    dh = model.dh
    
    # Вычисление нормы разности функционала
    
    if len(model.x_arr) == 1:
        val = 10000000.
    else:
        arr_1 = (model.x_arr[-1][-1,:] - model.y_arr) ** 2
        arr_2 = (model.x_arr[-2][-1,:] - model.y_arr) ** 2
        val_1 = integral(arr_1, dh)
        val_2 = integral(arr_2, dh)
        val = abs(val_1 - val_2) 
        
    return val

def criterion_3(model):
    val = 0.
    dh, dt = model.dh, model.dt
    
    # Вычисление нормы производной
    
    if len(model.psi_arr) == 1:
        val = 10000000.
    else:
        matr = -model.psi_arr[-1]
        val = abs(integral_2(matr, dh, dt))
        
    return val
#----------------------------------------------------------------------------------------------------------------------

# 1 задача
def solve_1(model, ind, f_arr=None):
    val = model.x_arr[0].copy()
    if f_arr is None:
        f_arr = model.f_arr[-1]
    
    # j == 0
    val[0,:] = model.fi_0_arr

    # j == 1
    val[1,:] = model.fi_0_arr +\
               model.dt * model.fi_1_arr +\
               model.dt ** 2 / 2. * (model.d2fi_0_arr * model.a ** 2 + f_arr[0,:])

    # j > 1
    for j in range(2, model.M + 1):

        # Правая часть 

        G_0_j = (2. * val[j - 1, 0] - val[j - 2, 0] +\
                 2. * model.const_2 * (val[j - 1, 1] - val[j - 1, 0]) +\
                 2. * model.const_1 * (val[j - 2, 1] - val[j - 2, 0]) +\
                 f_arr[j - 1, 0] * model.dt ** 2) / model.const_3

        p_j = 2. * model.p_arr[j - 1] * model.dt ** 2 / (model.dh * model.const_3)

        G_N_j = (2. * val[j - 1, -1] - val[j - 2, -1] +\
                 2. * model.const_2 * (val[j - 1, -2] - val[j - 1, -1]) +\
                 2. * model.const_1 * (val[j - 2, -2] - val[j - 2, -1]) +\
                 f_arr[j - 1, -1] * model.dt ** 2) / model.const_3

        f = [2. * val[j - 1, 1] - val[j - 2, 1] +\
             model.const_2 * (val[j - 1, 0] - 2. * val[j - 1, 1] + val[j - 1, 2]) +\
             model.const_1 * (val[j - 2, 0] - 2. * val[j - 2, 1] + val[j - 2, 2]) +\
             f_arr[j - 1, 1] * model.dt ** 2 +\
             model.const_1 * G_0_j -\
             model.const_1 * p_j
            ]

        f += [2. * val[j - 1, i] - val[j - 2, i] +\
              model.const_2 * (val[j - 1, i - 1] - 2. * val[j - 1, i] + val[j - 1, i + 1]) +\
              model.const_1 * (val[j - 2, i - 1] - 2. * val[j - 2, i] + val[j - 2, i + 1]) +\
              f_arr[j - 1, i] * model.dt ** 2
                for i in range(2, model.eq_l + 1)
            ]

        f[-1] += model.const_1 * G_N_j

        # Решение СЛАУ
        val[j, 1:1 + model.eq_l] = TDMA(model.a_arr, model.b, model.c, f)

        # Краевые условия
        val[j, 0] = 2. * model.const_1 / model.const_3 * val[j, 1] + G_0_j - p_j
        val[j, -1] = 2. * model.const_1 / model.const_3 * val[j, -2] + G_N_j
                
    return val

# 2 задача
def solve_2(model, ind, x_arr=None):
    # 2 задача
    val = model.x_arr[0].copy()
    if x_arr is None:
        x_arr = model.x_arr[-1]
    
    # j == M
    val[-1,:] = 0.
            
    # j == M - 1
    val[-2,:] = model.dt * 2. * (x_arr[-1,:] - model.y_arr)
            
    # j < M - 1
    for j in range(model.M - 2, -1, -1):
                
        # Правая часть 

        G_0_j = (2. * val[j + 1, 0] - val[j + 2, 0] +\
                 2. * model.const_2 * (val[j + 1, 1] - val[j + 1, 0]) +\
                 2. * model.const_1 * (val[j + 2, 1] - val[j + 2, 0])) / model.const_3

        G_N_j = (2. * val[j + 1, -1] - val[j + 2, -1] +\
                 2. * model.const_2 * (val[j + 1, -2] - val[j + 1, -1]) +\
                 2. * model.const_1 * (val[j + 2, -2] - val[j + 2, -1])) / model.const_3

        f = [2. * val[j + 1, 1] - val[j + 2, 1] +\
             model.const_2 * (val[j + 1, 0] - 2. * val[j + 1, 1] + val[j + 1, 2]) +\
             model.const_1 * (val[j + 2, 0] - 2. * val[j + 2, 1] + val[j + 2, 2]) +\
             model.const_1 * G_0_j
            ]

        f += [2. * val[j + 1, i] - val[j + 2, i] +\
              model.const_2 * (val[j + 1, i - 1] - 2. * val[j + 1, i] + val[j + 1, i + 1]) +\
              model.const_1 * (val[j + 2, i - 1] - 2. * val[j + 2, i] + val[j + 2, i + 1])
                for i in range(2, model.eq_l + 1)
            ]

        f[-1] += model.const_1 * G_N_j
                
        # Решение СЛАУ
        val[j, 1:1 + model.eq_l] = TDMA(model.a_arr, model.b, model.c, f)
                
        # Краевые условия
        val[j, 0] = 2. * model.const_1 / model.const_3 * val[j, 1] + G_0_j
        val[j, -1] = 2. * model.const_1 / model.const_3 * val[j, -2] + G_N_j 
        
    return val

#----------------------------------------------------------------------------------------------------------------------

def f_alpha(alpha, model, ind):
    val = .0
    ing_2 = integral_2(model.psi_arr[-1] * model.psi_arr[-1], model.dh, model.dt) ** .5
    if ing_2 == 0.:
        buf = 0.
    else:
        buf = -1. * alpha * model.R_1 * model.psi_arr[-1] / ing_2
    f_arr = model.f_arr[-1] * (1. - alpha) + buf
    
    matr = solve_1(model, ind, f_arr=f_arr)
        
    arr = (matr[-1,:] - model.y_arr) ** 2
    val = integral(arr, model.dh)
    
    return val

def get_alpha_1(model, ind):
    val = 0.
    bnds = ((0, None),)
    res = minimize(f_alpha, 1., args=(model, ind), bounds=bnds, tol=10**-8)
    val = res.x
    
    return val    
def get_alpha_2(model, ind):
    val = 1.
    arr = (model.x_arr[-1][-1,:] - model.y_arr) ** 2
    f_1 = integral(arr, model.dh)
    f_2 = f_1
    while f_2 >= f_1:
        ing_2 = integral_2(model.psi_arr[-1] * model.psi_arr[-1], model.dh, model.dt) ** .5
        if ing_2 == 0.:
            buf = 0.
        else:
            buf = -1. * val * model.R_1 * model.psi_arr[-1] / ing_2
        val /= 2.
        f_arr = model.f_arr[-1] * (1. - val) + buf
        x_arr = solve_1(model, ind, f_arr=f_arr)
        arr = (x_arr[-1,:] - model.y_arr) ** 2
        f_2 = integral(arr, model.dh)
        
    return val

def get_alpha_4(model, ind):
    l, eps, i = .5, 10 ** -10, 0
    val = l ** i
    flag = True
    arr = (model.x_arr[-1][-1,:] - model.y_arr) ** 2
    f_1 = integral(arr, model.dh)
    while flag:
        ing_2 = integral_2(model.psi_arr[-1] * model.psi_arr[-1], model.dh, model.dt) ** .5
        if ing_2 == 0.:
            buf = 0.
        else:
            buf = -1. * val * model.R_1 * model.psi_arr[-1] / ing_2
        f_arr = model.f_arr[-1] * (1. - val) + buf
        x_arr = solve_1(model, ind, f_arr=f_arr)
        arr = (x_arr[-1,:] - model.y_arr) ** 2
        f_2 = integral(arr, model.dh)
        psi_arr = solve_2(model, ind, x_arr=x_arr)
        bound = val * eps * abs(integral_2(-psi_arr * (buf - model.psi_arr[-1]), model.dh, model.dt))
        flag = (f_1 - f_2) <= bound
        i += 1
        val = l ** i
  
    return val

def get_alpha_4_1(model, ind):
    l, eps, i = .25, 10 ** -10, 0
    val = l ** i
    flag = True
    arr = (model.x_arr[-1][-1,:] - model.y_arr) ** 2
    f_1 = integral(arr, model.dh)
    while flag:
        ing_2 = integral_2(model.psi_arr[-1] * model.psi_arr[-1], model.dh, model.dt) ** .5
        if ing_2 == 0.:
            buf = 0.
        else:
            buf = -1. * val * model.R_1 * model.psi_arr[-1] / ing_2
        f_arr = model.f_arr[-1] * (1. - val) + buf
        x_arr = solve_1(model, ind, f_arr=f_arr)
        arr = (x_arr[-1,:] - model.y_arr) ** 2
        f_2 = integral(arr, model.dh)
        psi_arr = solve_2(model, ind, x_arr=x_arr)
        bound = val * eps * abs(integral_2(-psi_arr * (buf - model.psi_arr[-1]), model.dh, model.dt))
        flag = (f_1 - f_2) <= bound
        i += 1
        val = l ** i
  
    return val

def get_alpha_4_2(model, ind):
    l, eps, i = .5, 10 ** -5, 0
    val = l ** i
    flag = True
    arr = (model.x_arr[-1][-1,:] - model.y_arr) ** 2
    f_1 = integral(arr, model.dh)
    while flag:
        ing_2 = integral_2(model.psi_arr[-1] * model.psi_arr[-1], model.dh, model.dt) ** .5
        if ing_2 == 0.:
            buf = 0.
        else:
            buf = -1. * val * model.R_1 * model.psi_arr[-1] / ing_2
        f_arr = model.f_arr[-1] * (1. - val) + buf
        x_arr = solve_1(model, ind, f_arr=f_arr)
        arr = (x_arr[-1,:] - model.y_arr) ** 2
        f_2 = integral(arr, model.dh)
        psi_arr = solve_2(model, ind, x_arr=x_arr)
        bound = val * eps * abs(integral_2(-psi_arr * (buf - model.psi_arr[-1]), model.dh, model.dt))
        flag = (f_1 - f_2) <= bound
        i += 1
        val = l ** i
  
    return val

def get_alpha_5(model, ind):
    val = 0.
    c, alpha = 1., 3./4.
    
    # Вычисление коэффициента
    
    val = c * (float(ind) + 1.) ** -alpha
    
    return val

def get_alpha_5_1(model, ind):
    val = 0.
    c, alpha = 10., 3./4.
    
    # Вычисление коэффициента
    
    val = c * (float(ind) + 1.) ** -alpha
    
    return val

#-----------------------------------------------------------------------------

# Класс модели для Л.Р №2
class Lab2OptCtrlModel():
    
    def __init__(self, p_d):
        
        self.a, self.l, self.T = p_d['a'], p_d['l'], p_d['T']
        self.p, self.f = p_d['p(t)'], p_d['f(s, t)']
        self.R_0, self.R_1 = p_d['R_0'], p_d['R_1']
        self.y = p_d['y(s)']
        
        self.fi_0, self.fi_1, self.d2fi_0 = p_d['fi_0(s)'], p_d['fi_1(s)'], p_d['d2fi_0(s)']
        
        self.dh, self.dt = p_d['dh'], p_d['dt']
        self.N, self.M = p_d['N'], p_d['M']
        
        self.sigma = p_d['sigma']
        
        self.p_arr = array(self.p, [p_d['T']], [p_d['dt']])
        
        self.f_arr = []
        self.f_arr.append(array(self.f, [p_d['l'], p_d['T']], [p_d['dh'], p_d['dt']]))
        
        self.fi_0_arr = array(self.fi_0, [p_d['l']], [p_d['dh']])
        self.fi_1_arr = array(self.fi_1, [p_d['l']], [p_d['dh']])
        self.d2fi_0_arr = array(self.d2fi_0, [p_d['l']], [p_d['dh']])
        
        self.x_arr = []
        self.x_arr.append(array(lambda s, t: 0., [p_d['l'], p_d['T']], [p_d['dh'], p_d['dt']]))
        
        self.psi_arr = []
        self.psi_arr.append(array(self.f, [self.l, self.T], [self.dh, self.dt]))
        
        self.x_arr[0][0,:] = array(self.fi_0, [p_d['l']], [p_d['dh']])
        
        self.y_arr = array(self.y, [self.l], [self.dh])
        
        self.alpha = []
        self.final_step = 0
        self.err = []
        
    def solve(self, criterion, get_alpha, eps=10**-2, max_steps=None):
        
        self.eps = eps
        
        # Число уравнений
        self.eq_l = self.N - 1
        
        dh, dt, sigma = self.dh, self.dt, self.sigma
        
        a_dt_dh_2 = (self.a * dt / dh) ** 2
        
        self.const_1, self.const_2 = a_dt_dh_2 * sigma, a_dt_dh_2 * (1. - 2. * sigma)
        self.const_3 = (1. + 2 * self.const_1)
        
        # a
        self.a_arr =  [0.]
        self.a_arr += [-self.const_1 for i in range(2, self.eq_l)]
        self.a_arr += [-self.const_1]
        
        # b
        self.b =  [-self.const_1]
        self.b += [-self.const_1 for i in range(2, self.eq_l)]
        self.b += [0.]
        
        # c
        self.c =  [self.const_3 - 2. * self.const_1 ** 2 / self.const_3]
        self.c += [self.const_3 for i in range(2, self.eq_l)]
        self.c += [self.const_3 - 2. * self.const_1 ** 2 / self.const_3]
        
        ind = 0
        apr_max_steps = True
        self.err.append(criterion(self))
        while self.err[-1] > self.eps and apr_max_steps:
            
            # 1 задача

            self.x_arr[-1] = solve_1(self, ind)
            
            # 2 задача
            
            self.psi_arr[-1] = solve_2(self, ind)
        
            # Вычисляем новое f(s, t) по методу условного градиента
            self.alpha.append(get_alpha(self, ind))
            ing_2 = integral_2(self.psi_arr[-1] * self.psi_arr[-1], dh, dt) ** .5
            if ing_2 == 0.:
                buf = 0.
            else:
                buf = -1. * self.alpha[-1] * self.R_1 * self.psi_arr[-1] / ing_2
            self.f_arr.append(self.f_arr[-1] * (1. - self.alpha[-1]) + buf)
            
            self.final_step = ind
            ind += 1
            err = criterion(self)
            print(err)
            self.err.append(err)
            
            if max_steps is None:
                apr_max_steps = True
            else:
                apr_max_steps = ind < max_steps
                
            # Для нового шага
            self.x_arr.append(array(self.f, [self.l, self.T], [dh, dt]))
            self.x_arr[-1][0,:] = array(self.fi_0, [self.l], [dh])
            self.psi_arr.append(array(self.f, [self.l, self.T], [dh, dt]))
        
        self.x_arr.pop()
        self.psi_arr.pop()
        return self