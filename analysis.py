#!//usr/bin/env python3
import random_fp_generator
import logging
import math
from math import fabs
import os
import numpy
import random
import numpy as np
import struct
import itertools
import ctypes
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import scipy
from scipy.optimize import differential_evolution
from scipy.optimize import basinhopping
import sys
import time
import gc
f64max = sys.float_info.max
f64min = sys.float_info.min



verbose = False
#verbose = True
CUDA_LIB = ''
#MU = 1e-307
MU = 1.0
bo_iterations = 25 # number of iterations
smallest_subnormal = 4e-323
results = {}
runs_results = {}
trials_so_far = 0
trials_to_trigger = -1
trials_results = {}
random_results = {}
dfeg_results = {}
dfeg_inp_results = {"inf+":[],"inf-":[],"sub+":[],"sub-":[],"nan":[]}
dfeg_results_mc = {}
dfeg_inp_results_mc = {"inf+":[],"inf-":[],"sub+":[],"sub-":[],"nan":[]}
dfeg_results_de = {}
dfeg_inp_results_de = {"inf+":[],"inf-":[],"sub+":[],"sub-":[],"nan":[]}
fine_search_time=0.0

# ----- Status variables ------
found_inf_pos = False
found_inf_neg = False
found_under_pos = False
found_under_neg = False
# -----------------------------
# special inputs that may trigger exceptions
spinp_lst = [0.0,1.0,2.0,np.pi,2*np.pi,np.pi/2.0,np.e,1.0/1.3407807929942596e+154,2.2250738585072014e-308]

def initialize():
  global found_inf_pos, found_inf_neg, found_under_pos, found_under_neg
  found_inf_pos = False
  found_inf_neg = False
  found_under_pos = False
  found_under_neg = False

def set_max_iterations(n: int):
  global bo_iterations
  bo_iterations = n
def get_bo_inputs(opt_lst):
    inps = []
    for i in opt_lst:
        inps.append(i[1])
    return inps

def stop_func(x,f,accepted):
    if math.fabs(f)==0.0:
        return True
    else:
        return False
#----------------------------------------------------------------------------
# Ctype Wrappers
#----------------------------------------------------------------------------
def call_GPU_kernel_1(x0):
  script_dir = os.path.abspath(os.path.dirname(__file__))
  lib_path = os.path.join(script_dir, CUDA_LIB)
  E = ctypes.cdll.LoadLibrary(lib_path)
  E.kernel_wrapper_1.restype = ctypes.c_double
  res = E.kernel_wrapper_1(ctypes.c_double(x0))
  return res

def call_GPU_kernel_2(x0, x1):
  script_dir = os.path.abspath(os.path.dirname(__file__))
  lib_path = os.path.join(script_dir, CUDA_LIB)
  E = ctypes.cdll.LoadLibrary(lib_path)
  E.kernel_wrapper_1.restype = ctypes.c_double
  res = E.kernel_wrapper_1(ctypes.c_double(x0), ctypes.c_double(x1))
  return res

def call_GPU_kernel_3(x0, x1, x2):
  script_dir = os.path.abspath(os.path.dirname(__file__))
  lib_path = os.path.join(script_dir, CUDA_LIB)
  E = ctypes.cdll.LoadLibrary(lib_path)
  E.kernel_wrapper_1.restype = ctypes.c_double
  res = E.kernel_wrapper_1(ctypes.c_double(x0), ctypes.c_double(x1), ctypes.c_double(x2))
  return res

def call_GPU_kernel_4(x0, x1, x2, x3):
  script_dir = os.path.abspath(os.path.dirname(__file__))
  lib_path = os.path.join(script_dir, CUDA_LIB)
  E = ctypes.cdll.LoadLibrary(lib_path)
  E.kernel_wrapper_1.restype = ctypes.c_double
  res = E.kernel_wrapper_1(ctypes.c_double(x0), ctypes.c_double(x1), ctypes.c_double(x2), ctypes.c_double(x3))
  return res

def call_GPU_kernel_N(*args):
    script_dir = os.path.abspath(os.path.dirname(__file__))
    lib_path = os.path.join(script_dir, CUDA_LIB)
    E = ctypes.cdll.LoadLibrary(lib_path)
    E.kernel_wrapper_1.restype = ctypes.c_double
    res = E.kernel_wrapper_1(*[ctypes.c_double(arg) for arg in args])
    return res

#----------------------------------------------------------------------------
# Black box functions
#----------------------------------------------------------------------------

#def black_box_function(x0, x1, x2):
#  x0_fp = 1.0 * math.pow(10, x0)
#  x1_fp = 1.0 * math.pow(10, x1)
#  x2_fp = 1.0 * math.pow(10, x2)

  #return -call_GPU_kernel(x0, x1, x2)
  #r = call_GPU_kernel(x0, x1, x2)
#  r = call_GPU_kernel(x0_fp, x1_fp, x2_fp)
#  smallest_subnormal = -4.94e-323
#  if r==0.0 or r==-0.0:
#    return -1.0;
#  elif r > smallest_subnormal:
#    return -r
#  return r

#****************************** 1 Input ***********************************
# --------- Based on FP inputs --------
# Function goals: (1) maximize (2) find INF (3) use fp inputs
def function_max_inf_fp_1(x0):
  return call_GPU_kernel_1(x0)

# Function goals: (1) minimize (2) find INF (3) use fp inputs
def function_min_inf_fp_1(x0):
  return -call_GPU_kernel_1(x0)

# Function goals: (1) maximize (2) find Underflows (3) use fp inputs
def function_max_under_fp_1(x0):
  r = call_GPU_kernel_1(x0)
  if r==0.0 or r==-0.0:
    return -MU
  elif r > -smallest_subnormal:
    return -r
  return r

# Function goals: (1) minimize (2) find Underflows (3) use fp inputs
def function_min_under_fp_1(x0):
  r = call_GPU_kernel_1(x0)
  if r==0.0 or r==-0.0:
    return MU
  elif r < smallest_subnormal:
    return -r
  return r

# --------- Based on Exponent inputs --------
# Function goals: (1) maximize (2) find INF (3) use exp inputs
def function_max_inf_exp_1(x0):
  x0_fp = 1.0 * math.pow(10, x0)
  return call_GPU_kernel_1(x0_fp)

# Function goals: (1) minimize (2) find INF (3) use exp inputs
def function_min_inf_exp_1(x0):
  x0_fp = 1.0 * math.pow(10, x0)
  return -call_GPU_kernel_1(x0_fp)

# Function goals: (1) maximize (2) find Underflows (3) use exp inputs
def function_max_under_exp_1(x0):
  x0_fp = 1.0 * math.pow(10, x0)
  r = call_GPU_kernel_1(x0_fp)
  if r==0.0 or r==-0.0:
    return -MU
  elif r > -smallest_subnormal:
    return -r
  return r

# Function goals: (1) minimize (2) find Underflows (3) use exp inputs
def function_min_under_exp_1(x0):
  x0_fp = 1.0 * math.pow(10, x0)
  r = call_GPU_kernel_1(x0_fp)
  if r==0.0 or r==-0.0:
    return MU
  elif r < smallest_subnormal:
    return -r
  return r


#****************************** 2 Inputs ***********************************
# --------- Based on FP inputs --------
# Function goals: (1) maximize (2) find INF (3) use fp inputs
def function_max_inf_fp_2(x0, x1):
  return call_GPU_kernel_2(x0, x1)

# Function goals: (1) minimize (2) find INF (3) use fp inputs
def function_min_inf_fp_2(x0, x1):
  return -call_GPU_kernel_2(x0, x1)

# Function goals: (1) maximize (2) find Underflows (3) use fp inputs
def function_max_under_fp_2(x0, x1):
  r = call_GPU_kernel_2(x0, x1)
  if r==0.0 or r==-0.0:
    return -MU
  elif r > -smallest_subnormal:
    return -r
  return r

# Function goals: (1) minimize (2) find Underflows (3) use fp inputs
def function_min_under_fp_2(x0, x1):
  r = -call_GPU_kernel_2(x0, x1)
  if r==0.0 or r==-0.0:
    return MU
  elif r > smallest_subnormal:
    return -r
  return r

# --------- Based on Exponent inputs --------
# Function goals: (1) maximize (2) find INF (3) use exp inputs
def function_max_inf_exp_2(x0, x1):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  return call_GPU_kernel_2(x0_fp, x1_fp)

# Function goals: (1) minimize (2) find INF (3) use exp inputs
def function_min_inf_exp_2(x0, x1):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  return -call_GPU_kernel_2(x0_fp, x1_fp)

# Function goals: (1) maximize (2) find Underflows (3) use exp inputs
def function_max_under_exp_2(x0, x1):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  r = call_GPU_kernel_2(x0_fp, x1_fp)
  if r==0.0 or r==-0.0:
    return -MU
  elif r > -smallest_subnormal:
    return -r
  return r

# Function goals: (1) minimize (2) find Underflows (3) use exp inputs
def function_min_under_exp_2(x0, x1):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  r = -call_GPU_kernel_2(x0_fp, x1_fp)
  if r==0.0 or r==-0.0:
    return MU
  elif r > smallest_subnormal:
    return -r
  return r


#****************************** 3 Inputs ***********************************
# --------- Based on FP inputs --------
# Function goals: (1) maximize (2) find INF (3) use fp inputs
def function_max_inf_fp_3(x0, x1, x2):
  return call_GPU_kernel_3(x0, x1, x2)

# Function goals: (1) minimize (2) find INF (3) use fp inputs
def function_min_inf_fp_3(x0, x1, x2):
  return -call_GPU_kernel_3(x0, x1, x2)

# Function goals: (1) maximize (2) find Underflows (3) use fp inputs
def function_max_under_fp_3(x0, x1, x2):
  r = call_GPU_kernel_3(x0, x1, x2)
  if r==0.0 or r==-0.0:
    return -MU
  elif r > -smallest_subnormal:
    return -r
  return r

# Function goals: (1) minimize (2) find Underflows (3) use fp inputs
def function_min_under_fp_3(x0, x1, x2):
  r = call_GPU_kernel_3(x0, x1, x2)
  if r==0.0 or r==-0.0:
    return MU
  elif r < smallest_subnormal:
    return -r
  return r

# --------- Based on Exponent inputs --------
# Function goals: (1) maximize (2) find INF (3) use exp inputs
def function_max_inf_exp_3(x0, x1, x2):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  x2_fp = 1.0 * math.pow(10, x2)
  return call_GPU_kernel_3(x0_fp, x1_fp, x2_fp)

# Function goals: (1) minimize (2) find INF (3) use exp inputs
def function_min_inf_exp_3(x0, x1, x2):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  x2_fp = 1.0 * math.pow(10, x2)
  return -call_GPU_kernel_3(x0_fp, x1_fp, x2_fp)

# Function goals: (1) maximize (2) find Underflows (3) use exp inputs
def function_max_under_exp_3(x0, x1, x2):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  x2_fp = 1.0 * math.pow(10, x2)
  r = call_GPU_kernel_3(x0_fp, x1_fp, x2_fp)
  if r==0.0 or r==-0.0:
    return -MU
  elif r > -smallest_subnormal:
    return -r
  return r

# Function goals: (1) minimize (2) find Underflows (3) use exp inputs
def function_min_under_exp_3(x0, x1, x2):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  x2_fp = 1.0 * math.pow(10, x2)
  r = call_GPU_kernel_3(x0_fp, x1_fp, x2_fp)
  if r==0.0 or r==-0.0:
    return MU
  elif r < smallest_subnormal:
    return -r
  return r

#****************************** 4 Inputs ***********************************
# --------- Based on FP inputs --------
# Function goals: (1) maximize (2) find INF (3) use fp inputs
def function_max_inf_fp_4(x0, x1, x2, x3):
  return call_GPU_kernel_4(x0, x1, x2, x3)

# Function goals: (1) minimize (2) find INF (3) use fp inputs
def function_min_inf_fp_4(x0, x1, x2, x3):
  return -call_GPU_kernel_4(x0, x1, x2, x3)

# Function goals: (1) maximize (2) find Underflows (3) use fp inputs
def function_max_under_fp_4(x0, x1, x2, x3):
  r = call_GPU_kernel_4(x0, x1, x2, x3)
  if r==0.0 or r==-0.0:
    return -MU
  elif r > -smallest_subnormal:
    return -r
  return r

# Function goals: (1) minimize (2) find Underflows (3) use fp inputs
def function_min_under_fp_4(x0, x1, x2, x3):
  r = call_GPU_kernel_4(x0, x1, x2, x3)
  if r==0.0 or r==-0.0:
    return MU
  elif r < smallest_subnormal:
    return -r
  return r

# --------- Based on Exponent inputs --------
# Function goals: (1) maximize (2) find INF (3) use exp inputs
def function_max_inf_exp_4(x0, x1, x2, x3):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  x2_fp = 1.0 * math.pow(10, x2)
  x3_fp = 1.0 * math.pow(10, x3)
  return call_GPU_kernel_4(x0_fp, x1_fp, x2_fp, x3_fp)

# Function goals: (1) minimize (2) find INF (3) use exp inputs
def function_min_inf_exp_4(x0, x1, x2, x3):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  x2_fp = 1.0 * math.pow(10, x2)
  x3_fp = 1.0 * math.pow(10, x3)
  return -call_GPU_kernel_4(x0_fp, x1_fp, x2_fp, x3_fp)

# Function goals: (1) maximize (2) find Underflows (3) use exp inputs
def function_max_under_exp_3(x0, x1, x2, x3):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  x2_fp = 1.0 * math.pow(10, x2)
  x3_fp = 1.0 * math.pow(10, x3)
  r = call_GPU_kernel_4(x0_fp, x1_fp, x2_fp, x3_fp)
  if r==0.0 or r==-0.0:
    return -MU
  elif r > -smallest_subnormal:
    return -r
  return r

# Function goals: (1) minimize (2) find Underflows (3) use exp inputs
def function_min_under_exp_3(x0, x1, x2, x3):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  x2_fp = 1.0 * math.pow(10, x2)
  x3_fp = 1.0 * math.pow(10, x3)
  r = call_GPU_kernel_4(x0_fp, x1_fp, x2_fp, x3_fp)
  if r==0.0 or r==-0.0:
    return MU
  elif r < smallest_subnormal:
    return -r
  return r

#----------------------------------------------------------------------------
# Optimization loop
#----------------------------------------------------------------------------
max_normal = 1e+307

# -------------- 1 Input ----------------------
def bounds_fp_whole_1():
  b = []
  b.append({'x0': (-max_normal, max_normal)})
  return b

def bounds_fp_two_1():
  b = []
  b.append({'x0': (-max_normal, 0)})
  b.append({'x0': (0, max_normal)})
  return b

def bounds_fp_many_1():
  b = []
  limits = [0.0, 1e-307, 1e-100, 1e-10, 1e-1, 1e0, 1e+1, 1e+10, 1e+100, 1e+307]
  ranges = []
  for i in range(len(limits)-1):
    x = limits[i]
    y = limits[i+1]
    t = (min(x,y), max(x,y))
    ranges.append(t)
    x = -limits[i]
    y = -limits[i+1]
    t = (min(x,y), max(x,y))
    ranges.append(t)
  
  for r1 in ranges:
    b.append({'x0': r1})

  return b
  
def bounds_exp_whole_1():
  b = []
  b.append({'x0': (-307, 307)})
  return b

def bounds_exp_two_1():
  b = []
  b.append({'x0': (-307, 0)})
  b.append({'x0': (0, 307)})
  return b

def bounds_exp_many_1():
  b = []
  limits = [-307, -100, -10, -1, 0, +1, +10, +100, +307]
  for i in range(len(limits)-1):
    x = limits[i]
    y = limits[i+1]
    t = (min(x,y), max(x,y))
    b.append({'x0': t}) 
 
  return b

# -------------- 2 Inputs ----------------------
def bounds_fp_whole_2():
  b = []
  b.append({'x0': (-max_normal, max_normal), 'x1': (-max_normal, max_normal)})
  return b

def bounds_fp_two_2():
  b = []
  b.append({'x0': (-max_normal, 0), 'x1': (-max_normal, 0)})
  b.append({'x0': (0, max_normal), 'x1': (0, max_normal)})
  return b

def bounds_fp_many_2():
  b = []
  # {'target': -5e-324, 'params': {'x0': 1e-100, 'x1': 3.234942383692966}}
  #b.append({'x0': (1e-100, 1e-300), 'x1': (0, 4.0)}) # finds subnormal in pow(x0, x1)
  #b.append({'x0': (1e-100, 1e-307), 'x1': (0, 1e+1)}) # finds subnormal in pow(x0, x1)

  # finds subnormal in pow(x0, x1): 
  # {'target': -2.9950764292998e-310, 'params': {'x0': 6.639021130307829e-101, 'x1': 3.089739399676855}}
  # MU = 1.0
  # bo_iterations = 25 # number of iterations
  # utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.1e-1)
  #b.append({'x0': (1e-307, 1e-100), 'x1': (1e0, 1e+1)}) 

  limits = [0.0, 1e-307, 1e-100, 1e-10, 1e-1, 1e0, 1e+1, 1e+10, 1e+100, 1e+307]
  ranges = []
  for i in range(len(limits)-1):
    x = limits[i]
    y = limits[i+1]
    t = (min(x,y), max(x,y))
    ranges.append(t)
    x = -limits[i]
    y = -limits[i+1]
    t = (min(x,y), max(x,y))
    ranges.append(t)
  
  for r1 in ranges:
    for r2 in ranges:
      b.append({'x0': r1, 'x1': r2})

  return b

def bounds_exp_whole_2():
  b = []
  b.append({'x0': (-307, 307), 'x1': (-307, 307)})
  return b

def bounds_exp_two_2():
  b = []
  b.append({'x0': (-307, 0), 'x1': (-307, 0)})
  b.append({'x0': (0, 307),'x1': (0, 307)})
  return b

def bounds_exp_many_2():
  b = []
  limits = [-307, -100, -10, -1, 0, +1, +10, +100, +307]
  ranges = []
  for i in range(len(limits)-1):
    x = limits[i]
    y = limits[i+1]
    t = (min(x,y), max(x,y))
    ranges.append(t)
  
  for r1 in ranges:
    for r2 in ranges:
      b.append({'x0': r1, 'x1': r2})

  return b

# -------------- 3 Inputs ----------------------
def bounds_fp_whole_3():
  b = []
  b.append({'x0': (-max_normal, max_normal), 'x1': (-max_normal, max_normal), 'x2': (-max_normal, max_normal)})
  return b

def bounds_fp_two_3():
  b = []
  b.append({'x0': (-max_normal, 0), 'x1': (-max_normal, 0), 'x2': (-max_normal, 0)})
  b.append({'x0': (0, max_normal), 'x1': (0, max_normal), 'x2': (0, max_normal)})
  return b

def bounds_fp_many_3():
  b = []
  limits = [0.0, 1e-307, 1e-100, 1e-10, 1e-1, 1e0, 1e+1, 1e+10, 1e+100, 1e+307]
  ranges = []
  for i in range(len(limits)-1):
    x = limits[i]
    y = limits[i+1]
    t = (min(x,y), max(x,y))
    ranges.append(t)
    x = -limits[i]
    y = -limits[i+1]
    t = (min(x,y), max(x,y))
    ranges.append(t)
  
  for r1 in ranges:
    for r2 in ranges:
      #for r3 in ranges:
      b.append({'x0': r1, 'x1': r2, 'x2': r2})

  return b

def bounds_exp_whole_3():
  b = []
  b.append({'x0': (-307, 307), 'x1': (-307, 307), 'x2': (-307, 307)})
  return b

def bounds_exp_two_3():
  b = []
  b.append({'x0': (-307, 0), 'x1': (-307, 0), 'x2': (-307, 0)})
  b.append({'x0': (0, 307),'x1': (0, 307), 'x3': (0, 307)})
  return b

def bounds_exp_many_3():
  b = []
  limits = [-307, -100, -10, -1, 0, +1, +10, +100, +307]
  ranges = []
  for i in range(len(limits)-1):
    x = limits[i]
    y = limits[i+1]
    t = (min(x,y), max(x,y))
    ranges.append(t)
  
  for r1 in ranges:
    for r2 in ranges:
      #for r3 in ranges:
      b.append({'x0': r1, 'x1': r2, 'x2': r2})

  return b

# -------------- 4 Inputs ----------------------
#
# ----------------------------------------------

#----------------------------------------------------------------------------
# Results Checking
#----------------------------------------------------------------------------

def save_trials_to_trigger(exp_name: str):
  global trials_to_trigger, trials_so_far
  if trials_to_trigger == -1:
    trials_to_trigger = trials_so_far
    trials_results[exp_name] = trials_to_trigger

def is_inf_pos(val):
  if math.isinf(val):
    return val > 0.0
  return False

def is_inf_neg(val):
  if math.isinf(val):
    return val < 0.0
  return False

def is_under_pos(val):
  if numpy.isfinite(val):
    if val > 0.0 and val < 2.22e-308:
      return True
  return False

def is_under_neg(val):
  if numpy.isfinite(val):
    if val < 0.0 and val > -2.22e-308:
      return True
  return False

def save_results(val: float, exp_name: str):
  # Infinity
  if math.isinf(val):
    if exp_name not in results.keys():
      if val < 0.0:
        results[exp_name] = [1, 0, 0, 0, 0]
        save_trials_to_trigger(exp_name)
      else:
        results[exp_name] = [0, 1, 0, 0, 0]
        save_trials_to_trigger(exp_name)
    else:
      if val < 0.0:
        results[exp_name][0] += 1
      else:
        results[exp_name][1] += 1

  # Subnormals
  if numpy.isfinite(val):
    if val > -2.22e-308 and val < 2.22e-308:
      if val != 0.0 and val !=-0.0:
        if exp_name not in results.keys():
          if val < 0.0:
            results[exp_name] = [0, 0, 1, 0, 0]
            save_trials_to_trigger(exp_name)
          else:
            results[exp_name] = [0, 0, 0, 1, 0]
            save_trials_to_trigger(exp_name)
        else:
          if val < 0.0:
            results[exp_name][2] += 1
          else:
            results[exp_name][3] += 1

  if math.isnan(val):
    if exp_name not in results.keys():
      results[exp_name] = [0, 0, 0, 0, 1]
      save_trials_to_trigger(exp_name)
    else:
      results[exp_name][4] += 1

def are_we_done(func, recent_val, exp_name):
  global found_inf_pos, found_inf_neg, found_under_pos, found_under_neg

  # Finding INF+
  if 'max_inf' in func.__name__:
    if found_inf_pos:
      return True
    else:
      if is_inf_pos(recent_val):
        found_inf_pos = True
        save_results(recent_val, exp_name)
        return True

  # Finding INF-
  elif 'min_inf' in func.__name__:
    if found_inf_neg:
      return True
    else:
      if is_inf_neg(recent_val):
        found_inf_neg = True
        save_results(recent_val, exp_name)
        return True

  # Finding Under-
  elif 'max_under' in func.__name__:
    if found_under_neg:
      return True
    else:
      if is_under_neg(recent_val):
        found_under_neg = True
        save_results(recent_val, exp_name)
        return True

  # Finding Under+
  elif 'min_under' in func.__name__:
    if found_under_pos:
      return True
    else:
      if is_under_pos(recent_val):
        found_under_pos = True
        save_results(recent_val, exp_name)
        return True

  return False

def update_runs_table(exp_name: str):
  if exp_name not in runs_results.keys():
    runs_results[exp_name] = 0
  else:
    runs_results[exp_name] += 1

def run_optimizer(bounds, func, exp_name):
  global trials_to_trigger, trials_so_far
  trials_so_far = 0
  trials_to_trigger = -1
  if are_we_done(func, 0.0, exp_name):
    return
  optimizer = BayesianOptimization(f=func, pbounds=bounds, verbose=2, random_state=1)
  #print("bounds")
  #print(exp_name)
  #print(bounds)
  #print(len(bounds))
  #newbounds = []
  #for k,val in bounds.items():
  #    newbounds.append(list(val))
  #print(newbounds) 
  #newfunc = lambda x: func(*x)
  #optimizer = differential_evolution(newfunc, bounds=newbounds,popsize=3,polish=False,strategy='best1bin')
  try:
    if verbose: print('dfeg opt...')
    utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.1e-1)
    utility = UtilityFunction(kind="ucb", kappa=10, xi=0.1e-1)
    utility = UtilityFunction(kind="poi", kappa=10, xi=1e-1)
    for _ in range(bo_iterations):
      trials_so_far += 1
      next_point = optimizer.suggest(utility)
      target = func(**next_point)
      optimizer.register(params=next_point, target=target)

      update_runs_table(exp_name)

      # Check if we are done
      if are_we_done(func, target, exp_name):
        return
  except Exception as e:
    if verbose: print("Oops!", e.__class__, "occurred.")
    if verbose: print(e)
    if verbose: logging.exception("Something awful happened!")
  if verbose: print(optimizer.max)
  val = optimizer.max['target']
  #val = func(*optimizer.x)
  save_results(val, exp_name)

# input types: {"fp", "exp"}
def optimize(shared_lib: str, input_type: str, num_inputs: int, splitting: str):
  global CUDA_LIB
  CUDA_LIB = shared_lib

  assert num_inputs >= 1 and num_inputs <= 3

  funcs_fp_1 = [function_max_inf_fp_1, function_min_inf_fp_1, function_max_under_fp_1, function_min_under_fp_1]
  funcs_exp_1 = [function_max_inf_exp_1, function_min_inf_exp_1, function_max_under_exp_1, function_min_under_exp_1]
  funcs_fp_2 = [function_max_inf_fp_2, function_min_inf_fp_2, function_max_under_fp_2, function_min_under_fp_2]
  funcs_exp_2 = [function_max_inf_exp_2, function_min_inf_exp_2, function_max_under_exp_2, function_min_under_exp_2]
  funcs_fp_3 = [function_max_inf_fp_3, function_min_inf_fp_3, function_max_under_fp_3, function_min_under_fp_3]
  funcs_exp_3 = [function_max_inf_exp_3, function_min_inf_exp_3, function_max_under_exp_3, function_min_under_exp_3]
    
  if input_type == 'fp':
    if num_inputs == 1:
      if splitting == 'whole':
        initialize()
        for b in bounds_fp_whole_1():
          for f in funcs_fp_1:
            exp_name = [shared_lib, input_type, 'b_whole']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'two':
        initialize()
        for b in bounds_fp_two_1():
          for f in funcs_fp_1:
            exp_name = [shared_lib, input_type, 'b_two']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'many':
        initialize()
        for b in bounds_fp_many_1():
          for f in funcs_fp_1:
            exp_name = [shared_lib, input_type, 'b_many']
            run_optimizer(b, f, '|'.join(exp_name))
          
    elif num_inputs == 2:
      if splitting == 'whole':
        initialize()
        for b in bounds_fp_whole_2():
          for f in funcs_fp_2:
            exp_name = [shared_lib, input_type, 'b_whole']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'two':
        initialize()
        for b in bounds_fp_two_2():
          for f in funcs_fp_2:
            exp_name = [shared_lib, input_type, 'b_two']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'many':
        initialize()
        for b in bounds_fp_many_2():
          for f in funcs_fp_2:
            exp_name = [shared_lib, input_type, 'b_many']
            run_optimizer(b, f,  '|'.join(exp_name))

    elif num_inputs == 3:
      if splitting == 'whole':
        initialize()
        for b in bounds_fp_whole_3():
          for f in funcs_fp_3:
            exp_name = [shared_lib, input_type, 'b_whole']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'two':
        initialize()
        for b in bounds_fp_two_3():
          for f in funcs_fp_3:
            exp_name = [shared_lib, input_type, 'b_two']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'many':
        initialize()
        for b in bounds_fp_many_3():
          for f in funcs_fp_3:
            exp_name = [shared_lib, input_type, 'b_many']
            run_optimizer(b, f,  '|'.join(exp_name))

  elif input_type == 'exp':
    if num_inputs == 1:
      if splitting == 'whole':
        initialize()
        for b in bounds_exp_whole_1():
          for f in funcs_exp_1:
            exp_name = [shared_lib, input_type, 'b_whole']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'two':
        initialize()
        for b in bounds_exp_two_1():
          for f in funcs_exp_1:
            exp_name = [shared_lib, input_type, 'b_two']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'many':
        initialize()
        for b in bounds_exp_many_1():
          for f in funcs_exp_1:
            exp_name = [shared_lib, input_type, 'b_many']
            run_optimizer(b, f, '|'.join(exp_name))
        
    elif num_inputs == 2:
      if splitting == 'whole':
        initialize()
        for b in bounds_exp_whole_2():
          for f in funcs_exp_2:
            exp_name = [shared_lib, input_type, 'b_whole']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'two':
        initialize()
        for b in bounds_exp_two_2():
          for f in funcs_exp_2:
            exp_name = [shared_lib, input_type, 'b_two']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'many':
        initialize()
        for b in bounds_exp_many_2():
          for f in funcs_exp_2:
            exp_name = [shared_lib, input_type, 'b_many']
            run_optimizer(b, f, '|'.join(exp_name))

    elif num_inputs == 3:
      if splitting == 'whole':
        initialize()
        for b in bounds_exp_whole_3():
          for f in funcs_exp_3:
            exp_name = [shared_lib, input_type, 'b_whole']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'two':
        initialize()
        for b in bounds_exp_two_3():
          for f in funcs_exp_3:
            exp_name = [shared_lib, input_type, 'b_two']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'many':
        initialize()
        for b in bounds_exp_many_3():
          for f in funcs_exp_3:
            exp_name = [shared_lib, input_type, 'b_many']
            run_optimizer(b, f, '|'.join(exp_name))
  else:
    print('Invalid input type!')
    exit()

#-------------- Results --------------
#lassen60_26904/cuda_code_acos.cu.so|fp|b_many :    [0, 0, 0, 0, 32]
#lassen60_26904/cuda_code_hypot.cu.so|fp|b_many :     [0, 0, 5, 0, 0]
def print_results(shared_lib: str, number_sampling, range_splitting):
  key = shared_lib+'|'+number_sampling+'|b_'+range_splitting
  fun_name = os.path.basename(shared_lib)
  print('-------------- Results --------------')
  print(fun_name)
  if key in results.keys():
    print('\tINF+:', results[key][0])
    print('\tINF-:', results[key][1])
    print('\tSUB-:', results[key][2])
    print('\tSUB-:', results[key][3])
    print('\tNaN :', results[key][4])
  else:
    print('\tINF+:', 0)
    print('\tINF-:', 0)
    print('\tSUB-:', 0)
    print('\tSUB-:', 0)
    print('\tNaN :', 0) 

  print('\tRuns:', runs_results[key])
  #print('\n**** Runs ****')
  #for k in runs_results.keys():
  #  print(k, runs_results[k])

  print('')
# --------------- dfeg Optimizer -------------
def save_results_dfeg(val: float, exp_name: str, unbounded: bool, inp):
  found = False
  # Infinity
  if math.isinf(val):
    if exp_name not in dfeg_results.keys():
      if val > 0.0:
        dfeg_results[exp_name] = [1, 0, 0, 0, 0]
        dfeg_inp_results["inf+"].append(inp)
        found = True
      else:
        dfeg_results[exp_name] = [0, 1, 0, 0, 0]
        dfeg_inp_results["inf-"].append(inp)
        found = True
    else:
      if val > 0.0:
        dfeg_results[exp_name][0] += 1
        dfeg_inp_results["inf+"].append(inp)
        found = True
      else:
        dfeg_results[exp_name][1] += 1
        dfeg_inp_results["inf-"].append(inp)
        found = True

  # Subnormals
  if numpy.isfinite(val):
    if val > -f64min and val < f64min:
      if val != 0.0 and val !=-0.0:
        if exp_name not in dfeg_results.keys():
          if val > 0.0:
            dfeg_results[exp_name] = [0, 0, 1, 0, 0]
            dfeg_inp_results["sub+"].append(inp)
            found = True
          else:
            dfeg_results[exp_name] = [0, 0, 0, 1, 0]
            dfeg_inp_results["sub-"].append(inp)
            found = True
        else:
          if val > 0.0:
            dfeg_results[exp_name][2] += 1
            dfeg_inp_results["sub+"].append(inp)
            found = True
          else:
            dfeg_results[exp_name][3] += 1
            dfeg_inp_results["sub-"].append(inp)
            found = True

  if math.isnan(val):
    if exp_name not in dfeg_results.keys():
      dfeg_results[exp_name] = [0, 0, 0, 0, 1]
      dfeg_inp_results["nan"].append(inp)
      found = True
    else:
      dfeg_results[exp_name][4] += 1
      dfeg_inp_results["nan"].append(inp)
      found = True

  if exp_name not in dfeg_results.keys():
    dfeg_results[exp_name] = [0,0,0,0,0]

  if unbounded:
    return False
  else:
    return found
  #return False
def save_results_dfeg_de(val: float, exp_name: str, unbounded: bool, inp):
  found = False
  # Infinity
  if math.isinf(val):
    if exp_name not in dfeg_results_de.keys():
      if val > 0.0:
        dfeg_results_de[exp_name] = [1, 0, 0, 0, 0]
        dfeg_inp_results_de["inf+"].append(inp)
        found = True
      else:
        dfeg_results_de[exp_name] = [0, 1, 0, 0, 0]
        dfeg_inp_results_de["inf-"].append(inp)
        found = True
    else:
      if val > 0.0:
        dfeg_results_de[exp_name][0] += 1
        dfeg_inp_results_de["inf+"].append(inp)
        found = True
      else:
        dfeg_results_de[exp_name][1] += 1
        dfeg_inp_results_de["inf-"].append(inp)
        found = True

  # Subnormals
  if numpy.isfinite(val):
    if val > -f64min and val < f64min:
      if val != 0.0 and val !=-0.0:
        if exp_name not in dfeg_results_de.keys():
          if val > 0.0:
            dfeg_results_de[exp_name] = [0, 0, 1, 0, 0]
            dfeg_inp_results_de["sub+"].append(inp)
            found = True
          else:
            dfeg_results_de[exp_name] = [0, 0, 0, 1, 0]
            dfeg_inp_results_de["sub-"].append(inp)
            found = True
        else:
          if val > 0.0:
            dfeg_results_de[exp_name][2] += 1
            dfeg_inp_results_de["sub+"].append(inp)
            found = True
          else:
            dfeg_results_de[exp_name][3] += 1
            dfeg_inp_results_de["sub-"].append(inp)
            found = True

  if math.isnan(val):
    if exp_name not in dfeg_results_de.keys():
      dfeg_results_de[exp_name] = [0, 0, 0, 0, 1]
      dfeg_inp_results_de["nan"].append(inp)
      found = True
    else:
      dfeg_results_de[exp_name][4] += 1
      dfeg_inp_results_de["nan"].append(inp)
      found = True

  if exp_name not in dfeg_results_de.keys():
    dfeg_results_de[exp_name] = [0,0,0,0,0]

  if unbounded:
    return False
  else:
    return found
  #return False
def save_results_dfeg_mcmc(val: float, exp_name: str, unbounded: bool, inp):
  found = False
  # Infinity
  if math.isinf(val):
    if exp_name not in dfeg_results_mc.keys():
      if val > 0.0:
        dfeg_results_mc[exp_name] = [1, 0, 0, 0, 0]
        dfeg_inp_results_mc["inf+"].append(inp)
        found = True
      else:
        dfeg_results_mc[exp_name] = [0, 1, 0, 0, 0]
        dfeg_inp_results_mc["inf-"].append(inp)
        found = True
    else:
      if val > 0.0:
        dfeg_results_mc[exp_name][0] += 1
        dfeg_inp_results_mc["inf+"].append(inp)
        found = True
      else:
        dfeg_results_mc[exp_name][1] += 1
        dfeg_inp_results_mc["inf-"].append(inp)
        found = True

  # Subnormals
  if numpy.isfinite(val):
    if val > -f64min and val < f64min:
      if val != 0.0 and val !=-0.0:
        if exp_name not in dfeg_results_mc.keys():
          if val > 0.0:
            dfeg_results_mc[exp_name] = [0, 0, 1, 0, 0]
            dfeg_inp_results_mc["sub+"].append(inp)
            found = True
          else:
            dfeg_results_mc[exp_name] = [0, 0, 0, 1, 0]
            dfeg_inp_results_mc["sub-"].append(inp)
            found = True
        else:
          if val > 0.0:
            dfeg_results_mc[exp_name][2] += 1
            dfeg_inp_results_mc["sub+"].append(inp)
            found = True
          else:
            dfeg_results_mc[exp_name][3] += 1
            dfeg_inp_results_mc["sub-"].append(inp)
            found = True

  if math.isnan(val):
    if exp_name not in dfeg_results_mc.keys():
      dfeg_results_mc[exp_name] = [0, 0, 0, 0, 1]
      dfeg_inp_results_mc["nan"].append(inp)
      found = True
    else:
      dfeg_results_mc[exp_name][4] += 1
      dfeg_inp_results_mc["nan"].append(inp)
      found = True

  if exp_name not in dfeg_results_mc.keys():
    dfeg_results_mc[exp_name] = [0,0,0,0,0]

  if unbounded:
    return False
  else:
    return found
  #return False


# --------------- Random Sampling Optimizer -------------
def save_results_random(val: float, exp_name: str, unbounded: bool):
  found = False
  # Infinity
  if math.isinf(val):
    if exp_name not in random_results.keys():
      if val > 0.0:
        random_results[exp_name] = [1, 0, 0, 0, 0]
        found = True
      else:
        random_results[exp_name] = [0, 1, 0, 0, 0]
        found = True
    else:
      if val > 0.0:
        random_results[exp_name][0] += 1
        found = True
      else:
        random_results[exp_name][1] += 1
        found = True

  # Subnormals
  if numpy.isfinite(val):
    if val > -2.22e-308 and val < 2.22e-308:
      if val != 0.0 and val !=-0.0:
        if exp_name not in random_results.keys():
          if val > 0.0:
            random_results[exp_name] = [0, 0, 1, 0, 0]
            found = True
          else:
            random_results[exp_name] = [0, 0, 0, 1, 0]
            found = True
        else:
          if val > 0.0:
            random_results[exp_name][2] += 1
            found = True
          else:
            random_results[exp_name][3] += 1
            found = True

  if math.isnan(val):
    if exp_name not in random_results.keys():
      random_results[exp_name] = [0, 0, 0, 0, 1]
      found = True
    else:
      random_results[exp_name][4] += 1
      found = True

  if exp_name not in random_results.keys():
    random_results[exp_name] = [0,0,0,0,0]

  if unbounded:
    return False
  else:
    return found
  #return False


def floatToRawLongBits(value):
        return struct.unpack('Q', struct.pack('d', value))[0]

def longBitsToFloat(bits):
        return struct.unpack('d', struct.pack('Q', bits))[0]



#_tmp_lassen593_3682/cuda_code_acos.cu.so|RANDOM :    [0, 0, 0, 0, 271]
def print_results_random(shared_lib):
  key = shared_lib+'|RANDOM'
  fun_name = os.path.basename(shared_lib)
  print('-------------- Results --------------')
  print(fun_name)
  if key in random_results.keys():
    print('\tINF+:', random_results[key][0])
    print('\tINF-:', random_results[key][1])
    print('\tSUB-:', random_results[key][2])
    print('\tSUB-:', random_results[key][3])
    print('\tNaN :', random_results[key][4])
  else:
    print('\tINF+:', 0)
    print('\tINF-:', 0)
    print('\tSUB-:', 0)
    print('\tSUB-:', 0)
    print('\tNaN :', 0) 
  print('')

def print_results_dfeg(shared_lib):
  key = shared_lib+'|dfeg'
  fun_name = os.path.basename(shared_lib)
  print('-------------- Results --------------')
  print(fun_name)
  if key in dfeg_results.keys():
    print('\tINF+:', dfeg_results[key][0])
    print('\tINF-:', dfeg_results[key][1])
    print('\tSUB+:', dfeg_results[key][2])
    print('\tSUB-:', dfeg_results[key][3])
    print('\tNaN :', dfeg_results[key][4])
  else:
    print('\tINF+:', 0)
    print('\tINF-:', 0)
    print('\tSUB+:', 0)
    print('\tSUB-:', 0)
    print('\tNaN :', 0) 
  print('')
  print('--------------MCMC Results --------------')
  if key in dfeg_results_mc.keys():
    print('\tINF+:', dfeg_results_mc[key][0])
    print('\tINF-:', dfeg_results_mc[key][1])
    print('\tSUB+:', dfeg_results_mc[key][2])
    print('\tSUB-:', dfeg_results_mc[key][3])
    print('\tNaN :', dfeg_results_mc[key][4])
  else:
    print('\tINF+:', 0)
    print('\tINF-:', 0)
    print('\tSUB+:', 0)
    print('\tSUB-:', 0)
    print('\tNaN :', 0) 
  print('')
  #print('--------------DE Results --------------')
  #if key in dfeg_results_de.keys():
  #  print('\tINF+:', dfeg_results_de[key][0])
  #  print('\tINF-:', dfeg_results_de[key][1])
  #  print('\tSUB+:', dfeg_results_de[key][2])
  #  print('\tSUB-:', dfeg_results_de[key][3])
  #  print('\tNaN :', dfeg_results_de[key][4])
  #else:
  #  print('\tINF+:', 0)
  #  print('\tINF-:', 0)
  #  print('\tSUB+:', 0)
  #  print('\tSUB-:', 0)
  #  print('\tNaN :', 0) 
  #print('')
# Calls to wrappers:
# call_GPU_kernel_1(x0)
# call_GPU_kernel_1(x0,x1)
# call_GPU_kernel_1(x0,x1,x2)
def fdistribution_partition(in_min, in_max):
    tmp_l = []
    a = np.frexp(in_min)
    b = np.frexp(in_max)
    tmp_j = 0
    if (in_min < 0)&(in_max > 0):
        if in_min >= -1.0:
            tmp_l.append([in_min, 0])
        else:
            for i in range(1, a[1]+1):
                tmp_i = np.ldexp(-0.5, i)
                tmp_l.append([tmp_i, tmp_j])
                tmp_j = tmp_i
            if in_min != tmp_j:
                tmp_l.append([in_min, tmp_j])
        tmp_j = 0
        if in_max <= 1.0:
            tmp_l.append([0, in_max])
        else:
            for i in range(1, b[1]+1):
                tmp_i = np.ldexp(0.5, i)
                tmp_l.append([tmp_j, tmp_i])
                tmp_j = tmp_i
            if in_max != tmp_j:
                tmp_l.append([tmp_j, in_max])
    if (in_min < 0) & (0 >= in_max):
        if in_min >= -1:
            tmp_l.append([in_min, in_max])
            return tmp_l
        else:
            if in_max > -1:
                tmp_l.append([-1, in_max])
                tmp_j = -1.0
                for i in range(2, a[1] + 1):
                    tmp_i = np.ldexp(-0.5, i)
                    tmp_l.append([tmp_i, tmp_j])
                    tmp_j = tmp_i
                if in_min != tmp_j:
                    tmp_l.append([in_min, tmp_j])
            else:
                if a[1] == b[1]:
                    tmp_l.append([in_min, in_max])
                    return tmp_l
                else:
                    tmp_j = np.ldexp(-0.5, b[1]+1)
                    tmp_l.append([tmp_j, in_max])
                    if tmp_j != in_min:
                        for i in range(b[1]+2, a[1]+1):
                            tmp_i = np.ldexp(-0.5, i)
                            tmp_l.append([tmp_i, tmp_j])
                            tmp_j = tmp_i
                        if in_min != tmp_j:
                            tmp_l.append([in_min, tmp_j])
    if (in_min >= 0) & (in_max > 0):
        if in_max <= 1:
            tmp_l.append([in_min, in_max])
            return tmp_l
        else:
            if in_min < 1:
                tmp_l.append([in_min, 1])
                tmp_j = 1.0
                for i in range(2, b[1] + 1):
                    tmp_i = np.ldexp(0.5, i)
                    tmp_l.append([tmp_j, tmp_i])
                    tmp_j = tmp_i
                if in_max != tmp_j:
                    tmp_l.append([tmp_j, in_max])
            else:
                if a[1] == b[1]:
                    tmp_l.append([in_min, in_max])
                    return tmp_l
                else:
                    tmp_j = np.ldexp(0.5, a[1]+1)
                    tmp_l.append([in_min, tmp_j])
                    if tmp_j != in_max:
                        for i in range(a[1]+2, b[1]+1):
                            tmp_i = np.ldexp(0.5, i)
                            tmp_l.append([tmp_j, tmp_i])
                            tmp_j = tmp_i
                        if in_max != tmp_j:
                            tmp_l.append([tmp_j, in_max])
    return tmp_l




def optimize_randomly(shared_lib: str, num_inputs: int, max_iters: int, unbounded: bool):
  global CUDA_LIB
  CUDA_LIB = shared_lib
  exp_name = shared_lib+'|'+'RANDOM'
  for i in range(max_iters):
    if num_inputs == 1:
      x0 = random_fp_generator.fp64_generate_number()
      r = call_GPU_kernel_1(x0)
      found = save_results_random(r, exp_name, unbounded)
      #if found: break
    elif num_inputs == 2:
      x0 = random_fp_generator.fp64_generate_number()
      x1 = random_fp_generator.fp64_generate_number()
      r = call_GPU_kernel_2(x0,x1)
      found = save_results_random(r, exp_name, unbounded)
      #if found: break
    elif num_inputs == 3:
      x0 = random_fp_generator.fp64_generate_number()
      x1 = random_fp_generator.fp64_generate_number()
      x2 = random_fp_generator.fp64_generate_number()
      r = call_GPU_kernel_3(x0,x1,x2)
      found = save_results_random(r, exp_name, unbounded)
      #if found: break 
def exp_eva_fun2(func,x):
    if len(x) == 1:
        val = func(x[0])
    else:
        val = func(*x)
    if np.isnan(val):
        return 0.0,val
    if np.isinf(val):
        return 0.0,val
    if val > -f64min and val < f64min:
        if val !=0:
            return 0.0,val
        else:
            return 9218868437227405312.0,val
    int_val = floatToRawLongBits(abs(val))
    dist_nan = fabs(int_val- 9221120237041090560)
    dist_inf = fabs(int_val- 9218868437227405312)
    dist_subn = fabs(int_val - 4493330023422296)
    minval = min(dist_nan,dist_inf,dist_subn)
    return minval,val

#print(abs(floatToRawLongBits(1e308)- 9218868437227405312))
#print(floatToRawLongBits(1e308)-4493330023422296)
#print(4493330023422296)
#print(floatToRawLongBits(1e308))

def exp_eva_fun_wbo(func,x):
    if len(x) == 1:
        val = func(x[0])
    else:
        val = func(*x)
    if np.isnan(val):
        return 0.0
    if np.isinf(val):
        return 0.0
    if val > -f64min and val < f64min:
        if val !=0:
            return 0.0
        else:
            return 9218868437227405312.0
    int_val = floatToRawLongBits(abs(val))
    dist_nan = fabs(int_val- 9221120237041090560)
    dist_inf = fabs(int_val- 9218868437227405312)
    dist_subn = fabs(int_val - 4493330023422296)
    return min(dist_nan,dist_inf,dist_subn)

def exp_eva_fun(func,x):
    if len(x) == 1:
        val = func(x[0])
    else:
        val = func(*x)
    if np.isnan(val):
        return 0.0
    if np.isinf(val):
        return 0.0
    if val > -f64min and val < f64min:
        if val !=0:
            return 0.0
        else:
            return 9218868437227405312.0
    int_val = floatToRawLongBits(abs(val))
    dist_nan = fabs(int_val- 9221120237041090560)
    dist_inf = fabs(int_val- 9218868437227405312)
    dist_subn = fabs(int_val - 4493330023422296)
    return min(dist_nan,dist_inf,dist_subn)
def exp_eva_fun_inf(func,x):
    if len(x) == 1:
        val = func(x[0])
    else:
        val = func(*x)
    if np.isnan(val):
        return 9218868437227405312.0
    if np.isinf(val):
        return 0.0
    if val > -f64min and val < f64min:
        if val !=0:
            return 9218868437227405312.0
        else:
            return 9218868437227405312.0
    int_val = floatToRawLongBits(abs(val))
    dist_inf = fabs(int_val- 9218868437227405312)
    return dist_inf
def exp_eva_fun_subn(func,x):
    if len(x) == 1:
        val = func(x[0])
    else:
        val = func(*x)
    if np.isnan(val):
        return 9218868437227405312.0
    if np.isinf(val):
        return 9218868437227405312.0
    if val > -f64min and val < f64min:
        if val !=0:
            return 0.0
        else:
            return 4493330023422296
    int_val = floatToRawLongBits(abs(val))
    dist_subn = fabs(int_val - 4493330023422296)
    return dist_subn
def exp_eva_fun_nan(func,x):
    if len(x) == 1:
        val = func(x[0])
    else:
        val = func(*x)
    if np.isnan(val):
        return 0.0
    if np.isinf(val):
        return 9218868437227405312.0
    if val > -f64min and val < f64min:
        if val !=0:
            return 9218868437227405312.0
        else:
            return 9218868437227405312.0
    int_val = floatToRawLongBits(abs(val))
    dist_nan = fabs(int_val- 9221120237041090560)
    return dist_nan
def exp_eva_fun_nan(func,x):
    if len(x) == 1:
        val = func(x[0])
    else:
        val = func(*x)
    if np.isnan(val):
        return 0.0
    if np.isinf(val):
        return 0.0
    if val > -f64min and val < f64min:
        if val !=0:
            return 0.0
        else:
            return 9218868437227405312.0
    int_val = floatToRawLongBits(abs(val))
    dist_nan = fabs(int_val- 9221120237041090560)
    dist_inf = fabs(int_val- 9218868437227405312)
    dist_subn = fabs(int_val - 4493330023422296)
    return dist_nan
def fpartition(input_domain):
    l_var = []
    for i in input_domain:
        for j in i:
            tmp_l = fdistribution_partition(j[0], j[1])
            l_var.append(tmp_l)
    ini_confs = []
    for element in itertools.product(*l_var):
        temp_ele = []
        for i in list(element):
            temp_ele.append(tuple(i))
        ini_confs.append(temp_ele)
    return ini_confs


def gen_all_bounds(num):
    bounds = []
    bounds = [[0,pow(2.0,-1022)],[-pow(2.0,-1022),0]]
    size_step = pow(6,num)
    for i in range(-1022,1023,size_step):
        if i + size_step > 1023:
            bounds.append([pow(2.0,i),pow(2.0,1023)])
            bounds.append([-pow(2.0,1023),-pow(2.0,i)])
        else:
            bounds.append([pow(2.0,i),pow(2.0,i+size_step)])
            bounds.append([-pow(2.0,i+size_step),-pow(2.0,i)])
    return bounds
def generate_inpdm(num_inputs):
    basic_bounds = gen_all_bounds(num_inputs)
    l_var = []
    for i in range(num_inputs):
        l_var.append(basic_bounds)
    ini_confs = []
    for element in itertools.product(*l_var):
        temp_ele = []
        for i in list(element):
            temp_ele.append(tuple(i))
        ini_confs.append(temp_ele)
    return ini_confs



def produce_n_input(i,n):
    var_l = []
    n = int(n)
    #for k in i[0]:
    for k in i:
        #var_l.append(sorted(np.random.uniform(k[0], k[1], n)))
        inps = []
        for ips in range(0,n):
            rd = random.uniform(0,1)
            inps.append(k[1]*rd + k[0]*(1-rd))
        #var_l.append(sorted(np.random.uniform(k[0], k[1], n)))
        var_l.append(sorted(inps))
    input_l = []
    for element in itertools.product(*var_l):
        input_l.append(element)
    return input_l
def run_optimizer_bo_large(bounds, func, exp_name,unbounded,ori_func):
  global trials_to_trigger, trials_so_far
  trials_so_far = 0
  trials_to_trigger = -1
  #if are_we_done(func, 0.0, exp_name):
  #  return
  optimizer = BayesianOptimization(f=func, pbounds=bounds, verbose=2, random_state=1)
  try:
    if verbose: print('defped opt...')
    utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.1e-1)
    #utility = UtilityFunction(kind="ucb", kappa=10, xi=0.1e-1)
    #utility = UtilityFunction(kind="poi", kappa=10, xi=1e-1)
    for _ in range(bo_iterations):
      trials_so_far += 1
      next_point = optimizer.suggest(utility)
      inps = get_bo_inputs(next_point.items())
      target = func(inps)
      #target = func(next_point['x0'])
      optimizer.register(params=next_point, target=target)

      update_runs_table(exp_name)
      if target == 0.0:
        val = ori_func(*inps)
        save_results_dfeg(val, exp_name, unbounded,tuple(inps))
        save_results_dfeg_mcmc(val, exp_name, unbounded,tuple(inps))

  except ValueError as e:
    if verbose: print("Oops!", e.__class__, "occurred.")
    if verbose: print(e)
    if verbose: logging.exception("Something awful happened!")
  if verbose: print(optimizer.max)
  if optimizer.max!={}:
      inp_lst = list(optimizer.max['params'].items())
      inps = get_bo_inputs(inp_lst)
      val = ori_func(*inps)
      save_results_dfeg(val, exp_name, unbounded,tuple(inps))
      save_results_dfeg_mcmc(val, exp_name, unbounded,tuple(inps))
def run_optimizer_dfeg(bounds, func, exp_name,unbounded,ori_func):
  global trials_to_trigger, trials_so_far
  trials_so_far = 0
  trials_to_trigger = -1
  #if are_we_done(func, 0.0, exp_name):
  #  return
  optimizer = BayesianOptimization(f=func, pbounds=bounds, verbose=2, random_state=1)
  try:
    if verbose: print('defped opt...')
    utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.1e-1)
    #utility = UtilityFunction(kind="ucb", kappa=10, xi=0.1e-1)
    #utility = UtilityFunction(kind="poi", kappa=10, xi=1e-1)
    for _ in range(bo_iterations):
      trials_so_far += 1
      next_point = optimizer.suggest(utility)
      inps = get_bo_inputs(next_point.items())
      target = func(*inps)
      #target = func(next_point['x0'])
      optimizer.register(params=next_point, target=target)

      update_runs_table(exp_name)
      if target == 0.0:
        val = ori_func(*inps)
        save_results_dfeg(val, exp_name, unbounded,tuple(inps))
        save_results_dfeg_mcmc(val, exp_name, unbounded,tuple(inps))

  except ValueError as e:
    if verbose: print("Oops!", e.__class__, "occurred.")
    if verbose: print(e)
    if verbose: logging.exception("Something awful happened!")
  if verbose: print(optimizer.max)
  if optimizer.max!={}:
      inp_lst = list(optimizer.max['params'].items())
      inps = get_bo_inputs(inp_lst)
      val = ori_func(*inps)
      save_results_dfeg(val, exp_name, unbounded,tuple(inps))
      save_results_dfeg_mcmc(val, exp_name, unbounded,tuple(inps))

#Ablation analysis: WBO 
def optimize_dfegly_small_WBO(shared_lib: str, num_inputs: int, max_iters: int, unbounded: bool, rand_seed: int,res_cnt: int):
    global CUDA_LIB
    CUDA_LIB = shared_lib
    exp_name = shared_lib+'|'+'dfeg'
    if num_inputs == 1:
        func = call_GPU_kernel_1
    elif num_inputs == 2:
        func = call_GPU_kernel_2
    elif num_inputs == 3:
        func = call_GPU_kernel_3
    bounds = generate_inpdm(num_inputs)
    np.random.seed(rand_seed)
    sums = []
    all_inps = []
    #if res_cnt >= 100:
    #    sample_number = 12*(res_cnt-99)
    #else:
    sample_number = 12
    for i in bounds:
        inps = produce_n_input(i,int(math.pow(sample_number,1.0/num_inputs)))
        all_inps.append([inps,i])
    var_l = []
    input_l = []
    for _ in range(num_inputs):
        var_l.append(spinp_lst+[x*-1 for x in spinp_lst])
    for element in itertools.product(*var_l):
        input_l.append(element)
    all_inps.append([input_l,[]])
    for inps in all_inps:
      trig_flag = 0
      temp_sum = []
      for inp in inps[0]:
          res,val = exp_eva_fun2(func,inp)
          if res == 0.0: 
            save_results_dfeg(val, exp_name, unbounded,inp)
            trig_flag = 1
          else:
            temp_sum.append(res) 
      if trig_flag == 0:
          sums.append([min(temp_sum),inps[1]])
    sums.sort()
    len_index = int(np.fmin(20.0,len(sums)/2.0))
    st = time.time()
    af_bounds = []
    for i in range(0,len_index):
        af_bounds.append(sums[i][1])
    x_str = ['x0','x1','x2']
    new_bounds = []
    for bi in af_bounds:
      count = 0
      tp_bounds = {}
      for i in bi:
        tp_bounds[x_str[count]] = i
        count = count + 1
      new_bounds.append(tp_bounds)
    if num_inputs==1:
        new_func = lambda x: -exp_eva_fun(func,[x])
    if num_inputs==2:
        new_func = lambda x,y: -exp_eva_fun(func,[x,y])
    if num_inputs==3:
        new_func = lambda x,y,z: -exp_eva_fun(func,[x,y,z])
    st = time.time()
    for i in new_bounds:
      print(i)
      print("bounds")
      if i!={}:
          run_optimizer_dfeg(i, new_func, exp_name,unbounded,func)
    print("bo time")
    print(time.time()-st)
    fine_search_time=time.time()-st
def get_double_exponent_interval(x):
    if math.isnan(x):
        return 2049.0
    val = np.fabs(x)
    valint = floatToRawLongBits(val)
    # val = valint<<1
    if x<0:
        sign = -1
    else:
        sign = 1
    exponent = (valint>>52)-1023
    if x<0:
        return [-pow(2.0,exponent),-pow(2.0,exponent-1)]
    else:
        return [pow(2.0,exponent-2),pow(2.0,exponent)]
def optimize_dfegly_WBO(shared_lib: str, num_inputs: int, max_iters: int, unbounded: bool, rand_seed: int,res_cnt: int):
    if num_inputs <= 3:
        try:
            optimize_dfegly_small_WBO(shared_lib, num_inputs, max_iters, unbounded, rand_seed,res_cnt)
        except TimeoutError:
            return
    else:
        global CUDA_LIB
        CUDA_LIB = shared_lib
        exp_name = shared_lib+'|'+'dfeg'
        func = call_GPU_kernel_N
        np.random.seed(rand_seed)
        st = time.time()
        cnt = 1
        explist = [[j for j in range(-1072,1023,num_inputs*2)] for i in range(0,num_inputs)]
        #explist = [[j for j in range(-1072,7,num_inputs*2)] for i in range(0,num_inputs)]
        #oneexplist = [i for i in range(-1072,-1022,num_inputs)]+[j for j in range(-1022,0,num_inputs*6)]+[k for k in range(0,8)]
        #print(oneexplist)
        #print(len(oneexplist))
        #explist = [ oneexplist for i in range(0,num_inputs)]
        #zipexplist = list(itertools.product(*explist))
        #len_zip = len(zipexplist)
        #print(len(zipexplist))
        num = 1.0
        val_lst = []
        small_val = 0.0
        try:
            while True:
            #for zri in zipexplist: 
            #for lli in range(0,len_zip): 
                cnt = cnt + 1
                rinp = []
                #zri = random.choice(zipexplist)
                #zipexplist.remove(zri)
                for ri in range(0,num_inputs):
                    randnum = random.uniform(1, 2)
                    rint = random.choice(explist[ri])
                    if randnum > 1.5:
                        rinp.append(randnum*pow(2,rint))
                    elif randnum < 1.4:
                        rinp.append(-randnum*pow(2,rint))
                    else:
                        rinp.append(random.choice(spinp_lst))
                #for ri in zri:
                #    randnum = random.uniform(1, 2)
                #    rint = ri 
                #    if randnum > 1.5:
                #        rinp.append(randnum*pow(2,rint))
                #    elif randnum < 1.4:
                #        rinp.append(-randnum*pow(2,rint))
                #    else:
                #        rinp.append(random.choice(spinp_lst))
                rinp = tuple(rinp)
                #print(rinp)
                res,val = exp_eva_fun2(func,rinp)
                num = num + 1.0
                #print(val)
                if res == 0.0: 
                  save_results_dfeg(val, exp_name, unbounded,rinp)
                  rri = random.randint(0,num_inputs-1)
                  if num > pow(len(range(-1022,1023,num_inputs)),num_inputs): 
                      num = 0.0
                      explist[rri].remove(random.choice(explist[rri]))
                      if explist[rri] == []:
                          break
                else:
                  if len(val_lst)==1:
                      small_val = val
                      val_lst.append([val,rinp])
                  elif len(val_lst)<20:
                      val_lst.append([val,rinp])
                      if val > small_val:
                            small_val = val
                      val_lst.sort()
                  else:
                      if val < small_val:
                        val_lst[-1]=[val,rinp]
                      val_lst.sort()
                  continue
        except TimeoutError:
            for inps in val_lst: 
                key_pairs = []
                new_bounds = []
                for i in range(0,num_inputs):
                    x = 'x' + str(i)
                    val = get_double_exponent_interval(inps[1][i])
                    key_pairs.append(tuple([x,val]))
                new_bounds.append(dict(key_pairs))
                new_func = lambda x: exp_eva_fun_wbo(func,x)
                run_optimizer_bo_large(new_bounds[0], new_func, exp_name,unbounded,func)
            cp_dfeg_inp_results = dfeg_inp_results
            for key, value in cp_dfeg_inp_results.items():
                dfeg_inp_results[key]=list(set(value))
            cp_dfeg_inp_results_mc = dfeg_inp_results_mc
            for key, value in cp_dfeg_inp_results_mc.items():
                dfeg_inp_results_mc[key]=list(set(value))
            cp_dfeg_inp_results = {} 
            cp_dfeg_inp_results_mc = {} 
            fine_search_time=time.time()-st
            cp_dfeg_inp_results = {} 
            cp_dfeg_inp_results_mc = {} 
#Ablation analysis: WOMCMC 
def optimize_dfegly_small_WOMCMC(shared_lib: str, num_inputs: int, max_iters: int, unbounded: bool, rand_seed: int,res_cnt: int):
    global CUDA_LIB
    CUDA_LIB = shared_lib
    exp_name = shared_lib+'|'+'dfeg'
    if num_inputs == 1:
        func = call_GPU_kernel_1
    elif num_inputs == 2:
        func = call_GPU_kernel_2
    elif num_inputs == 3:
        func = call_GPU_kernel_3
    bounds = generate_inpdm(num_inputs)
    np.random.seed(rand_seed)
    sums = []
    all_inps = []
    #if res_cnt >= 100:
    #    sample_number = 12*(res_cnt-99)
    #else:
    sample_number = 12
    for i in bounds:
        inps = produce_n_input(i,int(math.pow(sample_number,1.0/num_inputs)))
        all_inps.append([inps,i])
    var_l = []
    input_l = []
    for _ in range(num_inputs):
        var_l.append(spinp_lst+[x*-1 for x in spinp_lst])
    for element in itertools.product(*var_l):
        input_l.append(element)
    all_inps.append([input_l,[]])
    for inps in all_inps:
      trig_flag = 0
      temp_sum = []
      for inp in inps[0]:
          res,val = exp_eva_fun2(func,inp)
          if res == 0.0: 
            save_results_dfeg(val, exp_name, unbounded,inp)
            trig_flag = 1
          else:
            temp_sum.append(res) 
      if trig_flag == 0:
          sums.append([min(temp_sum),inps[1]])
    sums.sort()
    cp_dfeg_inp_results = dfeg_inp_results
    for key, value in cp_dfeg_inp_results.items():
        dfeg_inp_results[key]=list(set(value))
def optimize_dfegly_WOMCMC(shared_lib: str, num_inputs: int, max_iters: int, unbounded: bool, rand_seed: int,res_cnt: int):
    if num_inputs <= 3:
        try:
            optimize_dfegly_small_WOMCMC(shared_lib, num_inputs, max_iters, unbounded, rand_seed,res_cnt)
        except TimeoutError:
            return
    else:
        global CUDA_LIB
        CUDA_LIB = shared_lib
        exp_name = shared_lib+'|'+'dfeg'
        func = call_GPU_kernel_N
        np.random.seed(rand_seed)
        st = time.time()
        cnt = 1
        explist = [[j for j in range(-1072,1023,num_inputs*2)] for i in range(0,num_inputs)]
        #explist = [[j for j in range(-1072,7,num_inputs*2)] for i in range(0,num_inputs)]
        #oneexplist = [i for i in range(-1072,-1022,num_inputs)]+[j for j in range(-1022,0,num_inputs*6)]+[k for k in range(0,8)]
        #print(oneexplist)
        #print(len(oneexplist))
        #explist = [ oneexplist for i in range(0,num_inputs)]
        #zipexplist = list(itertools.product(*explist))
        #len_zip = len(zipexplist)
        #print(len(zipexplist))
        num = 1.0
        val_lst = []
        small_val = 0.0
        try:
            while True:
            #for zri in zipexplist: 
            #for lli in range(0,len_zip): 
                cnt = cnt + 1
                rinp = []
                #zri = random.choice(zipexplist)
                #zipexplist.remove(zri)
                for ri in range(0,num_inputs):
                    randnum = random.uniform(1, 2)
                    rint = random.choice(explist[ri])
                    if randnum > 1.5:
                        rinp.append(randnum*pow(2,rint))
                    elif randnum < 1.4:
                        rinp.append(-randnum*pow(2,rint))
                    else:
                        rinp.append(random.choice(spinp_lst))
                #for ri in zri:
                #    randnum = random.uniform(1, 2)
                #    rint = ri 
                #    if randnum > 1.5:
                #        rinp.append(randnum*pow(2,rint))
                #    elif randnum < 1.4:
                #        rinp.append(-randnum*pow(2,rint))
                #    else:
                #        rinp.append(random.choice(spinp_lst))
                rinp = tuple(rinp)
                #print(rinp)
                res,val = exp_eva_fun2(func,rinp)
                num = num + 1.0
                #print(val)
                if res == 0.0: 
                  save_results_dfeg(val, exp_name, unbounded,rinp)
                  rri = random.randint(0,num_inputs-1)
                  if num > pow(len(range(-1022,1023,num_inputs)),num_inputs): 
                      num = 0.0
                      explist[rri].remove(random.choice(explist[rri]))
                      if explist[rri] == []:
                          break
                else:
                  if len(val_lst)==1:
                      small_val = val
                      val_lst.append([val,rinp])
                  elif len(val_lst)<20:
                      val_lst.append([val,rinp])
                      if val > small_val:
                            small_val = val
                      val_lst.sort()
                  else:
                      if val < small_val:
                        val_lst[-1]=[val,rinp]
                      val_lst.sort()
                  continue
        except TimeoutError:
            cp_dfeg_inp_results = dfeg_inp_results
            for key, value in cp_dfeg_inp_results.items():
                dfeg_inp_results[key]=list(set(value))
            cp_dfeg_inp_results = {} 
            cp_dfeg_inp_results_mc = {} 
#Ablation analysis: WORAND 
def optimize_dfegly_small_WORAND(shared_lib: str, num_inputs: int, max_iters: int, unbounded: bool, rand_seed: int,res_cnt: int):
    global CUDA_LIB
    CUDA_LIB = shared_lib
    exp_name = shared_lib+'|'+'dfeg'
    if num_inputs == 1:
        func = call_GPU_kernel_1
    elif num_inputs == 2:
        func = call_GPU_kernel_2
    elif num_inputs == 3:
        func = call_GPU_kernel_3
    bounds = generate_inpdm(num_inputs)
    np.random.seed(rand_seed)
    sums = []
    all_inps = []
    st = time.time()
    for si in bounds:
        if si !=[]:
          newfunc = lambda x: exp_eva_fun(func,x)
          x = produce_n_input(si,1)
          minimizer_kwargs = { "method": "Nelder-Mead"}
          #minimizer_kwargs = { "method": "powell"}
          optimizer2 = basinhopping(newfunc,x,minimizer_kwargs=minimizer_kwargs,niter_success=5,niter=200)
          if optimizer2.fun == 0.0:
              x = optimizer2.x
              #r = func(*x)
              flag = 1
              new_x = []
              for xi in x:
                if np.isinf(xi):
                  if xi > 0:
                      new_x.append(f64max)
                  else:
                      new_x.append(-f64max)
                else:
                    new_x.append(xi)
                if np.isnan(xi):
                  flag = 0
                  break
              if flag == 1:
                r = func(*new_x)
                nx = tuple(new_x)
                save_results_dfeg(r, exp_name, unbounded,nx)
                save_results_dfeg_mcmc(r, exp_name, unbounded,nx)
          del optimizer2
    print("mcmc time")
    print(time.time()-st)
    cp_dfeg_inp_results = dfeg_inp_results
    for key, value in cp_dfeg_inp_results.items():
        dfeg_inp_results[key]=list(set(value))
    cp_dfeg_inp_results_mc = dfeg_inp_results_mc
    for key, value in cp_dfeg_inp_results_mc.items():
        dfeg_inp_results_mc[key]=list(set(value))
    fine_search_time=time.time()-st
def optimize_dfegly_WORAND(shared_lib: str, num_inputs: int, max_iters: int, unbounded: bool, rand_seed: int,res_cnt: int):
    if num_inputs <= 3:
        try:
            optimize_dfegly_small_WORAND(shared_lib, num_inputs, max_iters, unbounded, rand_seed,res_cnt)
        except TimeoutError:
            return
    else:
        global CUDA_LIB
        CUDA_LIB = shared_lib
        exp_name = shared_lib+'|'+'dfeg'
        func = call_GPU_kernel_N
        np.random.seed(rand_seed)
        st = time.time()
        cnt = 1
        explist = [[j for j in range(-1072,1022,num_inputs*2)] for i in range(0,num_inputs)]
        #oneexplist = [i for i in range(-1072,-1022,num_inputs)]+[j for j in range(-1022,0,num_inputs*6)]+[k for k in range(0,8)]
        #print(oneexplist)
        #print(len(oneexplist))
        #explist = [ oneexplist for i in range(0,num_inputs)]
        #zipexplist = list(itertools.product(*explist))
        #len_zip = len(zipexplist)
        #print(len(zipexplist))
        num = 1.0
        val_lst = []
        small_val = 0.0
        try:
            while True:
                cnt = cnt + 1
                rinp = []
                #zri = random.choice(zipexplist)
                #zipexplist.remove(zri)
                for ri in range(0,num_inputs):
                    randnum = random.uniform(1, 2)
                    rint = random.choice(explist[ri])
                    if randnum > 1.5:
                        rinp.append(randnum*pow(2,rint))
                    elif randnum < 1.4:
                        rinp.append(-randnum*pow(2,rint))
                    else:
                        rinp.append(random.choice(spinp_lst))
                #for ri in zri:
                #    randnum = random.uniform(1, 2)
                #    rint = ri 
                #    if randnum > 1.5:
                #        rinp.append(randnum*pow(2,rint))
                #    elif randnum < 1.4:
                #        rinp.append(-randnum*pow(2,rint))
                #    else:
                #        rinp.append(random.choice(spinp_lst))
                rinp = tuple(rinp)
                newfunc = lambda x: exp_eva_fun(func,x)
                x = rinp 
                minimizer_kwargs = { "method": "Nelder-Mead"}
                # minimizer_kwargs = { "method": "Nelder-Mead", "options": {"maxiter": 3}}
                # optimizer2 = basinhopping(newfunc,x,callback=stop_func,minimizer_kwargs=minimizer_kwargs,niter_success=1,niter=10)
                optimizer2 = basinhopping(newfunc,x,minimizer_kwargs=minimizer_kwargs,niter_success=3,niter=20)
                if optimizer2.fun == 0.0:
                    x = optimizer2.x
                    flag = 1
                    new_x = []
                    for xi in x:
                      if np.isinf(xi):
                        if xi > 0:
                            new_x.append(f64max)
                        else:
                            new_x.append(-f64max)
                      else:
                          new_x.append(xi)
                      if np.isnan(xi):
                        flag = 0
                        break
                    if flag == 1:
                      r = func(*new_x)
                      nx = tuple(new_x)
                      save_results_dfeg(r, exp_name, unbounded,nx)
                      save_results_dfeg_mcmc(r, exp_name, unbounded,nx)
                del optimizer2
        except TimeoutError:
                cp_dfeg_inp_results = dfeg_inp_results
                for key, value in cp_dfeg_inp_results.items():
                    dfeg_inp_results[key]=list(set(value))
                cp_dfeg_inp_results_mc = dfeg_inp_results_mc
                for key, value in cp_dfeg_inp_results_mc.items():
                    dfeg_inp_results_mc[key]=list(set(value))
                cp_dfeg_inp_results = {} 
                cp_dfeg_inp_results_mc = {} 
                fine_search_time=time.time()-st
                cp_dfeg_inp_results = {} 
                cp_dfeg_inp_results_mc = {} 
def init_binary_inpdm_large(num_inputs):
    basic_bounds = [[-128,0],[0,128]]
    l_var = []
    for i in range(num_inputs):
        l_var.append(basic_bounds)
    ini_confs = []
    for element in itertools.product(*l_var):
        temp_ele = []
        for i in list(element):
            temp_ele.append(tuple(i))
        ini_confs.append(temp_ele)
    return ini_confs
#Ablation analysis: BGRT 
def init_binary_inpdm(num_inputs):
    basic_bounds = [[-f64max,0],[0,f64max]]
    l_var = []
    for i in range(num_inputs):
        l_var.append(basic_bounds)
    ini_confs = []
    for element in itertools.product(*l_var):
        temp_ele = []
        for i in list(element):
            temp_ele.append(tuple(i))
        ini_confs.append(temp_ele)
    return ini_confs
def split_bounds(ib):
    if type(ib[0])==float:
        ib = [ib]
    i = list(ib[0])
    midpoint = i[1]/2.0 + i[0]/2.0 
    return [[i[0],midpoint],[midpoint,i[1]]]
def generate_binary_inpdm(num_inputs,new_bounds):
    basic_bounds = []
    for i in new_bounds:
        basic_bounds.append(split_bounds(i))
    l_var = []
    for i in range(num_inputs):
        l_var.append(basic_bounds[i])
    ini_confs = []
    for element in itertools.product(*l_var):
        temp_ele = []
        for i in list(element):
            temp_ele.append(tuple(i))
        ini_confs.append(temp_ele)
    return ini_confs
    
def optimize_dfegly_small_WBGRT(shared_lib: str, num_inputs: int, max_iters: int, unbounded: bool, rand_seed: int,res_cnt: int):
    global CUDA_LIB
    CUDA_LIB = shared_lib
    exp_name = shared_lib+'|'+'dfeg'
    if num_inputs == 1:
        func = call_GPU_kernel_1
    elif num_inputs == 2:
        func = call_GPU_kernel_2
    elif num_inputs == 3:
        func = call_GPU_kernel_3
    bounds = init_binary_inpdm(num_inputs)
    #print(bounds)
    #print(generate_inpdm(num_inputs))
    new_bounds = bounds
    while True:
        bounds = generate_binary_inpdm(num_inputs,new_bounds)
        #print(bounds)
        np.random.seed(rand_seed)
        sums = []
        all_inps = []
        #if res_cnt >= 100:
        #    sample_number = 12*(res_cnt-99)
        #else:
        sample_number = 12
        for i in bounds:
            #print(i)
            inps = produce_n_input(i,int(math.pow(sample_number,1.0/num_inputs)))
            #print(inps)
            all_inps.append([inps,i])
        var_l = []
        input_l = []
        #print(all_inps)
        for inps in all_inps:
          trig_flag = 0
          temp_sum = []
          for inp in inps[0]:
              res,val = exp_eva_fun2(func,inp)
              if res == 0.0: 
                save_results_dfeg(val, exp_name, unbounded,inp)
                trig_flag = 1
              else:
                temp_sum.append(res) 
          if trig_flag == 0:
              sums.append([min(temp_sum),inps[1]])
        sums.sort()
        len_index = int(np.fmin(20.0,len(sums)/2.0))
        st = time.time()
        for si in sums[0:len_index]:
            if si[1] !=[]:
              newfunc = lambda x: exp_eva_fun(func,x)
              x = produce_n_input(si[1],1)
              minimizer_kwargs = { "method": "Nelder-Mead"}
              #minimizer_kwargs = { "method": "powell"}
              optimizer2 = basinhopping(newfunc,x,minimizer_kwargs=minimizer_kwargs,niter_success=5,niter=200)
              if optimizer2.fun == 0.0:
                  x = optimizer2.x
                  #r = func(*x)
                  flag = 1
                  new_x = []
                  for xi in x:
                    if np.isinf(xi):
                      if xi > 0:
                          new_x.append(f64max)
                      else:
                          new_x.append(-f64max)
                    else:
                        new_x.append(xi)
                    if np.isnan(xi):
                      flag = 0
                      break
                  if flag == 1:
                    r = func(*new_x)
                    nx = tuple(new_x)
                    save_results_dfeg(r, exp_name, unbounded,nx)
                    save_results_dfeg_mcmc(r, exp_name, unbounded,nx)
              else:  
                  new_bounds = si[1]
              del optimizer2
        cp_dfeg_inp_results = dfeg_inp_results
        for key, value in cp_dfeg_inp_results.items():
            dfeg_inp_results[key]=list(set(value))
        cp_dfeg_inp_results_mc = dfeg_inp_results_mc
        for key, value in cp_dfeg_inp_results_mc.items():
            dfeg_inp_results_mc[key]=list(set(value))
        fine_search_time=time.time()-st
def optimize_dfegly_WBGRT(shared_lib: str, num_inputs: int, max_iters: int, unbounded: bool, rand_seed: int,res_cnt: int):
    if num_inputs <= 3:
        try:
            optimize_dfegly_small_WBGRT(shared_lib, num_inputs, max_iters, unbounded, rand_seed,res_cnt)
        except TimeoutError:
            return
    else:
        global CUDA_LIB
        CUDA_LIB = shared_lib
        exp_name = shared_lib+'|'+'dfeg'
        func = call_GPU_kernel_N
        np.random.seed(rand_seed)
        st = time.time()
        cnt = 1
        #explist = [[j for j in range(-1072,1023,num_inputs*2)] for i in range(0,num_inputs)]
        #explist = [[j for j in range(-1072,1023,num_inputs*2)] for i in range(0,num_inputs)]
        try:
            num = 1.0
            val_lst = []
            small_val = 0.0
            bounds = init_binary_inpdm(num_inputs)
            #print(bounds)
            #print(generate_inpdm(num_inputs))
            new_bounds = bounds
            while True:
            #for zri in zipexplist: 
            #for lli in range(0,len_zip): 
                cnt = cnt + 1
                rinp = []
                bounds = generate_binary_inpdm(num_inputs,new_bounds)
                #zri = random.choice(zipexplist)
                #zipexplist.remove(zri)
                for tbds in bounds:
                    rinp = produce_n_input(tbds,1)
                    rinp = tuple(rinp[0])
                    res,val = exp_eva_fun2(func,rinp)
                    num = num + 1.0
                    #print(val)
                    if res == 0.0: 
                      save_results_dfeg(val, exp_name, unbounded,rinp)
                      # rri = random.randint(0,num_inputs-1)
                      # if num > pow(len(range(-1022,1023,num_inputs)),num_inputs): 
                      #     num = 0.0
                      #     explist[rri].remove(random.choice(explist[rri]))
                      #     if explist[rri] == []:
                      #         break
                    else:
                      if len(val_lst)==1:
                          small_val = val
                          val_lst.append([val,rinp])
                      elif len(val_lst)<10*num_inputs:
                          val_lst.append([val,rinp])
                          if val > small_val:
                                small_val = val
                          val_lst.sort()
                      else:
                          if val < small_val:
                            val_lst[-1]=[val,rinp]
                          val_lst.sort()
                      new_bounds = tbds 
                      continue
        except TimeoutError:
            for inps in val_lst: 
              newfunc = lambda x: exp_eva_fun(func,x)
              x = inps[1] 
              minimizer_kwargs = { "method": "Nelder-Mead"}
              # minimizer_kwargs = { "method": "Nelder-Mead", "options": {"maxiter": 3}}
              # optimizer2 = basinhopping(newfunc,x,callback=stop_func,minimizer_kwargs=minimizer_kwargs,niter_success=1,niter=10)
              optimizer2 = basinhopping(newfunc,x,minimizer_kwargs=minimizer_kwargs,niter_success=3,niter=20)
              if optimizer2.fun == 0.0:
                  x = optimizer2.x
                  flag = 1
                  new_x = []
                  for xi in x:
                    if np.isinf(xi):
                      if xi > 0:
                          new_x.append(f64max)
                      else:
                          new_x.append(-f64max)
                    else:
                        new_x.append(xi)
                    if np.isnan(xi):
                      flag = 0
                      break
                  if flag == 1:
                    r = func(*new_x)
                    nx = tuple(new_x)
                    save_results_dfeg(r, exp_name, unbounded,nx)
                    save_results_dfeg_mcmc(r, exp_name, unbounded,nx)
              del optimizer2
            cp_dfeg_inp_results = dfeg_inp_results
            for key, value in cp_dfeg_inp_results.items():
                dfeg_inp_results[key]=list(set(value))
            cp_dfeg_inp_results_mc = dfeg_inp_results_mc
            for key, value in cp_dfeg_inp_results_mc.items():
                dfeg_inp_results_mc[key]=list(set(value))
            cp_dfeg_inp_results = {} 
            cp_dfeg_inp_results_mc = {} 
            fine_search_time=time.time()-st
            cp_dfeg_inp_results = {} 
            cp_dfeg_inp_results_mc = {} 

#Ablation analysis:  no partitioning
def optimize_dfegly_small_WOFP(shared_lib: str, num_inputs: int, max_iters: int, unbounded: bool, rand_seed: int,res_cnt: int):
    global CUDA_LIB
    CUDA_LIB = shared_lib
    exp_name = shared_lib+'|'+'dfeg'
    if num_inputs == 1:
        func = call_GPU_kernel_1
    elif num_inputs == 2:
        func = call_GPU_kernel_2
    elif num_inputs == 3:
        func = call_GPU_kernel_3
    bounds = [[[-f64max,f64max] for x in range(0,num_inputs)]] 
    np.random.seed(rand_seed)
    while True:
        sums = []
        all_inps = []
        #if res_cnt >= 100:
        #    sample_number = 12*(res_cnt-99)
        #else:
        sample_number = 12
        for i in bounds:
            inps = produce_n_input(i,int(math.pow(sample_number,1.0/num_inputs)))
            all_inps.append([inps,i])
        var_l = []
        for inps in all_inps:
          trig_flag = 0
          temp_sum = []
          for inp in inps[0]:
              res,val = exp_eva_fun2(func,inp)
              if res == 0.0: 
                save_results_dfeg(val, exp_name, unbounded,inp)
                trig_flag = 1
              else:
                temp_sum.append(res) 
          if trig_flag == 0:
              sums.append([min(temp_sum),inps[1]])
        sums.sort()
        len_index = int(np.fmin(20.0,len(sums)/2.0))
        st = time.time()
        for bd in bounds:
          newfunc = lambda x: exp_eva_fun(func,x)
          x = produce_n_input(bd,1)
          minimizer_kwargs = { "method": "Nelder-Mead"}
          #minimizer_kwargs = { "method": "powell"}
          optimizer2 = basinhopping(newfunc,x,minimizer_kwargs=minimizer_kwargs,niter_success=5,niter=200)
          if optimizer2.fun == 0.0:
              x = optimizer2.x
              #r = func(*x)
              flag = 1
              new_x = []
              for xi in x:
                if np.isinf(xi):
                  if xi > 0:
                      new_x.append(f64max)
                  else:
                      new_x.append(-f64max)
                else:
                    new_x.append(xi)
                if np.isnan(xi):
                  flag = 0
                  break
              if flag == 1:
                r = func(*new_x)
                nx = tuple(new_x)
                save_results_dfeg(r, exp_name, unbounded,nx)
                save_results_dfeg_mcmc(r, exp_name, unbounded,nx)
          del optimizer2
    cp_dfeg_inp_results = dfeg_inp_results
    for key, value in cp_dfeg_inp_results.items():
        dfeg_inp_results[key]=list(set(value))
    cp_dfeg_inp_results_mc = dfeg_inp_results_mc
    for key, value in cp_dfeg_inp_results_mc.items():
        dfeg_inp_results_mc[key]=list(set(value))
    fine_search_time=time.time()-st
def optimize_dfegly_WOFP(shared_lib: str, num_inputs: int, max_iters: int, unbounded: bool, rand_seed: int,res_cnt: int):
    if num_inputs <= 3:
        try:
            optimize_dfegly_small_WOFP(shared_lib, num_inputs, max_iters, unbounded, rand_seed,res_cnt)
        except TimeoutError:
            return
    else:
        global CUDA_LIB
        CUDA_LIB = shared_lib
        exp_name = shared_lib+'|'+'dfeg'
        func = call_GPU_kernel_N
        np.random.seed(rand_seed)
        st = time.time()
        cnt = 1
        num = 1.0
        val_lst = []
        small_val = 0.0
        try:
            while True:
            #for zri in zipexplist: 
            #for lli in range(0,len_zip): 
                cnt = cnt + 1
                rinp = []
                #zri = random.choice(zipexplist)
                #zipexplist.remove(zri)
                for ri in range(0,num_inputs):
                    rd = random.uniform(0,1)
                    rinp.append(f64max*(2*rd-1))
                #for ri in zri:
                #    randnum = random.uniform(1, 2)
                #    rint = ri 
                #    if randnum > 1.5:
                #        rinp.append(randnum*pow(2,rint))
                #    elif randnum < 1.4:
                #        rinp.append(-randnum*pow(2,rint))
                #    else:
                #        rinp.append(random.choice(spinp_lst))
                rinp = tuple(rinp)
                #print(rinp)
                res,val = exp_eva_fun2(func,rinp)
                num = num + 1.0
                #print(val)
                if res == 0.0: 
                  save_results_dfeg(val, exp_name, unbounded,rinp)
                  rri = random.randint(0,num_inputs-1)
                  #if num > pow(len(range(-1022,1023,num_inputs)),num_inputs): 
                  #    num = 0.0
                  #    explist[rri].remove(random.choice(explist[rri]))
                  #    if explist[rri] == []:
                  #        break
                else:
                  if len(val_lst)==1:
                      small_val = val
                      val_lst.append([val,rinp])
                  elif len(val_lst)<10*num_inputs:
                      val_lst.append([val,rinp])
                      if val > small_val:
                            small_val = val
                      val_lst.sort()
                  else:
                      if val < small_val:
                        val_lst[-1]=[val,rinp]
                      val_lst.sort()
                  continue
        except TimeoutError:
            for inps in val_lst: 
              newfunc = lambda x: exp_eva_fun(func,x)
              x = inps[1] 
              minimizer_kwargs = { "method": "Nelder-Mead"}
              # minimizer_kwargs = { "method": "Nelder-Mead", "options": {"maxiter": 3}}
              # optimizer2 = basinhopping(newfunc,x,callback=stop_func,minimizer_kwargs=minimizer_kwargs,niter_success=1,niter=10)
              optimizer2 = basinhopping(newfunc,x,minimizer_kwargs=minimizer_kwargs,niter_success=3,niter=20)
              if optimizer2.fun == 0.0:
                  x = optimizer2.x
                  flag = 1
                  new_x = []
                  for xi in x:
                    if np.isinf(xi):
                      if xi > 0:
                          new_x.append(f64max)
                      else:
                          new_x.append(-f64max)
                    else:
                        new_x.append(xi)
                    if np.isnan(xi):
                      flag = 0
                      break
                  if flag == 1:
                    r = func(*new_x)
                    nx = tuple(new_x)
                    save_results_dfeg(r, exp_name, unbounded,nx)
                    save_results_dfeg_mcmc(r, exp_name, unbounded,nx)
              del optimizer2
              # gc.collect()
            cp_dfeg_inp_results = dfeg_inp_results
            for key, value in cp_dfeg_inp_results.items():
                dfeg_inp_results[key]=list(set(value))
            cp_dfeg_inp_results_mc = dfeg_inp_results_mc
            for key, value in cp_dfeg_inp_results_mc.items():
                dfeg_inp_results_mc[key]=list(set(value))
            fine_search_time=time.time()-st
            cp_dfeg_inp_results = {} 
            cp_dfeg_inp_results_mc = {} 
            


def optimize_dfegly_small(shared_lib: str, num_inputs: int, max_iters: int, unbounded: bool, rand_seed: int,res_cnt: int):
    global CUDA_LIB
    CUDA_LIB = shared_lib
    exp_name = shared_lib+'|'+'dfeg'
    if num_inputs == 1:
        func = call_GPU_kernel_1
    elif num_inputs == 2:
        func = call_GPU_kernel_2
    elif num_inputs == 3:
        func = call_GPU_kernel_3
    try:
        cnt = 0
        bounds = generate_inpdm(num_inputs)
        np.random.seed(rand_seed)
        sums = []
        all_inps = []
        # if res_cnt >= 100:
        #    sample_number = 12*(res_cnt-99)
        # else:
        sample_number = 12
        for i in bounds:
            inps = produce_n_input(i,int(math.pow(sample_number,1.0/num_inputs)))
            all_inps.append([inps,i])
        var_l = []
        input_l = []
        for _ in range(num_inputs):
            var_l.append(spinp_lst+[x*-1 for x in spinp_lst])
        for element in itertools.product(*var_l):
            input_l.append(element)
        all_inps.append([input_l,[]])
        for inps in all_inps:
          trig_flag = 0
          temp_sum = []
          for inp in inps[0]:
              res,val = exp_eva_fun2(func,inp)
              if res == 0.0: 
                save_results_dfeg(val, exp_name, unbounded,inp)
                trig_flag = 1
              else:
                temp_sum.append(res) 
          if trig_flag == 0:
              sums.append([min(temp_sum),inps[1]])
        sums.sort()
        len_index = int(np.fmin(20.0,len(sums)/2.0))
        st = time.time()
        for si in sums[0:len_index]:
            if si[1] !=[]:
              newfunc = lambda x: exp_eva_fun(func,x)
              x = produce_n_input(si[1],1)
              minimizer_kwargs = { "method": "Nelder-Mead"}
              #minimizer_kwargs = { "method": "powell"}
              #minimizer_kwargs = { "method": "Nelder-Mead", "options": {"maxiter": 10}}
              #optimizer2 = basinhopping(newfunc,x,callback=stop_func,minimizer_kwargs=minimizer_kwargs,niter_success=3,niter=10)
              optimizer2 = basinhopping(newfunc,x,minimizer_kwargs=minimizer_kwargs,niter_success=5,niter=200)
              if optimizer2.fun == 0.0:
                  x = optimizer2.x
                  #r = func(*x)
                  flag = 1
                  new_x = []
                  for xi in x:
                    if np.isinf(xi):
                      if xi > 0:
                          new_x.append(f64max)
                      else:
                          new_x.append(-f64max)
                    else:
                        new_x.append(xi)
                    if np.isnan(xi):
                      flag = 0
                      break
                  if flag == 1:
                    r = func(*new_x)
                    nx = tuple(new_x)
                    save_results_dfeg(r, exp_name, unbounded,nx)
                    save_results_dfeg_mcmc(r, exp_name, unbounded,nx)
              del optimizer2
              # gc.collect()
            cnt = cnt + 1
        print("mcmc time")
        print(time.time()-st)
        cp_dfeg_inp_results = dfeg_inp_results
        for key, value in cp_dfeg_inp_results.items():
            dfeg_inp_results[key]=list(set(value))
        cp_dfeg_inp_results_mc = dfeg_inp_results_mc
        for key, value in cp_dfeg_inp_results_mc.items():
            dfeg_inp_results_mc[key]=list(set(value))
        fine_search_time=time.time()-st
    except TimeoutError:
        sums.sort()
        len_index = int(np.fmin(20.0,len(sums)/2.0))
        st = time.time()
        for si in sums[cnt:len_index]:
            if si[1] !=[]:
              newfunc = lambda x: exp_eva_fun(func,x)
              x = produce_n_input(si[1],1)
              minimizer_kwargs = { "method": "Nelder-Mead"}
              #minimizer_kwargs = { "method": "powell"}
              #minimizer_kwargs = { "method": "Nelder-Mead", "options": {"maxiter": 10}}
              #optimizer2 = basinhopping(newfunc,x,callback=stop_func,minimizer_kwargs=minimizer_kwargs,niter_success=3,niter=10)
              optimizer2 = basinhopping(newfunc,x,minimizer_kwargs=minimizer_kwargs,niter_success=5,niter=200)
              #minimizer_kwargs = { "method": "Nelder-Mead", "options": {"maxiter": 10}}
              #optimizer2 = basinhopping(newfunc,x,callback=stop_func,minimizer_kwargs=minimizer_kwargs,niter_success=3,niter=10)
              if optimizer2.fun == 0.0:
                  x = optimizer2.x
                  #r = func(*x)
                  flag = 1
                  new_x = []
                  for xi in x:
                    if np.isinf(xi):
                      if xi > 0:
                          new_x.append(f64max)
                      else:
                          new_x.append(-f64max)
                    else:
                        new_x.append(xi)
                    if np.isnan(xi):
                      flag = 0
                      break
                  if flag == 1:
                    r = func(*new_x)
                    nx = tuple(new_x)
                    save_results_dfeg(r, exp_name, unbounded,nx)
                    save_results_dfeg_mcmc(r, exp_name, unbounded,nx)
              del optimizer2
        print("mcmc time")
        print(time.time()-st)
        cp_dfeg_inp_results = dfeg_inp_results
        for key, value in cp_dfeg_inp_results.items():
            dfeg_inp_results[key]=list(set(value))
        cp_dfeg_inp_results_mc = dfeg_inp_results_mc
        for key, value in cp_dfeg_inp_results_mc.items():
            dfeg_inp_results_mc[key]=list(set(value))
        fine_search_time=time.time()-st
def optimize_dfegly(shared_lib: str, num_inputs: int, max_iters: int, unbounded: bool, rand_seed: int,res_cnt: int):
    if num_inputs <= 3:
        optimize_dfegly_small(shared_lib, num_inputs, max_iters, unbounded, rand_seed,res_cnt)
    else:
        global CUDA_LIB
        CUDA_LIB = shared_lib
        exp_name = shared_lib+'|'+'dfeg'
        func = call_GPU_kernel_N
        np.random.seed(rand_seed)
        st = time.time()
        cnt = 1
        explist = [[j for j in range(-1072,1022,num_inputs*2)] for i in range(0,num_inputs)]
        num = 1.0
        val_lst = []
        small_val = 0.0
        try:
            while True:
                cnt = cnt + 1
                rinp = []
                for ri in range(0,num_inputs):
                    randnum = random.uniform(1, 2)
                    rint = random.choice(explist[ri])
                    if randnum > 1.5:
                        rinp.append(randnum*pow(2,rint))
                    elif randnum < 1.4:
                        rinp.append(-randnum*pow(2,rint))
                    else:
                        rinp.append(random.choice(spinp_lst))
                rinp = tuple(rinp)
                res,val = exp_eva_fun2(func,rinp)
                num = num + 1.0
                if res == 0.0: 
                  save_results_dfeg(val, exp_name, unbounded,rinp)
                  # rri = random.randint(0,num_inputs-1)
                  # if num > pow(len(range(-1022,1023,num_inputs)),num_inputs): 
                  #     num = 0.0
                  #     explist[rri].remove(random.choice(explist[rri]))
                  #     if explist[rri] == []:
                  #         break
                else:
                  if len(val_lst)==1:
                      small_val = val
                      val_lst.append([val,rinp])
                  elif len(val_lst)<10*num_inputs:
                      val_lst.append([val,rinp])
                      if val > small_val:
                        small_val = val
                      val_lst.sort()
                  else:
                      if val < small_val:
                        val_lst[-1]=[val,rinp]
                      val_lst.sort()
                  continue
        except TimeoutError:
            mctime = time.time()
            for inps in val_lst: 
              newfunc = lambda x: exp_eva_fun(func,x)
              x = inps[1] 
              # minimizer_kwargs = { "method": "Nelder-Mead", "options": {"maxiter": 3}}
              minimizer_kwargs = { "method": "Nelder-Mead"}
              # optimizer2 = basinhopping(newfunc,x,callback=stop_func,minimizer_kwargs=minimizer_kwargs,niter_success=1,niter=10)
              optimizer2 = basinhopping(newfunc,x,minimizer_kwargs=minimizer_kwargs,niter_success=3,niter=20)
              if optimizer2.fun == 0.0:
                  x = optimizer2.x
                  flag = 1
                  new_x = []
                  for xi in x:
                    if np.isinf(xi):
                      if xi > 0:
                          new_x.append(f64max)
                      else:
                          new_x.append(-f64max)
                    else:
                        new_x.append(xi)
                    if np.isnan(xi):
                      flag = 0
                      break
                  if flag == 1:
                    r = func(*new_x)
                    nx = tuple(new_x)
                    save_results_dfeg(r, exp_name, unbounded,nx)
                    save_results_dfeg_mcmc(r, exp_name, unbounded,nx)
              del optimizer2
              # gc.collect()
              cp_dfeg_inp_results = dfeg_inp_results
              for key, value in cp_dfeg_inp_results.items():
                  dfeg_inp_results[key]=list(set(value))
              cp_dfeg_inp_results_mc = dfeg_inp_results_mc
              for key, value in cp_dfeg_inp_results_mc.items():
                  dfeg_inp_results_mc[key]=list(set(value))
            print("MCMC time")
            print(time.time()-mctime)
            fine_search_time=time.time()-st
            cp_dfeg_inp_results = {} 
            cp_dfeg_inp_results_mc = {} 

