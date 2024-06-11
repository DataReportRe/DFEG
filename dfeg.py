#!/usr/bin/env python3

import argparse
import math
import subprocess
import socket
import os
import analysis
import sys
import shutil
import time
import pickle
import signal
import gc
def timeout_handler(signum, frame):
    raise TimeoutError("Function timed out")

def save_line_list(file_name,l):
    with open(file_name, "wb") as fp:
        pickle.dump(l, fp)

def save_line_list(file_name,l):
    with open(file_name, "wb") as fp:
        pickle.dump(l, fp)
#------------------------------------------------------------------------------
# Random Pool
#------------------------------------------------------------------------------


rd_seed = [82547955,18805512,51059660,67951510,96673401,92529168,43798981,\
           77041498,99700547,46432894,47637490,44611437,39774397,41271573,\
           4645333,25792865,3175680,69902962,60120588,56215621,86667354,\
           74905104,94207956,38027412,8741397,12937909,1370902,43545965,\
           47452337,66102720,86237691,61455401,14149645,39284815,92388247,\
           55354625,59213294,89102079,21502948,94527829,91610400,26056364,\
           41300704,79553483,78203397,20052848,70074407,21862765,17505322,\
           49703457,51989781,63982162,54105705,73199553,27712144,14028450,\
           57895331,88862329,99534636,50330848,14753501,65359048,62069927,\
           73549214,16226155,56551595,14029581,12154538,38929924,19960712,\
           85095147,72225765,25708618,28371123,55480794,21371248,7507139,\
           80070951,61317037,83546642,41962927,83218340,4355823,6686600,\
           18774345,84066402,41611436,22633123,45560493,11142569,37733241,\
           67382830,56461630,59719238,65235752,6412769,69435498,94266224,2120562,14276357]



#------------------------------------------------------------------------------
# Globals
#------------------------------------------------------------------------------
compute_cap = 'sm_35'

#------------------------------------------------------------------------------
# Code generation functions
#------------------------------------------------------------------------------

# Generates CUDA code for a given math function
def generate_CUDA_code(fun_name: str, params: list, directory: str) -> str:
  file_name = 'cuda_code_'+fun_name+'.cu'
  with open(directory+'/'+file_name, 'w') as fd:
    fd.write('// Atomatically generated - do not modify\n\n')
    fd.write('#include <stdio.h>\n\n')
    fd.write('__global__ void kernel_1(\n')
    signature = ""
    param_names = ""
    for i in range(len(params)):
      if params[i] == 'double':
        signature += 'double x'+str(i)+','
        param_names += 'x'+str(i)+','
    fd.write('  '+signature)
    fd.write('double *ret) {\n')
    fd.write('   *ret = '+fun_name+'('+param_names[:-1]+');\n')
    fd.write('}\n\n')

    fd.write('extern "C" {\n')
    fd.write('double kernel_wrapper_1('+signature[:-1]+') {\n')
    fd.write('  double *dev_p;\n')
    fd.write('  cudaMalloc(&dev_p, sizeof(double));\n')
    fd.write('  kernel_1<<<1,1>>>('+param_names+'dev_p);\n')
    fd.write('  double res;\n')
    fd.write('  cudaMemcpy (&res, dev_p, sizeof(double), cudaMemcpyDeviceToHost);\n')
    fd.write('  return res;\n')
    fd.write('  }\n')
    fd.write(' }\n\n\n')
  return file_name

# Generates C++ code for a given math function
def generate_CPP_code(fun_name: str, params: list, directory: str) -> str:
  file_name = 'cpp_code_'+fun_name+'.cpp'
  with open(directory+'/'+file_name, 'w') as fd:
    fd.write('// Atomatically generated - do not modify\n\n')
    fd.write('#include <cmath>\n\n')
    fd.write('double cpp_kernel_1( ')
    signature = ""
    param_names = ""
    for i in range(len(params)):
      if params[i] == 'double':
        signature += 'double x'+str(i)+','
        param_names += 'x'+str(i)+','
    fd.write(signature[:-1]+') {\n')
    fd.write('   return '+fun_name+'('+param_names[:-1]+');\n')
    fd.write('}\n\n')
  return file_name

#------------------------------------------------------------------------------
# Compilation & running external programs
#------------------------------------------------------------------------------

def run_command(cmd: str):
  try:
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
  except subprocess.CalledProcessError as e:
    print(e.output)
    exit()

def compile_CUDA_code(file_name: str, d: str):
  global compute_cap 
  shared_lib = d+'/'+file_name+'.so'
  cmd = 'nvcc '+' -arch='+compute_cap+' '+d+'/'+file_name+' -o '+shared_lib+' -shared -Xcompiler -fPIC'
  print('Running:', cmd)
  run_command(cmd)
  return shared_lib

def compile_CPP_code(file_name: str, d: str):
  cmd = 'g++ '+d+'/'+file_name+' -o '+d+'/'+file_name+'.so -shared -fPIC'
  print('Running:', cmd)
  run_command(cmd)

#------------------------------------------------------------------------------
# File and directory creation 
#------------------------------------------------------------------------------

def dir_name():
  return '_tmp_'+socket.gethostname()+"_"+str(os.getpid())

def create_experiments_dir() -> str:
    p = dir_name()
    print("Creating dir:", p)
    try:
        os.mkdir(p)
    except OSError:
        print ("Creation of the directory %s failed" % p)
        exit()
    return p

#------------------------------------------------------------------------------
# Function Classes
#------------------------------------------------------------------------------
class SharedLib:
  def __init__(self, path, inputs):
    self.path = path
    self.inputs = int(inputs)

class FunctionSignature:
  def __init__(self, fun_name, input_types):
    self.fun_name = fun_name
    self.input_types = input_types

#------------------------------------------------------------------------------
# Main driver
#------------------------------------------------------------------------------

#FUNCTION:acos (double)
##FUNCTION:acosh (double)
#SHARED_LIB:./app_kernels/CFD_Rodinia/cuda_code_cfd.cu.so, N
#SHARED_LIB:./app_kernels/backprop_Rodinia/cuda_code_backprop.cu.so, N
def parse_functions_to_test(fileName):
  #function_signatures = []
  #shared_libs = []
  ret = []
  with open(fileName, 'r') as fd:
    for line in fd:
      # Comments
      if line.lstrip().startswith('#'):
        continue
      # Empty line
      if ''.join(line.split()) == '':
        continue

      if line.lstrip().startswith('FUNCTION:'):
        no_spaces = ''.join(line.split())
        signature = no_spaces.split('FUNCTION:')[1]
        fun = signature.split('(')[0]
        params = signature.split('(')[1].split(')')[0].split(',')
        ret.append(FunctionSignature(fun, params))
        #function_signatures.append((fun, params))

      if line.lstrip().startswith('SHARED_LIB:'):
        lib_path = line.split('SHARED_LIB:')[1].split(',')[0].strip()
        inputs = line.split('SHARED_LIB:')[1].split(',')[1].strip()
        #shared_libs.append((lib_path, inputs))
        ret.append(SharedLib(lib_path, inputs))

  #return (function_signatures, shared_libs)
  return ret

# Namespace(af='ei', function=['./function_signatures.txt'], number_sampling='fp', range_splitting='many', samples=30)
def areguments_are_valid(args):
  if args.af != 'ei' and args.af != 'ucb' and args.af != 'pi':
    return False
  if args.samples < 1:
    return False
  if args.range_splitting != 'whole' and args.range_splitting != 'two' and args.range_splitting != 'many':
    return False
  if args.number_sampling != 'fp' and args.number_sampling != 'exp':
    return False
  return True

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Xscope tool')
  parser.add_argument('function', metavar='FUNCTION_TO_TEST', nargs=1, help='Function to test (file or shared library .so)')
  parser.add_argument('-a', '--af', default='ei', help='Acquisition function: ei, ucb, pi')
  parser.add_argument('-n', '--number-sampling', default='fp', help='Number sampling method: fp, exp')
  parser.add_argument('-r', '--range-splitting', default='many', help='Range splitting method: whole, two, many')
  parser.add_argument('-s', '--samples', type=int, default=30, help='Number of BO samples (default: 30)')
  parser.add_argument('--random_sampling', action='store_true', help='Use random sampling')
  parser.add_argument('-rid', '--rid', type=int, default=0, help='id of runs')
  parser.add_argument('--random_sampling_unb', action='store_true', help='Use random sampling unbounded')
  parser.add_argument('--dfeg_sampling', action='store_true', help='Use dfeg sampling')
  parser.add_argument('-c', '--clean', action='store_true', help='Remove temporal directories (begin with _tmp_)')
  args = parser.parse_args()

  # --------- Cleaning -------------
  if (args.clean):
    print('Removing temporal dirs...')
    this_dir = './'
    for fname in os.listdir(this_dir):
      if fname.startswith("_tmp_"):
        #os.remove(os.path.join(my_dir, fname))
        shutil.rmtree(os.path.join(this_dir, fname))
    exit()

  # --------- Checking arguments for BO approach ---------
  if (not areguments_are_valid(args)):
    print('Invalid input!')
    parser.print_help()

  input_file = args.function[0]
  functions_to_test = []
  if input_file.endswith('.txt'):
    functions_to_test = parse_functions_to_test(input_file)
  else:
    exit()

  # Create directory to save experiments
  d = create_experiments_dir()
  rid = args.rid

  # --------------- BO approach -----------------
  # Set BO  max iterations
  analysis.set_max_iterations(args.samples)

  # Generate CUDA and compile them
  print(len(rd_seed))
  #for rd in rd_seed[0:10]: 
  rd = rd_seed[rid] 
  results_save = []
  inps_num = []
  for i in functions_to_test:
    if type(i) is FunctionSignature:
      f = generate_CUDA_code(i.fun_name, i.input_types, d)
      shared_lib = compile_CUDA_code(f, d)
      num_inputs = len(i.input_types)
    elif type(i) is SharedLib:
      shared_lib = i.path
      num_inputs = i.inputs
    # Random Sampling

    inps_num.append(num_inputs)
    if args.random_sampling or args.random_sampling_unb:
      print('******* RANDOM SAMPLING on:', shared_lib)
      # Total samples per each input depends on:
      # 18 ranges, 30 max samples (per range), n inputs
      inputs = num_inputs
      max_iters = 600 * int(math.pow(9, inputs))
      unbounded = False
      if args.random_sampling_unb:
        unbounded = True
      start_time = time.time()
      analysis.optimize_randomly(shared_lib, inputs, max_iters, unbounded)
      analysis.print_results_random(shared_lib)
      end_time = time.time()
      print("time is "+str(end_time-start_time))
    # dfeg Sampling
    if args.dfeg_sampling:
      #print('******* DFEG SAMPLING on:', shared_lib)
      inputs = num_inputs
      max_iters = 30 * int(math.pow(18, inputs))
      unbounded = True
      start_time = time.time()
      start_time = time.time()
      signal.signal(signal.SIGALRM, timeout_handler)
      signal.alarm(6*num_inputs)
      try:
        analysis.optimize_dfegly(shared_lib, inputs, max_iters, unbounded,rd,rid)
      finally:
        signal.alarm(0)
      signal.alarm(0)
      analysis.print_results_dfeg(shared_lib)
      inputs_res = list(analysis.dfeg_inp_results.items())
      inputs_res_mc = list(analysis.dfeg_inp_results_mc.items())
      analysis.dfeg_inp_results = {"inf+":[],"inf-":[],"sub+":[],"sub-":[],"nan":[]}
      analysis.dfeg_inp_results_mc = {"inf+":[],"inf-":[],"sub+":[],"sub-":[],"nan":[]}
      end_time = time.time()
      exec_time = end_time-start_time
      print("time is "+str(end_time-start_time))
      if type(i) is FunctionSignature:
          results_save.append([list(analysis.dfeg_results.items()),exec_time,i.fun_name,inputs_res,list(analysis.dfeg_results_mc.items()),inputs_res_mc,analysis.fine_search_time])
      elif type(i) is SharedLib:
          results_save.append([list(analysis.dfeg_results.items()),exec_time,i.path,inputs_res,list(analysis.dfeg_results_mc.items()),inputs_res_mc,analysis.fine_search_time])
      analysis.dfeg_results = {}
      analysis.dfeg_results_mc = {}
      analysis.fine_search_time = 0
  save_line_list("results/dfeg_res/dfeg_res_"+str(rid)+".pkl",results_save)
