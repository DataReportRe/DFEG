# How to Reproduce DFEG Results

## Setting up the Docker Image

We provide a docker image with all the requirements to reproduce the key results in our paper.

### Step 0: Obtain the Docker Image

First, make sure the NVIDIA Container Toolkit is installed. See [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

Pull the image:

```
$ docker pull datareport/dfeg_im:latest
```

### Step 1: Run the Image

To run the docker image, use the following:

```
$ sudo docker run --runtime=nvidia --gpus all -it datareport/dfeg_im:latest /bin/bash
```

### Step 2:

When you are in the bash shell, move to the dfeg directory and untar the source code.

```
$ cd /dfeg
$ tar -xf dfeg.tgz 
```

Now move to the source code directory:

```
$ cd dfeg 
```

## Execution

We provide easy-to-use scripts to run the key experiments of the paper. The artifact's experimental workflow consists of two tasks:  $T_1$:Compiling Benchmarks; $T_2$:Running the experiments.

#### $T_1$: Compiling Benchmarks
Navigate to the directory (app_kernels) containing the source code of benchmarks and proceed with the compilation process

```
$ cd dfeg/app_kernels
$ make
```


#### $T_2$: Running the experiments


The main script is `dfeg.py`, which takes several options:

```
$ ./dfeg.py --dfeg_sampling func_test.txt --rid 1
```
The input file “func test.txt” contains all functions to be tested. To exclude a function from testing, add “#” at the beginning of the line corresponding to that function in “func test.txt”. The “rid” sets the ID of the random seed,
which also serves as the suffix for the output file.
The terminal will display temporary results for each function as follows:

```
-------------- Results --------------
cuda_code_matrixDet.cu.so
        INF+: 295874
        INF-: 294901
        SUB+: 587
        SUB-: 592
        NaN : 229941

--------------MCMC Results --------------
        INF+: 0
        INF-: 90
        SUB+: 0
        SUB-: 0
        NaN : 0
```
To reproduce main results in out paper, we provide easy-to-use scripts to run the key experiments of the paper. Just run:

```
$ test.sh
```

Results will be saved into “results/dfeg_res/dfeg_res_1.pkl”.
To do the ablation analysis, just run:

```
./dfeg_ablation.py --dfeg_sampling func_ablation_test.txt --rid 101 --ablation WOFP
```
All results will be saved into ``results/dfeg\_res\_ablation /dfeg\_res\_WOFP.pkl''.

To do all ablation analysis in Table 5 of our paper, just run:

```
$ ./test_ablation.sh
```

Results will be saved into ``results/dfeg\_res\_ablation/''.

## Analysis results

To get Table 3,4 and Figure 3-5 in our paper, run script


```
$ cd results
$ python3 extract_datas.py
```


Table 3 and 4 will be stored in “res_table.xls”, Figure 3 will be stored in “compare_typesX.pdf”, Figure 4 will be stored in “plot_domain11cosh.png”, and Figure 5 will be stored in “scala_compare.pdf”.

To get Table 5 in our paper, run script

```
$ python3 extract_datas_ablation.py
```

Table 5 will be saved in ``res\_table\_ablation.xls''.


# Get results of other options of xscope


Into "results" and run:

```
$python3 extract_datas_appendix.py
```

Results be saved in res_table_[fptwo|fpwhole|expmany|exptwo|expwhole].xls". 


# Run Xscope (Optional)
Move to the dfeg directory and untar the source code.

```
$ cd /dfeg
$ tar -xf xscope.tgz 
```

Now move to the source code directory:

```
$ cd xscope
```
Run the test script:

```
$./test.sh
```

Results will be saved to "xscope_res"

### Run Xscope for other options "fp|two,fp|whole,exp|many,exp|two,exp|whole"
Run the test script:

```
$./optionstest.sh
```
Results will be saved to "xscope_res_appendix"





