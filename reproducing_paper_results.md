# How to Reproduce DFEG Results

## Setting up the Docker Image

We provide a docker image with all the requirements to reproduce the key result of in our paper.

### Step 0: Obtain the Docker Image

First, make sure the NVIDIA Container Toolkit is installed. See [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

Pull the image:

```
$ docker pull datareport/dfeg_im:v1.0
```

### Step 1: Run the Image

To run the docker image, use the following:

```
$ sudo docker run --runtime=nvidia --gpus all -it datareport/dfeg_im:v1.0 /bin/bash
```

### Step 2:

When you are in the bash shell, move to the dfeg directory and untar the source code.

```
$ cd /dfeg
$ tar -xf dfeg_source.tgz 
```

Now move to the source code directory:

```
$ cd dfeg 
$ ls
README.md results app_kernels  analysis.py dfeg.py dfeg_bo.py functions_to_test.txt  random_fp_generator.py  xscope.py
```

## Running the Experiments

We provide easy-to-use scripts to run the key experiments of the paper. These include the results presented in Table 3, Figures 3 to 8.

The main script is `dfeg`, which takes several options:

```
$ ./dfeg --dfeg_sampling functions_to_test.txt --rid 1
```
The input file "functions_to_test.txt" includes all functions to be tested. A function can be exclude to be tested by add "#" at begin of the line of the function. Results will be saved into "results/dfeg_res/dfeg_res_1.pkl":
#### Results of Table 2 and Figures 3 to 8. 

Run the shell script:
```
$ ./test_100_trials.sh
```

It will run  100 times of dfeg over 69 functions that be tested in paper, and save results "dfeg_res_0.pkl" to "dfeg_res_99.pkl" into "results/dfeg_res".


Into "results" and run:

```
$python3 extract_datas.py
```

Table 2 will be saved in "res_table.xls" and figures 3 to 8 will be save into "graph".

# Get appendix results


Into "results" and run:

```
$python3 extract_datas_appendix.py
```

Table 4 to 8 will be saved in res_table_[fptwo|fpwhole|expmany|exptwo|expwhole].xls". 


# Paper datas
All paper datas store into "results/paper_dates", and you can run:
```
$python3 extract_datas.py
```
in the "results/paper_dates" without need to rerun "dfeg" to analysis datas for paper.

# Run DFEG_BO
To run DFEG_BO in section 4.5 of paper:

```
$python3 dfeg_bo.py --dfeg_sampling functions_to_test.txt --rid 0
```

Results will be stored in "results/dfeg_bo_res"

# Run Xscope "fp|many"
Move to the dfeg directory and untar the source code.

```
$ cd /dfeg
$ tar -xf xscope_source.tgz 
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

# Run Xscope for other options "fp|two,fp|whole,exp|many,exp|two,exp|whole"
Run the test script:

```
$./optionstest.sh
```
Results will be saved to "xscope_res_appendix"





