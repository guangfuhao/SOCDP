# CpmCode

This is the code of our method for paper **Exploring Role Division in Neural Network Structures**.

## Training

* train all configurations with multiple GPUs on the server

```
sbatch run.slurm
```

* train all configurations with single GPU

```
python <your_path_here>/codes_py/main.py --d "<your_path_here>/datasets/mnist" --n 100 --batch-size 64 --ms <change from 16 to 256, one at once> --cnr 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30,35,40,45,50 --save-path "<your_path_here>/codes_py/output/"
```

## Show results

Use [jupyter_plot.ipynb](https://github.com/guangfuhao/CPM_code/blob/main/codes_py/jupyter_plot.ipynb) to get all analysis results in the paper.
