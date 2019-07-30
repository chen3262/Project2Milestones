# Project2Milestones
Projecting Molecular Dynamics trajectories into cells between milestones. Accelerated by MPI using mpi4py.

<img src ="https://github.com/chen3262/Project2Milestones/master/demo.png" width="600">

## Requirements
python modules: ```numpy```, ```pytraj```, ```matplotlib```, ```mpi4py```

To check if you have these modules installed (excepting ```pytraj```), you can either do
```bash
conda list | grep "module_name"
pip list | grep "module_name"
```
If nothing is shown, you will need to install the module using either of the following commands
```bash
conda install module_name
pip install module_name
```
To install ```pytraj```, please do either of the following commands
```bash
conda install -c ambermd pytraj
pip install -i https://pypi.anaconda.org/ambermd/simple pytraj
```

## To use the script
Provide the path of the file containing PC1 and PC2 components of MD trajectories in the ```OptMilestone_MPI.py```.

## License

Copyright (C) 2019 Si-Han Chen chen.3262@osu.edu
