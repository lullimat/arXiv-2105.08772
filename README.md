# A Mesoscale Perspective on the Tolman Length

**This repository is being constantly updated !!**

**The first version of the Jupyter notebook is now available**

**The Supplemental Material figures will be added soon**

Welcome to the repository related to the paper [https://arxiv.org/abs/2105.08772](https://arxiv.org/abs/2105.08772).
In order to use this repository you should have cloned it from the parent project [idea.deploy](https://github.com/lullimat/idea.deploy) in the directory ./papers by means of the local python script idpy-papers.py.

In order to open the present Jupyter notebook "MesoscopicTolmanLength.ipynb" you should perform the following steps
- install the [idea.deploy](https://github.com/lullimat/idea.deploy) project, following the instructions in the README.md in the section "Installation"
- load the idea.deploy python virtual environment: if you installed the bash aliases for the [idea.deploy](https://github.com/lullimat/idea.deploy) project you can issue the command "idpy-load"
- launch locally the Jupyter server with "idpy-jupyter"
- copy and paste in your browser the url prompted in the terminal in order to open the Jupyter server interface
- click the file "MesoscopicTolmanLength.ipynb"
- wait for all the extension to be loaded: when the notebook is loaded you should see a "Table of Contents" on the left side and the different code cells in the "folded" mode with a small grey arrow on the left

In order to excute the code content of a cell, select it and enter the key combination "shift + enter"
As of today, after much testing Google Chrome has offered the most reliable expreience in handling the notebook.

## Dependencies
A part from the python dependencies which are taken care by the [idea.deploy](https://github.com/lullimat/idea.deploy) installation, a working "latex" environment needs to be installed on the system in order to reproduce the plots which contain latex symbols

## Hardware Requirements
The largest sizes of the simulations need a rather capable hardware with roughly 16GB of allocatable memory. The published results for such systems have been computed from simulations performed on NVIDIA P100 GPUs.

