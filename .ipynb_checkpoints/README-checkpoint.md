# Teacher-Student Networks with Multiple Decoders for Solving Math Word Problem.

PyTorch implementation of Graph based Math Word Problem solver described in our IJCAI 2020 paper Teacher-Student Networks with Multiple Decoders for Solving Math Word Problem. In this work, we propose an enhancement method for Math Word Problem Solving systems.

## Steps to run the experiments

### Requirements
* ``Python 3.6 ``
* ``>= PyTorch 1.0.0``

For more details, please refer to requiremnt file.

### Training
We have provided the pre-processed soft target file in corresponding ``data`` directory. If you want to extract soft target from your own model, you can refer to ``math23k/seq2tree_save_softtarget.py`` and adapt it based on your code.
#### [MATH23K]
first get into the math23k directory:
* ``cd math23k``

training-test setting :
* ``python run_seq2tree_diverse.py``

cross-validation setting :
It's easy to modify ``run_seq2tree_diverse.py`` and adapt it to cross-validation setting.

#### [MAWPS]
cross-validation setting :
* ``cd mawps``
* ``python run_seq2tree_diverse.py``

### Reference
```
@article{zhang2020tsnmd,
  title={Graph-to-Tree Learning for Solving Math Word Problems},
  author={Jipeng Zhang, Roy Ka-Wei Lee, Ee-Peng Lim, Wei Qin, Lei Wang, Jie Shao and Qianru Sun},
  journal={IJCAI 2020},
  year={2020}
}
```
