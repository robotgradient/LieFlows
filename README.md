# Pytorch Implementation of Stable Vector Fields on Lie Groups.
The following code repository implements stable vector fields on Lie groups for Robotics.
Code repository in relation with the ICRA 2023-RA-L submission 'Learning Stable Vector Fields
on Lie Groups'.

[[Preprint]](https://arxiv.org/pdf/2110.11774.pdf)
[[Webpage]](https://sites.google.com/view/svf-on-lie-groups/?pli=1)

<img src="figures/main.gif" alt="main" style="width:800px;"/>

## Installation

Build Conda Environment

```angular2html
 conda env create -f environment.yml
```

activate environment and install library

```
pip install -e .
```

## Examples

### S2 Stable Vector Fields
```angular2html
python scripts/s2_svf/test_s2_model/visualize_vector_field.py
```
### SE(2) Stable Vector Fields
```angular2html
python scripts/se2_svf/test_trained_models/load_and_test.py
```
### SE(3) Stable Vector Fields
```angular2html
python scripts/se3_svf/test_trained_models/load_and_test.py
```

## References

[1] Julen Urain*, Davide Tateo, Jan Peters. 
"Learning Stable Vector Fields on Lie Groups" 
RA-L 2022.
[[arxiv]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9935105)

```
@article{urain2022liesvf,
  title={Learning Stable Vector Fields on Lie Groups},
  author={Urain, Julen and Tateo, Davide and Peters, Jan},
  journal={IEEE Robotics and Automation Letters (RA-L)},
  year={2022}
```


### Contact

If you have any questions or find any bugs, please let me know: [Julen Urain](http://robotgradient.com/) julen[at]robot-learning[dot]de

