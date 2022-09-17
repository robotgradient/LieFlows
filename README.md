# Pytorch Implementation of Stable Vector Fields on Lie Groups.
The following code repository implements stable vector fields on Lie groups for Robotics.
Code repository in relation with the ICRA 2023-RA-L submission 'Learning Stable Vector Fields
on Lie Groups'.
[[Preprint]](https://arxiv.org/pdf/2110.11774.pdf)



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
