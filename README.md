# DeepFRI: A Deep Plug-and-Play Technique for Finite-Rate-of-Innovation Signal Reconstruction
This repository contains a reproducible PyTorch implementation for the paper: https://ieeexplore.ieee.org/abstract/document/11080367. 

Code authors:
[Sharan Basav Patil](mailto:sharanbasav23@gmail.com) and [Abijith J. Kamath](mailto:abijithj@iisc.ac.in), Indian Institute of Science

### 1. Requirements

This work requires a modified version of the ComplexPyTorch library (coming soon).

### 2. Prepare training data

Generate training and testing dataset using
``` bash
python3 generate_dataset.py --N 21 --K 2 --gamma 5 --T 1.0 --num_data 100000 --mode train
python3 generate_dataset.py --N 21 --K 2 --gamma 5 --T 1.0 --num_data 10000 --mode test

```

### 3. Training example

Train using
``` bash
python3 main.py
```

### 4. Evaluation example

Test using
``` bash
python3 test_deepfri.py
```

<!-- ### 5. Data and examples for ultrasound imaging -->

## Citation
If you use code from this repository, please cite the following:

``` bash
@article{kamath2025deepfri,
	author={Kamath, Abijith Jagannath and Patil, Sharan Basav and Seelamantula, Chandra Sekhar},
	journal={IEEE Transactions on Signal Processing}, 
	title={DeepFRI: A Deep Plug-and-Play Technique for Finite-Rate-of-Innovation Signal Reconstruction}, 
	year={2025},
	volume={73},
	number={},
	pages={2998-3013},
	doi={10.1109/TSP.2025.3589394}
}
```

## Contact

For further questions, contact:

[Abijith J. Kamath](mailto:abijithj@iisc.ac.in)<br>
Department of Electrical Engineering<br>
Indian Institute of Science

![Spectrum Lab logo](assets/spectrum.png)

---

