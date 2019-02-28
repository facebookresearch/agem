# Efficient Lifelong Learning with A-GEM

This is the official implementation of the [Averaged Gradient Episodic Memory (A-GEM)](https://arxiv.org/abs/1812.00420) and [Experience Replay with Tiny Memories](http://arxiv.org/abs/1902.10486) in Tensorflow.

## Requirements

TensorFlow >= v1.9.0.

## Training

To replicate the results of the paper on a particular dataset, execute (see the Note below for downloading the CUB and AWA datasets):
```bash
$ ./replicate_results_iclr19.sh <DATASET> <THREAD-ID> <JE>
```
Example runs are:
```bash
$ ./replicate_results_iclr19.sh MNIST 3      /* Train PNN and A-GEM on MNIST */
$ ./replicate_results_iclr19.sh CUB 1 1      /* Train JE models of RWALK and A-GEM on CUB */
```

### Note
For CUB and AWA experiments, download the dataset prior to running the above script. Run following for downloading the datasets:

```bash
$ ./download_cub_awa.sh
```
The plotting code is provided under the folder `plotting_code/`. Update the paths in the plotting code accordingly.
 
## Experience Replay
 The code provides an implementation of experience replay (ER) with reservoir sampling on MNIST and CIFAR datasets. To run the ER experiments execute the following script:
```bash
$ ./replicate_results_er.sh
```

When using this code, please cite our papers:

```
@inproceedings{AGEM,
  title={Efficient Lifelong Learning with A-GEM},
  author={Chaudhry, Arslan and Ranzato, Marc’Aurelio and Rohrbach, Marcus and Elhoseiny, Mohamed},
  booktitle={ICLR},
  year={2019}
}

@article{chaudhryER_2019,
  title={Continual Learning with Tiny Episodic Memories},
  author={Chaudhry, Arslan and Rohrbach, Marcus and Elhoseiny, Mohamed and Ajanthan, Thalaiyasingam and Dokania, Puneet K and Torr, Philip HS and Ranzato, Marc’Aurelio},
  journal={arXiv preprint arXiv:1902.10486, 2019},
  year={2019}
}

@inproceedings{chaudhry2018riemannian,
  title={Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence},
  author={Chaudhry, Arslan and Dokania, Puneet K and Ajanthan, Thalaiyasingam and Torr, Philip HS},
  booktitle={ECCV},
  year={2018}
}
```

## Questions/ Bugs
* For questions, contact the author Arslan Chaudhry (arslan.chaudhry@eng.ox.ac.uk).
* Feel free to open the bugs if anything is broken.

## License
This source code is released under The MIT License found in the LICENSE file in the root directory of this source tree. 
