# Deep JSCC
This implements training of deep JSCC models for wireless image transmission as described in the paper [Deep Joint Source-Channel Coding for Wireless Image Transmission](https://ieeexplore.ieee.org/abstract/document/8723589) by Pytorch. And there has been a [Tensorflow and keras implementations ](https://github.com/irdanish11/DJSCC-for-Wireless-Image-Transmission).

This is my first time to use PyTorch and git to reproduce a paper, so there may be some mistakes. If you find any, please let me know. Thanks!

## Installation
conda or other virtual environment is recommended.

```
git clone https://github.com/chunbaobao/Deep-JSCC-PyTorch.git
pip install requirements.txt
```

## Usage
### Training Model
Run(example)
```
cd ./Deep-JSCC-PyTorch
python train.py --seed 2048 --epochs 200 --batch_size 256 --channel 'AWGN' --saved ./saved --snr_list [1,4,7,13,19] --ratio_list [1/6,1/12] --dataset imagenet
```

### Evaluation


## Citation
If you find (part of) this code useful for your research, please consider citing
```
@misc{chunhang_Deep-JSCC,
  author = {chunhang},
  title = {a pytorch implementation of Deep JSCC},
  url ={https://github.com/chunbaobao/Deep-JSCC-PyTorch},
  year = {2023}
}

