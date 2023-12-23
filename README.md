# Deep JSCC
This implements training of deep JSCC models for wireless image transmission as described in the paper [Deep Joint Source-Channel Coding for Wireless Image Transmission](https://ieeexplore.ieee.org/abstract/document/8723589) by Pytorch. And there has been a [Tensorflow and keras implementations ](https://github.com/irdanish11/DJSCC-for-Wireless-Image-Transmission).

This is my first time to use PyTorch and git to reproduce a paper, so there may be some mistakes. If you find any, please let me know. Thanks!
## Architecture

![architecture](./demo/arc.png)

## Demo
I spend 3 days from 12-20 to 12-24 to reproduce the paper, and i get the result as follow. The result is not good, because i trained the model on cifar10 which is 32*32 but test on kodim which is 768*512 and the model is not trained enough. 
That is all enough!!！
![demo](./demo/demo.png)


## Installation
conda or other virtual environment is recommended.

```
git clone https://github.com/chunbaobao/Deep-JSCC-PyTorch.git
pip install requirements.txt
```

## Usage
### Training Model
Run(example presented in paper)
```
cd ./Deep-JSCC-PyTorch
```

```
python train.py --lr 10e-4 --epochs 100 --batch_size 32 --channel 'AWGN' --saved ./saved --snr_list 1 4 7 13 19 --ratio_list 1/6 1/12 --dataset imagenet
```
or
```
python train.py --lr 10e-3 --epochs 100 --batch_size 512 --channel 'AWGN' --saved ./saved --dataset cifar10 --num_workers 4 --parallel True
```
### Evaluation
Run(example presented in paper)
```
python eval.py --channel 'AWGN' --saved ./saved/${mode_path} --snr 20 --ratio_list 1/3 --test_img ./test_image ./demo/kodim08.png
```


## Citation
If you find (part of) this code useful for your research, please consider citing
```
@misc{chunhang_Deep-JSCC,
  author = {chunhang},
  title = {a pytorch implementation of Deep JSCC},
  url ={https://github.com/chunbaobao/Deep-JSCC-PyTorch},
  year = {2023}
}

