import os
from torch.utils.data import Dataset
from PIL import Image


class Vanilla(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = os.listdir(root)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.imgs[index])
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, 0 # 0 is a fake label not important

    def __len__(self):
        return len(self.imgs)


def main():
    data_path = './Dataset'
    os.makedirs(data_path, exist_ok=True)

    if not os.path.exists('./Dataset/ILSVRC2012_img_train.tar') or not os.path.exists('./Dataset/ILSVRC2012_img_val.tar'):
        print('Please download the dataset from http://www.image-net.org/challenges/LSVRC/2012/downloads and put it in ./Dataset')
        raise Exception('not find dataset')
    phases = ['train', 'val']
    for phase in phases:
        print("extracting {} dataset".format(phase))
        path = './Dataset/ImageNet/{}'.format(phase)
        print('path is {}'.format(path))
        os.makedirs(path, exist_ok=True)
        print('tar -xf ./Dataset/ILSVRC2012_img_{}.tar -C {}'.format(phase, path))
        os.system('tar -xf ./Dataset/ILSVRC2012_img_{}.tar -C {}'.format(phase, path))
        if phase == 'train':
            for tar in os.listdir(path):
                print('tar -xf {}/{} -C {}/{}'.format(path, tar, path, tar.split('.')[0]))
                os.makedirs('{}/{}'.format(path, tar.split('.')[0]), exist_ok=True)
                os.system('tar -xf {}/{} -C {}/{}'.format(path, tar, path, tar.split('.')[0]))
                os.remove('{}/{}'.format(path, tar))


if __name__ == '__main__':
    main()
