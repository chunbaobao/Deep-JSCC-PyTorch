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
        return img, 0  # 0 is a fake label not important

    def __len__(self):
        return len(self.imgs)


def main():
    data_path = './dataset'
    os.makedirs(data_path, exist_ok=True)
    # ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar should be downloaded from https://image-net.org/
    if not os.path.exists('./dataset/ILSVRC2012_img_train.tar') or not os.path.exists('./dataset/ILSVRC2012_img_val.tar'):
        print('ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar should be downloaded from https://image-net.org/')
        print('Please download the dataset from https://image-net.org/challenges/LSVRC/2012/2012-downloads and put it in ./dataset')
        raise Exception('not find dataset')
    phases = ['train', 'val']
    for phase in phases:
        print("extracting {} dataset".format(phase))
        path = './dataset/ImageNet/{}'.format(phase)
        print('path is {}'.format(path))
        os.makedirs(path, exist_ok=True)
        print('tar -xf ./dataset/ILSVRC2012_img_{}.tar -C {}'.format(phase, path))
        os.system('tar -xf ./dataset/ILSVRC2012_img_{}.tar -C {}'.format(phase, path))
        if phase == 'train':
            for tar in os.listdir(path):
                print('tar -xf {}/{} -C {}/{}'.format(path, tar, path, tar.split('.')[0]))
                os.makedirs('{}/{}'.format(path, tar.split('.')[0]), exist_ok=True)
                os.system('tar -xf {}/{} -C {}/{}'.format(path, tar, path, tar.split('.')[0]))
                os.remove('{}/{}'.format(path, tar))


if __name__ == '__main__':
    main()
