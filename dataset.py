import os


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
        for tar in os.listdir(path):
            print('tar -xf {}/{} -C {}/{}'.format(path, tar, path, tar.split('.')[0]))
            os.system('tar -xf {}/{} -C {}/{}'.format(path, tar, path, tar.split('.')[0]))
            os.remove('{}/{}'.format(path, tar))


if __name__ == '__main__':
    main()
