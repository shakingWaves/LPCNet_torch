'''
预处理
'''
import os
import argparse
import random

def get_file_list(dir):
    files = os.listdir(dir)
    val_files = []
    for file in files:
        if not (os.path.isdir(file) or file.endswith("u8")):
            val_files.append(dir + '/' +file)
    return val_files

def split_file_set(files, split_ratio):
    random.shuffle(files)
    sr = split_ratio.split(":")
    size = len(files)
    ratio = [int(s)/10 * size  for s in sr]

    sr = []
    cnt = 0
    for idx in range(len(ratio) - 1):
        sr.append(int(ratio[idx]))
        cnt+=int(ratio[idx])
    sr.append(size - cnt)

    pos = 0
    file_split = []
    for r in sr:
        file_split.append(files[pos:pos+ r])
        pos+=r
    return file_split

def write_file(outdir, file_split):
    train_set=["/train.txt", "/test.txt", "/validate.txt"]
    for idx in range(len(file_split)):
        with open(outdir + train_set[idx], 'w', encoding='utf-8') as f:
            for file_name in file_split[idx]:
                prefix = file_name.rsplit('.', 1)
                pcm_file = prefix[0] + ".u8"
                f.write(file_name+" "+pcm_file+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', '--feature_directory', type=str,
                        default="E:/Research/Synthesis/BZNSYP/Wave48_Feature_Pytorch/",
                        required=False, help='comma separated name=value pairs')
    #输出目录
    parser.add_argument('-outdir', '--out_directory', type=str,
                        default="E:/Research/Synthesis/LPCNet-carlfm01.48k.torch/training_data/",
                        required=False, help='comma separated name=value pairs')

    parser.add_argument('-sr', '--split_ratio', type=str,
                        default="8:1:1",
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    files = get_file_list(args.feature_directory)
    file_split = split_file_set(files, args.split_ratio)
    write_file(args.out_directory, file_split)


if __name__=='__main__':
    main()