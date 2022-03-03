import argparse
import os
import numpy as np
id=[0,1,3,4,6,7,8,10,11,12,14,15,16]
num=[64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512,512]
parser = argparse.ArgumentParser()
parser.add_argument('-z_partition_id', type=int, default=0)
parser.add_argument('-z_pruning_num', type=int, default=64)
parser.add_argument('-z_quantization', type=int, default=8)
parser.add_argument('-gpu', type=int, default=0)
args = parser.parse_args()


z_partition_id=id[args.z_partition_id]
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

os.system('cp ./0+64+8+0.151/best_model.pt ./logs/')
os.system('python logs/z.py '+'-z_num '+str(args.z_pruning_num)+ ' -id '+ str(z_partition_id)) #修改pt的模型结构
os.system('cp ./apps/vgg0.yml ./apps/vgg.yml')
with open('./apps/vgg.yml','a') as f:
	f.write('z_partition_id: '+str(z_partition_id)+'\n')
	f.write('z_pruning_num: '+str(args.z_pruning_num)+'\n')
	f.write('z_quantization: '+str(args.z_quantization)+'\n')
	f.write('gpu: '+str(args.gpu)+'\n')
os.system('python train.py app:apps/vgg.yml')
