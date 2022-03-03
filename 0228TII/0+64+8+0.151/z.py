import torch as t
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-z_num', type=int, default=0)
parser.add_argument('-gpu', type=int, default=0)
parser.add_argument('-id', type=int, default=0)
args = parser.parse_args()
z_module=[0,3,7,10,14,17,20,24,27,30,34,37,40]
vgg_id=[0,1,3,4,6,7,8,10,11,12,14,15]
z_id=str(z_module[vgg_id.index(args.id)])
z_id2=str(z_module[vgg_id.index(args.id)]+1)
z_id3=str(z_module[vgg_id.index(args.id)+1])
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
if os.path.isfile('./logs/best_model.pt'):
	checkpoint = t.load('./logs/best_model.pt', map_location=lambda storage, loc: storage)
	checkpoint=checkpoint['model']
	checkpoint2={}
	z_compression=args.z_num
	for k,v in checkpoint.items():
		if k=='module.features.'+z_id+'.weight' or k=='module.features.'+z_id+'.bias' or k=='module.features.'+z_id2+'.weight' or k=='module.features.'+z_id2+'.bias' or k=='module.features.'+z_id2+'.running_var' or k=='module.features.'+z_id2+'.running_mean':
			checkpoint2[k]=v[0:z_compression]
		elif k=='module.features.'+z_id3+'.weight':
			checkpoint2[k]=v[:,0:z_compression]
		else:
			checkpoint2[k]=v
	t.save(checkpoint2,'./logs/best_model.pt')
