#coding=utf-8
import numpy
import random
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
import random
import os
from numpy import argmax
import argparse
global accuracy_loss_constraint, latency_constraint,optimal_scheme,avaliable_scheme,avaliable_evaluated_scheme
import warnings
warnings.filterwarnings("ignore")
# User define the following parameters, 10000 as a loose constraint means the constraint actually has no influence
parser = argparse.ArgumentParser()
parser.add_argument('-accuracy_or_latency', type=bool, default=False)
parser.add_argument('-accuracy_loss', type=float, default=10000)
parser.add_argument('-latency', type=float, default=10000)
parser.add_argument('-bandwidth', type=float, default=1)
parser.add_argument('-gpu', type=int, default=0)
args = parser.parse_args()
data_base_address='./latency_search.txt'

#The latency profiler of the IoT-Cloud system
bandwidth=args.bandwidth/8 #MB/s
accuracy_loss_base = 0.151
only_pi_latency=3.2571
only_cloud=360*240*3/1024/1024/bandwidth
base_transmission=224*224/8/1024/1024 #MB
layer=[64, 64,'M', 128, 128,'M', 256, 256, 256,'M', 512, 512, 512,'M', 512, 512, 512]

pi_layer_latency=[]
pi_latency=[0.0597,0.0937,0.1179,0.5048,0.5054,0.5074,0.51,0.5382,0.5557,0.5682,0.9844,1.0018,1.0142,1.055,1.2156,1.2246,1.2318,1.5411,1.5501,1.5577,1.8679,1.8768,1.8845,1.9072,2.0437,2.0484,2.0532,2.3183,2.3231,2.3279,2.5918,2.5966,2.6014,2.6134,2.6898,2.6913,2.6938,2.7694,2.7709,2.7735,2.8487,2.8503,2.8528,2.8569,3.2571]
gpu_layer_latency=[]
gpu_latency=[ 0.0001,    0.0002,    0.0002,    0.0003,    0.0003,    0.0004,    0.0004,    0.0005,    0.0006,    0.0006,    0.0008,    0.0008,    0.0008,    0.0009,    0.0010,    0.0010,    0.0011,    0.0011,    0.0012,    0.0012,    0.0013,    0.0014,    0.0014,    0.0014,    0.0015,    0.0018,    0.0019,    0.0020,    0.0020,    0.0020,    0.0021,    0.0022,    0.0022,    0.0023,    0.0023,    0.0024,    0.0024,    0.0025,    0.0026,    0.0026,    0.0027,    0.0028,    0.0028,    0.0029,    0.0032]
layer_latency_index=[3,7,10,14,17,20,24,27,30,34,37,40,44]


def gaussin_process(avaliable_evaluated_scheme):
    kernel = Matern(length_scale=[1,1,1],length_scale_bounds=[(1e-10,1e10),(1e-10,1e10),(1e-5,1e10)],nu=0.01)
    reg = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=100,alpha=1e-5,normalize_y=True)
    scheme_tmp=[]
    accuracy_tmp=[]
    for i in avaliable_evaluated_scheme:
        scheme_tmp.append(i[0])
        accuracy_tmp.append(i[1])
    for i in range(13):
        scheme_tmp.append([i,0,1])
        accuracy_tmp.append(1.0)
        scheme_tmp.append([i,1,0])
        accuracy_tmp.append(1.0)
        scheme_tmp.append([i,1000,10])
        accuracy_tmp.append(0)
    scheme_tmp=ennormlization(scheme_tmp)
    reg.fit(scheme_tmp, accuracy_tmp)
    return reg

def ennormlization(avaliable_scheme):
	scheme_tmp=[]
	layer_tmp=[64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
	layer_num=len(layer_tmp)
	for i in avaliable_scheme:
		scheme_tmp.append([i[0]/len(layer_tmp),i[1]/layer_tmp[i[0]],i[2]/8])
	return scheme_tmp

# probability of improvement acquisition function
def opt_acquisition(avaliable_scheme, gp_model):
	def acquisition(avaliable_scheme, gp_model):
		Xsamples = ennormlization(avaliable_scheme)
		mu, std = gp_model.predict(Xsamples,return_std=True)
		if accuracy_or_latency_demand:
			if numpy.std(mu)<1e-2:
				probs=-1*numpy.arange(len(mu.tolist()))
			else:
				probs = norm.cdf((accuracy_loss_base+accuracy_loss_constraint-mu) / (std+1E-9))
		else:
			if numpy.std(mu)<1e-2:
				probs=numpy.arange(len(mu.tolist()))
			else:
				probs = norm.cdf((accuracy_loss_base+accuracy_loss_constraint-mu) / (std + 1E-9))
		return probs
	scores = acquisition(avaliable_scheme,gp_model)
	ix = argmax(scores)
	return avaliable_scheme[ix]


for i in layer_latency_index:
    gpu_layer_latency.append(gpu_latency[i-1])
    pi_layer_latency.append(pi_latency[i-1])

def scheme_generation(device_layer_latency):
	global base_transmission
	partition_num = -1
	scheme = []
	transmission_data_size = []
	scheme_latency = []
	scheme_count=-1
	layer_start_index=[]
	layer_count=-1
	layer_tmp=[64, 64,'M', 128, 128,'M', 256, 256, 256,'M', 512, 512, 512,'M', 512, 512, 512,'M']
	for i in layer:
		layer_count+=1
		if i!='M':
			partition_num += 1
			if layer_tmp[layer_count+1]=='M':
				base_transmission=base_transmission/4
			for z in range(i):
				for j in range(1,9):
					scheme.append([partition_num,z+1,j])
					transmission_data_size.append(base_transmission*(z+1)*j)
					scheme_latency.append(device_layer_latency[partition_num]+(gpu_layer_latency[-1]-gpu_layer_latency[partition_num])+(base_transmission*(z+1)*j)/bandwidth)
					scheme_count+=1
			layer_start_index.append(scheme_count)
	return(scheme,transmission_data_size,scheme_latency,layer_start_index)



def filter():
    def find_closet(set, num, bit):
        distance = 10000
        closet = []
        for i, j, k in set:
            distance_tmp = abs(i + j - num - bit)
            if distance_tmp < distance:
                distance = distance_tmp
                closet = [i, j, k]
        return closet
    filter_scheme=[]
    for id,num,bit in avaliable_scheme:
        nlarge = []
        nsmall = []
        blarge = []
        bsmall = []
        accuracy_loss_upper_bound0=10000
        accuracy_loss_upper_bound1=10000
        accuracy_loss_upper_bound2=10000
        monotonicity_accuracy_loss_upper_bound_set = []
        for [id_eva,num_eva,bit_eva],accuracy_loss_eva in avaliable_evaluated_scheme:
            if id==id_eva:
                if (num_eva<=num and bit_eva<=bit):
                    monotonicity_accuracy_loss_upper_bound_set.append(accuracy_loss_eva)
                if num_eva==num:
                    if bit_eva<bit:
                        bsmall.append([num_eva, bit_eva, accuracy_loss_eva])
                    if bit_eva>bit:
                        blarge.append([num_eva, bit_eva, accuracy_loss_eva])
                if bit_eva==bit:
                    if num_eva<num:
                        nsmall.append([num_eva, bit_eva, accuracy_loss_eva])
                    if num_eva>num:
                        nlarge.append([num_eva, bit_eva, accuracy_loss_eva])
        if monotonicity_accuracy_loss_upper_bound_set!=[]:
            accuracy_loss_upper_bound0=min(monotonicity_accuracy_loss_upper_bound_set)
            if len(nlarge)*len(nsmall)!=0:
                nlarge_point = find_closet(nlarge, num, bit)
                nsmall_point = find_closet(nsmall, num, bit)
                accuracy_loss_upper_bound1=(nlarge_point[2] - nsmall_point[2]) / (
                        nlarge_point[0] - nsmall_point[0]) * (num - nsmall_point[0]) + nsmall_point[2]
            if len(blarge) * len(bsmall) != 0:
                blarge_point = find_closet(blarge, num, bit)
                bsmall_point = find_closet(bsmall, num, bit)
                accuracy_loss_upper_bound2 = (blarge_point[2] - bsmall_point[2]) / (
                            blarge_point[1] - bsmall_point[1]) * (bit - bsmall_point[1]) + bsmall_point[2]
        else:
            continue
        actual_accuracy_loss_upper_bound=min(accuracy_loss_upper_bound0,accuracy_loss_upper_bound1,accuracy_loss_upper_bound2)
        if actual_accuracy_loss_upper_bound<=accuracy_loss_constraint:
            filter_scheme.append([id,num,bit])
    print('filtered scheme', filter_scheme)
    return filter_scheme

def space_shrink():
    global avaliable_scheme
    def find_closet(set, num, bit):
        distance = 10000
        closet = []
        for i, j, k in set:
            distance_tmp = abs(i + j - num - bit)
            if distance_tmp < distance:
                distance = distance_tmp
                closet = [i, j, k]
        return closet
    #avaliable_scheme.copy() avoids the modification of avaliable_scheme_tmp, when use avaliable_scheme.remove()
    avaliable_scheme_tmp=avaliable_scheme.copy()
    avaliable_scheme_tmp2=[]
    for [id,num,bit] in avaliable_scheme_tmp:
        nlarge = []
        nsmall = []
        blarge = []
        bsmall = []
        accuracy_loss_lower_bound0=0
        accuracy_loss_lower_bound1=0
        accuracy_loss_lower_bound2=0
        accuracy_loss_lower_bound3=0
        accuracy_loss_lower_bound4=0
        monotonicity_accuracy_loss_lower_bound_set=[]
        for [id_eva,num_eva,bit_eva],accuracy_loss_eva in avaliable_evaluated_scheme:
            if id==id_eva:
                if (num_eva>=num and bit_eva>=bit):
                    monotonicity_accuracy_loss_lower_bound_set.append(accuracy_loss_eva)
                if num_eva==num:
                    if bit_eva<bit:
                        bsmall.append([num_eva, bit_eva, accuracy_loss_eva])
                    if bit_eva>bit:
                        blarge.append([num_eva, bit_eva, accuracy_loss_eva])
                if bit_eva==bit:
                    if num_eva<num:
                        nsmall.append([num_eva, bit_eva, accuracy_loss_eva])
                    if num_eva>num:
                        nlarge.append([num_eva, bit_eva, accuracy_loss_eva])
        if monotonicity_accuracy_loss_lower_bound_set!=[]:
            accuracy_loss_lower_bound0=max(monotonicity_accuracy_loss_lower_bound_set)
        if len(nlarge)>=2:
            nlarge_point1 = find_closet(nlarge, num, bit)
            nlarge.remove(nlarge_point1)
            nlarge_point2= find_closet(nlarge, num, bit)
            accuracy_loss_upper_lower1=(nlarge_point1[2] - nlarge_point2[2]) / (
                    nlarge_point1[0] - nlarge_point2[0]) * (num - nlarge_point2[0]) + nlarge_point2[2]
        if len(nsmall) >= 2:
            nsmall_point1 = find_closet(nsmall, num, bit)
            nsmall.remove(nsmall_point1)
            nsmall_point2= find_closet(nsmall, num, bit)
            accuracy_loss_lower_bound2=(nsmall_point1[2] - nsmall_point2[2]) / (
                    nsmall_point1[0] - nsmall_point2[0]) * (num - nsmall_point2[0]) + nsmall_point2[2]
        if len(blarge)>=2:
            blarge_point1 = find_closet(blarge, num, bit)
            blarge.remove(blarge_point1)
            blarge_point2= find_closet(blarge, num, bit)
            accuracy_loss_lower_bound3=(blarge_point1[2] - blarge_point2[2]) / (
                    blarge_point1[1] - blarge_point2[1]) * (bit - blarge_point2[1]) + blarge_point2[2]
        if len(bsmall) >= 2:
            bsmall_point1 = find_closet(bsmall, num, bit)
            bsmall.remove(bsmall_point1)
            bsmall_point2= find_closet(bsmall, num, bit)
            accuracy_loss_lower_bound4=(bsmall_point1[2] - bsmall_point2[2]) / (
                    bsmall_point1[1] - bsmall_point2[1]) * (bit - bsmall_point2[1]) + bsmall_point2[2]
        if max(accuracy_loss_lower_bound0,accuracy_loss_lower_bound1,accuracy_loss_lower_bound2,accuracy_loss_lower_bound3,accuracy_loss_lower_bound4)>=accuracy_loss_constraint:
            avaliable_scheme.remove([id, num, bit])
            avaliable_scheme_tmp2.append([id, num, bit])
    print('Space_shrink:')
    print(' accuracy_loss_constraint',accuracy_loss_constraint)
    print(' latency_constraint',latency_constraint)
    print(' removed scheme numer',len(avaliable_scheme_tmp2))
    print(' avaliable scheme numer',len(avaliable_scheme))

def evaluate_update_scheme(select_scheme):
    global accuracy_loss_constraint,latency_constraint,optimal_scheme, avaliable_evaluated_scheme,search_times,avaliable_scheme
    print('Evaluate_Update_scheme',select_scheme)
    ##evaluate the select_scheme
    accuracy_loss_tmp=find_in_database(select_scheme)
    if not accuracy_loss_tmp:
        os.system('python evaluate_update.py -z_partition_id {:.0f} -z_pruning_num {:.0f} -z_quantization {:.0f} -gpu {:.0f}'.format(select_scheme[0], select_scheme[1], select_scheme[2], gpu))
        accuracy_loss_tmp=find_in_database(select_scheme)
    search_times+=1
    ##remove the select_scheme from avaliable_scheme, and store it into database avaliable_evaluated_scheme
    avaliable_scheme.remove(select_scheme)
    avaliable_evaluated_scheme.append([select_scheme, accuracy_loss_tmp]) #add evaluated scheme to database
    if accuracy_loss_tmp-accuracy_loss_constraint<=1e-5:
        optimal_scheme=select_scheme
        if accuracy_or_latency_demand:
            latency_constraint=all_latency[all_scheme.index(select_scheme)]
        else:
            accuracy_loss_constraint=accuracy_loss_tmp
    else:
        print('The accuracy_loss_constraint and evaluated accuracy_loss is {:.5f} and {:.5f}'.format(accuracy_loss_constraint,accuracy_loss_tmp))


def index_conut_latency():
    index=[]
    for i in range(len(all_latency)):
        if all_latency[i]<latency_constraint:
            index.append(i)
    return index

def find_in_database(select_scheme):
	accuracy_loss_tmp=None
	with open(data_base_address, 'r') as f:
		data = f.readlines()
		for i in data:
			x = i.find('+')
			y = i.find('-')
			z = i.find('*')
			partition_id_tmp=int(i[:x])
			num_tmp=int(i[x:y])
			bit_tmp = int(i[y+1:z])
			if partition_id_tmp==select_scheme[0] and num_tmp==select_scheme[1] and bit_tmp==select_scheme[2]:
				accuracy_loss_tmp = float(i[z + 1:])-accuracy_loss_base
	return accuracy_loss_tmp


def get_same_element(set1,set2):
    tmp=[]
    tmp=[i for i in set1 if i in set2]
    return tmp

def main():
    global accuracy_loss_constraint, latency_constraint, optimal_scheme, avaliable_scheme, avaliable_evaluated_scheme,search_times
    # filter schemes that are meant to satisfy the accuracy constrain, according to their accuracy lower bound
    filter_scheme_set = filter()
    while len(filter_scheme_set) != 0:
        filter_scheme = random.sample(filter_scheme_set, 1)[0]  # randomly select a qualified scheme
        evaluate_update_scheme(filter_scheme)  # evaluate the scheme, and store the data in avaliable_evaluated_scheme，and update latency&accuracy constraints
        filter_scheme_set = filter()
        avaliable_index = index_conut_latency()
        avaliable_scheme_tmp = [all_scheme[i] for i in avaliable_index]
        avaliable_scheme = get_same_element(avaliable_scheme, avaliable_scheme_tmp)
    avaliable_index = index_conut_latency()
    avaliable_scheme_tmp = [all_scheme[i] for i in avaliable_index]
    avaliable_scheme = get_same_element(avaliable_scheme, avaliable_scheme_tmp)
    # remove the schemes that are meant to not satisfy the accuracy_constraint
    space_shrink()
    while len(avaliable_scheme)>0:
        #bulid the Guassian Process model based on the avaliable_evaluated_scheme
        GP=gaussin_process(avaliable_evaluated_scheme)
        #find the promising scheme according to the cquisition function
        promising_scheme=opt_acquisition(avaliable_scheme,GP)
        #evaluate and update
        evaluate_update_scheme(promising_scheme)
        avaliable_index=index_conut_latency()#filter scheme according to latency constraint
        avaliable_scheme_tmp=[all_scheme[i] for i in avaliable_index]
        avaliable_scheme=get_same_element(avaliable_scheme,avaliable_scheme_tmp)
        # filter schemes that are meant to satisfy the accuracy constrain， according to their accuracy lower bound
        filter_scheme_set = filter()
        while len(filter_scheme_set) != 0:
            filter_scheme = random.sample(filter_scheme_set, 1)[0]
            evaluate_update_scheme(filter_scheme)
            filter_scheme_set = filter()
            avaliable_index=index_conut_latency()
            avaliable_scheme_tmp=[all_scheme[i] for i in avaliable_index]
            avaliable_scheme=get_same_element(avaliable_scheme,avaliable_scheme_tmp)
        avaliable_index = index_conut_latency()
        avaliable_scheme_tmp = [all_scheme[i] for i in avaliable_index]
        avaliable_scheme=get_same_element(avaliable_scheme,avaliable_scheme_tmp)
        space_shrink()
        print('One Iteration is Finished:\n','evaluation budget', search_times,'\n optimal scheme',optimal_scheme,'\n accuracy_loss_constraint',accuracy_loss_constraint,'\n latency_constraint',latency_constraint)
        print('\n')
    print('The final result:')
    print('\n evaluation budget',search_times,'\n optimal_scheme',optimal_scheme,'\n accuracy_loss_constraint',accuracy_loss_constraint,'\n latency_constraint',latency_constraint)


if __name__ == "__main__":

    accuracy_or_latency_demand=args.accuracy_or_latency #True for accuracy demand
    if accuracy_or_latency_demand:
        accuracy_loss_constraint = args.accuracy_loss
        latency_constraint = min(only_cloud,only_pi_latency)
    else:
        accuracy_loss_constraint = 0.1
        latency_constraint = min(only_cloud,only_pi_latency)*args.latency #True for accuracy demand
    gpu = args.gpu

##initialize the scheme space
    search_times=0
    optimal_scheme = []
    all_scheme, all_transmission, all_latency, all_layer_start_index = scheme_generation(pi_layer_latency)
    all_evaluated_scheme = [[i, None] for i in all_scheme]#evaluated_scheme=[[scheme],accuracy_loss]
    avaliable_scheme = all_scheme
    avaliable_evaluated_scheme = [i for i in all_evaluated_scheme if i[1] != None]#avaliable_evaluated_scheme=[[scheme],accuracy_loss]
    print('The whole searching space =',len(avaliable_scheme))
    # filter schemes satisfying the latency constraint
    avaliable_index = index_conut_latency()
    avaliable_scheme = [all_scheme[i] for i in avaliable_index]
    for i in avaliable_evaluated_scheme:
        if i[0] in avaliable_scheme:
            avaliable_scheme.remove(i[0])
##start searching
    main()
