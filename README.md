# Opt-CoInfer
Opt-CoInfer is a systematic framework to provide promising scheme and achieve the optimal CCI for any given industrial scenarios (i.e. the combination of CNN model, IoT-Cloud system and inference latency/accuracy requirement).  

## Overview
For efficient CNN inference in various industrial scenarios with resource limitations, Collaborative CNN Inference (CCI) based on partition and compression techniques offloads most of the computed and compressed transmission data from IoT devices to Cloud, and achieves efficient CNN inference through a carefully selected partition and compression schemes.   

Therefore, we propose Opt-CoInfer, a systematic framework to provide promising scheme and achieve the optimal CCI for any given industrial scenarios (i.e. the combination of CNN model, IoT-Cloud system and inference latency/accuracy requirement).

For a given CNN model, IoT-Cloud system and inference ***latency*** requirement, Opt-CoInfer provides the ***most accurate*** scheme satisfying the ***latency*** requirement.   

For a given CNN model, IoT-Cloud system and inference ***accuracy*** requirement, Opt-CoInfer provides the ***fastest*** scheme satisfying the ***accuracy*** requirement.
## Installation
Clone repo and install requirements.txt in a Python>=3.7.0 environment.   

```
git clone https://github.com/Anonymity2022/Opt-CoInfer.git  # clone
cd Opt-CoInfer
pip install -r requirements.txt  # install
```

## Example
Here, an example of VGG-16 on Stanford-Cars dataset with a practical IoT-Cloud system (illustrated in the following figure) is provided for evaluation.

![img1](./assets/img/img1.png)


To avoid time-consuming evaluation of various schemes, the database of evaluated scheme (i.e., [accuracy_search.txt](https://github.com/Anonymity2022/Opt-CoInfer/blob/main/accuracy_search.txt), [latency_search.txt](https://github.com/Anonymity2022/Opt-CoInfer/blob/main/latency_search.txt)) is provided.  

The evaluation requires several settings:  
1.	Click the link to download the original[ VGG-16 model](https://drive.google.com/file/d/1R5IsvLMvbWZ5zehyWLxvx-jzAspiFjcW/view?usp=sharing). Then place this model at `Opt-CoInfer/0+64+8+0.151/`.
2.	Modify the items (e.g., dataset_dir, data_base_address) at file `Opt-CoInfer/apps/vgg0.yml` to identify the location of `dataset/database` and others settings of CNN training.
3.	For accuracy requirement, run
```
python accuracy_search.py -latency x -bandwidth x -gpu x.  
```
For latency requirement, run 
```
python latency_search.py -accuracy_loss x -bandwidth x -gpu x.  
```  

The parameter `-latency` represents the ratio of the minimal latency derived from single-end approach (i.e., make whole CNN inference on IoT or Cloud), and can be set as **0.6/0.7**.  

The parameter `-accuracy` loss represents the accuracy loss compared with the original CNN model, and can be set as **0.01/0.05**.  

The parameter `-bandwidth` represents the bandwidth (Mbps) between IoT, and can be set as **0.5/1/2**.  

The parameter `-gpu` represents the id of GPU for the experiment, and can be set **according to the user**.


