#########################################################################
# File Name: run.sh
# Author: geeker
# mail: 932834897@qq.com
# Created Time: 2019年12月06日 星期五 13时53分52秒
#########################################################################
#!/bin/bash


CHK_PATH="/data/home/chenyu/Yangfeiyu/singlePose/log/zq_224/model/yang_mv3_cpm_batch-300_lr-0.001_gpus-2_224x224_..-experiments-zq21_cpm/model-21000"


/data/home/chenyu/Yangfeiyu/App/anaconda3/envs/tf13/bin/python\
	gen_frozen_pb.py\
	--checkpoint=${CHK_PATH}

/data/home/chenyu/Yangfeiyu/App/anaconda3/envs/tf13/bin/python\
	benchmark.py\
	--frozen_pb_CHK_PATH="${PATH}.pb"


