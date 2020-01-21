# -*- coding: utf-8 -*-
# @Time    : 18-3-6 3:20 PM
# @Author  : zengzihua@huya.com
# @FileName: data_filter.py
# @Software: PyCharm

import network_mv2_cpm
import network_mv2_hourglass
#import network_shufflenetv2_cpm
#import network_mv2_hg_cpm
#import shufflenetv2_cpm
#import network_lednet_cpm
#import shufflenetv2_cpm_new
#import network_zq11_cpm
import network_zq21_cpm
import network_zq21_cpm_backbone_2x2
from mobilenet import network_yang_mv3_cpm
from mobilenet import network_yang_mv3_cpm_4x3x3
from mobilenet import network_yang_mv3_2x2_cpm_4x3x3
from mobilenet import network_yang_mv3_2x2_cpm_2x7x7

from mobilenet import network_yang_mv3_3_2x2_cpm_4x3x3
from mobilenet import network_yang_mv3_3_2x2_cpm
from mobilenet import network_yang_mv3_cpm_8x3x3



#import espnetv2_cpm
def get_network(type, input, trainable=True):
    if type == 'mv2_cpm':
        net, loss = network_mv2_cpm.build_network(input, trainable)
        # 这里的loss实际上不是loss？而是pred？
    #elif type == "mv2_hourglass":
        #net, loss = network_mv2_hourglass.build_network(input, trainable)
    #elif type=="shufflenet_v2":
        #net,loss=network_shufflenetv2_cpm.build_network(input,trainable)
    #elif type=="mv2_hg_cpm":
        #net,loss = network_mv2_hg_cpm.build_network(input, trainable)   
    #elif type=="shufflenetv2_cpm":
        #net, loss =shufflenetv2_cpm_new.build_network(input, trainable)
        #print('net',net)
    #elif type=="lednet_cpm":
        #net, loss = network_lednet_cpm.build_network(input, trainable)
    #elif type=="zq11_cpm":
        #net, loss = network_zq11_cpm.build_network(input, trainable)
    elif type=="zq21_cpm":
        net, loss = network_zq21_cpm.build_network(input, trainable)
    elif type=="zq21_cpm_backbone_2x2":
        net, loss = network_zq21_cpm_backbone_2x2.build_network(input, trainable)
    elif type=="yang_mv3_cpm":
        net, loss = network_yang_mv3_cpm.build_network(input, trainable)
    elif type=="yang_mv3_cpm_4x3x3":
        net, loss = network_yang_mv3_cpm_4x3x3.build_network(input, trainable)
    elif type=="yang_mv3_2x2_cpm_4x3x3":
        net, loss = network_yang_mv3_2x2_cpm_4x3x3.build_network(input, trainable)
    elif type=="yang_mv3_2x2_cpm_2x7x7":
        net, loss = network_yang_mv3_2x2_cpm_2x7x7.build_network(input, trainable)
    elif type=="yang_mv3_3_2x2_cpm_4x3x3":
        net, loss = network_yang_mv3_3_2x2_cpm_4x3x3.build_network(input, trainable)
    elif type=="yang_mv3_3_2x2_cpm":
        net, loss = network_yang_mv3_3_2x2_cpm.build_network(input, trainable)
    elif type=="yang_mv3_cpm_8x3x3":
        net, loss = network_yang_mv3_cpm_8x3x3.build_network(input, trainable)
    #elif type=="espnetv2_cpm":
        #net, loss = espnetv2_cpm.build_network(input, trainable)    
    return net, loss
