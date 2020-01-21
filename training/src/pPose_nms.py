# -*- coding: utf-8 -*-
import json
import os
import zipfile
import time
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np

''' Constant Configuration '''
delta1 = 1
mu = 1.7
delta2 = 2.65
gamma = 22.48
scoreThreds = 0.3
matchThreds = 5
areaThres = 0#40 * 40.5
alpha = 0.1

def write_json(all_results, outputpath, for_eval=False):
    '''
    all_result: result dict of predictions
    outputpath: output directory
    '''
    form = 'coco'
    json_results = []
    json_results_cmu = {}
    for im_res in all_results:
        im_name = im_res['imgname']
        for human in im_res['result']:
            keypoints = []
            keyscores = []
            result = {}
            if for_eval:
                result['image_id'] = int(im_name.split('/')[-1].split('.')[0].split('_')[-1])
            else:
                result['image_id'] = im_name.split('/')[-1]
            result['category_id'] = 1

            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            #pro_scores = human['proposal_score']
            for n in range(len(kp_scores)):
                keypoints.append([float(kp_preds[n][0]), float(kp_preds[n][1])])
                #keypoints.append(float(kp_preds[n, 1]))
                keyscores.append(float(kp_scores[n]))
            result['keypoints'] = keypoints
            result['kp_scores'] = keyscores
            #result['score'] = float(pro_scores)

            if form == 'cmu': # the form of CMU-Pose
                if result['image_id'] not in json_results_cmu.keys():
                    json_results_cmu[result['image_id']]={}
                    json_results_cmu[result['image_id']]['version']="AlphaPose v0.2"
                    json_results_cmu[result['image_id']]['bodies']=[]
                tmp={'joints':[]}
                result['keypoints'].append((result['keypoints'][15]+result['keypoints'][18])/2)
                result['keypoints'].append((result['keypoints'][16]+result['keypoints'][19])/2)
                result['keypoints'].append((result['keypoints'][17]+result['keypoints'][20])/2)
                indexarr=[0,51,18,24,30,15,21,27,36,42,48,33,39,45,6,3,12,9]
                for i in indexarr:
                    tmp['joints'].append(result['keypoints'][i])
                    tmp['joints'].append(result['keypoints'][i+1])
                    tmp['joints'].append(result['keypoints'][i+2])
                json_results_cmu[result['image_id']]['bodies'].append(tmp)
            elif form == 'open': # the form of OpenPose
                if result['image_id'] not in json_results_cmu.keys():
                    json_results_cmu[result['image_id']]={}
                    json_results_cmu[result['image_id']]['version']="AlphaPose v0.2"
                    json_results_cmu[result['image_id']]['people']=[]
                tmp={'pose_keypoints_2d':[]}
                result['keypoints'].append((result['keypoints'][15]+result['keypoints'][18])/2)
                result['keypoints'].append((result['keypoints'][16]+result['keypoints'][19])/2)
                result['keypoints'].append((result['keypoints'][17]+result['keypoints'][20])/2)
                indexarr=[0,51,18,24,30,15,21,27,36,42,48,33,39,45,6,3,12,9]
                for i in indexarr:
                    tmp['pose_keypoints_2d'].append(result['keypoints'][i])
                    tmp['pose_keypoints_2d'].append(result['keypoints'][i+1])
                    tmp['pose_keypoints_2d'].append(result['keypoints'][i+2])
                json_results_cmu[result['image_id']]['people'].append(tmp)
            else:
                json_results.append(result)

    if form == 'cmu': # the form of CMU-Pose
        with open(os.path.join(outputpath,'alphapose-results.json'), 'w') as json_file:
            json_file.write(json.dumps(json_results_cmu))
            if not os.path.exists(os.path.join(outputpath,'sep-json')):
                os.mkdir(os.path.join(outputpath,'sep-json'))
            for name in json_results_cmu.keys():
                with open(os.path.join(outputpath,'sep-json',name.split('.')[0]+'.json'),'w') as json_file:
                    json_file.write(json.dumps(json_results_cmu[name]))
    elif form == 'open': # the form of OpenPose
        with open(os.path.join(outputpath,'alphapose-results.json'), 'w') as json_file:
            json_file.write(json.dumps(json_results_cmu))
            if not os.path.exists(os.path.join(outputpath,'sep-json')):
                os.mkdir(os.path.join(outputpath,'sep-json'))
            for name in json_results_cmu.keys():
                with open(os.path.join(outputpath,'sep-json',name.split('.')[0]+'.json'),'w') as json_file:
                    json_file.write(json.dumps(json_results_cmu[name]))
    else:
        with open(os.path.join(outputpath,'alphapose-results.json'), 'w') as json_file:
            json_file.write(json.dumps(json_results))

