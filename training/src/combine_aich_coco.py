import json
import os

aich_json_train = r'../dataset/ai_challenger/ai_challenger_train.json'
aich_json_valid = r'../dataset/ai_challenger/ai_challenger_valid.json'
coco_json = r'../dataset/single_person_from_coco/coco_single_short_name.json'

aich_anns_train = json.load(open(aich_json_train))
aich_anns_valid = json.load(open(aich_json_valid))
coco_anns = json.load(open(coco_json))

print(aich_anns_train.keys())
print(coco_anns.keys())



# modify aich valid file name
for i, img_ann in enumerate(aich_anns_valid['images']):
    aich_anns_valid['images'][i]['file_name'] = os.path.join('aich_plus_coco',
                                                             'valid',
                                                              img_ann['file_name'].split('/')[-1])
with open('../dataset/aich_plus_coco/ai_challenger_valid.json', 'w') as f:
    json.dump(aich_anns_valid, f)


# modify aich train file name
for i, img_ann in enumerate(aich_anns_train['images']):
    aich_anns_train['images'][i]['file_name'] = os.path.join('aich_plus_coco',
                                                             'train',
                                                              img_ann['file_name'].split('/')[-1])
with open('../dataset/aich_plus_coco/ai_challenger_train.json', 'w') as f:
    json.dump(aich_anns_train, f)





# gen long name coco json
for i, img_ann in enumerate(coco_anns['images']):
    coco_anns['images'][i]['file_name'] = os.path.join('aich_plus_coco',
                                                        'coco_single',
                                                        img_ann['file_name'])
with open('../dataset/aich_plus_coco/coco_train.json', 'w') as f:
    json.dump(coco_anns, f)

# combine
aich_anns_train['images'] += coco_anns['images']
aich_anns_train['annotations'] += coco_anns['annotations']

with open('../dataset/aich_plus_coco/ai_challenger_coco_train.json', 'w') as f:
    json.dump(aich_anns_train, f)



