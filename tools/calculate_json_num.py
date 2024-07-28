import json
from collections import Counter


# # with open('/home/yb/code/Panoptic_Perception/data/mar20/annotations/panoptic_mar20_final_val.json', 'r') as f:
# with open('/home/yb/code/Panoptic_Perception/panoptic_mar20_final_val_revised.json', 'r') as f:
#     data = json.load(f)

# print(len(data['images']))

# # 
# file_names = []
# for item in data['images']:
#     file_names.append(item['file_name'])

# ## print(file_names)
# # counter = Counter(file_names)
# # print(len(counter))
# # print(counter)

# duplicates = set([x for x in file_names if file_names.count(x) > 1])
# print(len(duplicates))
# print(duplicates)

# ## get repeated samples
# # string_set = [str(item) for item in duplicates]
# # with open('duplicated_samples_in_val.txt', 'w') as file:
# #     file.write('\n'.join(string_set))

## ==============delete the repeated samples in final_val.json==================
with open('/home/yb/code/Panoptic_Perception/panoptic_mar20_final_val.json', 'r') as f:
    data = json.load(f)

print('keys:',data.keys())
unique = []

# for item in data['images']: 
for item in data['annotations']:
# for item in data['categories']:
    flag=0
    for check in unique:
        if check['file_name']==item['file_name']:
            flag=1
            break
    if flag==0:
        unique.append(item)

print(len(unique))
with open("panoptic_mar20_final_val_revised.json","w") as f:
    json.dump(unique,f)




