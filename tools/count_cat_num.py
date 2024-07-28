import json
import matplotlib.pyplot as plt
from typing import Dict

def count_cat_um_from_json(json_file_path,cat_id_to_num_dict:Dict):
    f = open(json_file_path,"r")
    panoptic_data = json.load(f)
    annotations = panoptic_data["annotations"]
    categories = panoptic_data["categories"]

    for annot in annotations:
        for segment_info in annot['segments_info']:
            cat_id = segment_info["category_id"]
            if cat_id not in cat_id_to_num_dict:
                cat_id_to_num_dict[cat_id] = 1
            else:
                cat_id_to_num_dict[cat_id] += 1

def cat_num_barplot(cat_id_to_num_dict_A,cat_id_to_num_dict_B,cat_id_to_name_dict,tagA,tagB):
    all_cat_ids = sorted(set(cat_id_to_num_dict_A.keys()) | set(cat_id_to_num_dict_B.keys()))
    
    labels = [cat_id_to_name_dict[cat_id] for cat_id in all_cat_ids]
    a_values = [cat_id_to_num_dict_A.get(cat_id, 0) for cat_id in all_cat_ids] # train
    b_values = [cat_id_to_num_dict_B.get(cat_id, 0) for cat_id in all_cat_ids] # validation
    total_values = list(map(lambda x, y: x + y, a_values, b_values))   # total
    total_values_temp = [0 for _ in range(25)]

    x = range(len(labels))
    plt.subplots(figsize=(16,7))
    bars = plt.bar(x, total_values,  color='white', alpha=0)
    for bar in bars:
        y_val = bar.get_height()
        plt.text(bar.get_x()+ bar.get_width()/2,y_val,y_val, ha='center', va='bottom')

    plt.bar(x, a_values, label=tagA, color='#8AB1D2', alpha=1.0)
    plt.bar(x, b_values, bottom=a_values, label=tagB, color='#ED9F9B', alpha=1.0)
    

    plt.xticks(x, labels, rotation=30)

    for i in x:
        plt.text(i, a_values[i] / 2, str(a_values[i]), ha='center', va='bottom', color='black')
        plt.text(i, a_values[i] + b_values[i] / 2, str(b_values[i]), ha='center', va='bottom', color='black')

    plt.legend()
    # plt.title('Category-wise Count Comparison')

    plt.tight_layout()
    plt.legend(loc=0, fontsize=14)
    plt.savefig( 'dataset_info.pdf',dpi=600,format='pdf')
    plt.show()

if __name__ == "__main__":
    
    id2category_dict =   {
                            0: "SU-35",
                            1: "C-130",
                            2: "C-17",
                            3: "C-5",
                            4: "F-16",
                            5: "TU-160",
                            6: "E-3",
                            7: "B-52",
                            8: "P-3C",
                            9: "B-1B",
                            10: "E-8",
                            11: "TU-22",
                            12: "F-15",
                            13: "KC-135",
                            14: "F-22",
                            15: "FA-18",
                            16: "TU-95",
                            17: "KC-10",
                            18: "SU-34",
                            19: "SU-24",
                            20: "Land",
                            21: "Runway",
                            22: "Hardstand",
                            23: "Parking-apron",
                            24: "Building",
                        }

 

    train_cat2num_dict,test_cat2num_dict = {},{}

    panoptic_train_json_path = "D://YuanBo//Dataset//FineGrip//panoptic_mar20_final_train.json"
    panoptic_test_json_path = "D://YuanBo//Dataset//FineGrip//panoptic_mar20_final_val.json"

    count_cat_um_from_json(panoptic_train_json_path,train_cat2num_dict)
    count_cat_um_from_json(panoptic_test_json_path,test_cat2num_dict)

    print(train_cat2num_dict)
    print(test_cat2num_dict)

    assert len(train_cat2num_dict.keys()) == len(test_cat2num_dict.keys())

    # cat_id_to_name_dict = {i:str(i) for i in range(25)}
    cat_id_to_name_dict = id2category_dict
    print('cat_id_to_name_dict:',cat_id_to_name_dict)

    cat_num_barplot(train_cat2num_dict,test_cat2num_dict,cat_id_to_name_dict,"Training set","Validation set")

