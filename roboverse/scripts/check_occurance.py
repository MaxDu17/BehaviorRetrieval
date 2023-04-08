import os
import os.path as osp
import argparse
import re

def main(args):
    object_names = {'eraser', 'shed', 'pepsi_bottle', 'gatorade'}
    dir = args.save_directory
    filenames = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".txt"): filenames.append(os.path.join(root, file))
    
    total_area_occurance = [0, 0, 0]
    total_object_occurance = {}
    for object_name in object_names:
        total_object_occurance[object_name] = 0
    
    for filename in filenames:
        for line in open(filename, "r"):
            # print(line)
            # print("next")
            str = line.strip()
            num = list(map(int, re.findall('\d+', str)))
            if len(num) == 3:
                for i in range(len(num)):
                    total_area_occurance[i] += num[i]
            else:
                for i, object_name in enumerate(object_names):
                    total_object_occurance[object_name] += num[i]
    
    # print(filenames)
    area = total_area_occurance 
    total_area = sum(area)
    obj = total_object_occurance
    total_obj = 0
    obj_names = []
    obj_num = []
    for ob in obj:
        obj_names.append(ob)
        obj_num.append(obj[ob])
        total_obj += obj[ob]
    print(f"total area: {total_area}, upper left: {area[0]/total_area:.3f}, upper middle: {area[1]/total_area:.3f}, lower right: {area[2]/total_area:.3f}")
    
    print(f"total obj: {total_obj}, {obj_names[0]}: {obj_num[0]/total_obj:.3f}, {obj_names[1]}: {obj_num[1]/total_obj:.3f}, {obj_names[2]}: {obj_num[2]/total_obj:.3f}, {obj_names[3]}: {obj_num[3]/total_obj:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--save-directory", type=str, default="./data/testfolder"),
    args = parser.parse_args()

    main(args)