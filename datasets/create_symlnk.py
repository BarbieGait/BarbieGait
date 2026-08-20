import os
import json
import pdb

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

# Symbolic link output path
link_path = os.path.join(project_root, 'BarbieGait_data', 'P2_BarbieGait_predsil_pkl')

def custom_sort(folder_name):
    # Split folder name by "cloth index" and "suffix number"
    parts = folder_name.split('-')

    cloth_number = int(parts[0].replace('cloth', ''))

    suffix_number = int(parts[1])

    return (cloth_number, suffix_number)

data_path = os.path.join(project_root, 'BarbieGait_data', 'BarbieGait_predsil_pkl')
thick_path = os.path.join(project_root, 'BarbieGait_data', 'thick_label_by_nakeddiffnorm_eqchg')
personlist = os.listdir(data_path)
seqnum = 0
lacklist = []
for personid in personlist:
    idpath = os.path.join(data_path, personid)
    personid_int = str(int(personid))
    thick_label_path = os.path.join(thick_path, personid_int, '{}_thick_data.json'.format(personid_int))
    if not os.path.exists(thick_label_path):
        lacklist.append(personid)
        continue
    with open(thick_label_path, 'r') as json_file:
        thick_label = json.load(json_file)
    thick = thick_label[personid_int]
    thick_seq_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

    for seq in sorted(os.listdir(idpath), key=custom_sort):
        clothname = seq.split('-')[0]
        seqname = seq.split('-')[1]

        if int(clothname[5:]) > 99:
            continue
        thick_value = thick['cloth' + str(int(clothname[5:]))]

        thick_seq_name = 'thick{:01d}-{:02d}'.format(int(thick_value), thick_seq_dict[thick_value])
        thick_seq_dict[thick_value] = thick_seq_dict[thick_value] + 1

        newdir_path = os.path.join(link_path, personid, thick_seq_name + '-' + seq)
        olddir_path = os.path.join(data_path, personid, seq)
        newdir_parent_dir = os.path.dirname(newdir_path)
        if not os.path.exists(newdir_parent_dir):
            os.makedirs(newdir_parent_dir)
        try:
            os.symlink(olddir_path, newdir_path)
            print(f'Symbolic link created: {olddir_path} -> {newdir_path}')
        except OSError as e:
            print(f'Failed to create symbolic link: {e}')

print(lacklist)
