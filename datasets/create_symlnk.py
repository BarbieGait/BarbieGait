import argparse
import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-root',
    default=os.path.join(project_root, 'BarbieGait_data'),
    help='Directory containing the BarbieGait predicted-modality PKL directories.',
)
parser.add_argument(
    '--modality',
    choices=('sil', 'pose', 'heatmap'),
    default='sil',
    help='Predicted modality to process. Defaults to sil.',
)
parser.add_argument(
    '--label-root',
    default=os.path.join(script_dir, 'BarbieGait', 'thick_label_by_nakeddiffnorm_eqchg'),
    help='Directory containing per-identity clothing thickness labels.',
)
args = parser.parse_args()

data_root = os.path.abspath(args.data_root)
label_root = os.path.abspath(args.label_root)
pred_root = os.path.join(data_root, f'BarbieGait_pred{args.modality}_pkl')
link_path = os.path.join(pred_root, 'P2_pkl')

def custom_sort(folder_name):
    # Split folder name by "cloth index" and "suffix number"
    parts = folder_name.split('-')

    cloth_number = int(parts[0].replace('cloth', ''))

    suffix_number = int(parts[1])

    return (cloth_number, suffix_number)

data_path = os.path.join(pred_root, f'{args.modality}_pkl')
personlist = os.listdir(data_path)
lacklist = []
created_count = 0
existing_count = 0
for personid in personlist:
    idpath = os.path.join(data_path, personid)
    personid_int = str(int(personid))
    thick_label_path = os.path.join(label_root, personid, '{}_thick_data.json'.format(personid_int))
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
            created_count += 1
        except FileExistsError:
            existing_count += 1
        except OSError as e:
            print(f'Failed to create symbolic link: {e}')

print(f'Created links: {created_count}')
print(f'Existing links: {existing_count}')
print(f'IDs without thickness labels: {len(lacklist)}')
