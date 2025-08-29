import numpy as np
import pandas as pd
import glob

def file_names_2_df(path, split_by):
    ndpi_files = sorted(glob.glob(path + '/*'), reverse=True)
    name_annotations = []
    for ndpi_file in ndpi_files:
        name_annotations.append(ndpi_file.replace('.ndpi','').split(split_by)[-1].split(')_'))

    name_annot_dict = {}
    for i, name_annotation in enumerate(name_annotations):
        if len(name_annotation)== 2 :
            combined_annot = name_annotation[-1]
            if combined_annot == '1a': final_label = 1
            elif combined_annot == '1b' or combined_annot=='1a_1b': final_label = 2
            elif combined_annot == '2'  or combined_annot=='1b_2': final_label = 3
            elif combined_annot == '3'  or combined_annot=='2_3': final_label = 4
            else: final_label = combined_annot

            name_annot_dict[name_annotation[0].replace(' (', '-')] = [ndpi_files[i], combined_annot, final_label]
        else:
            name_annot_dict[name_annotation[0]] = [ndpi_files[i], np.nan, np.nan]

    annotated= pd.DataFrame.from_dict(name_annot_dict, orient='index')
    annotated.reset_index(inplace=True)

    annotated.columns = ['bcc code', 'address',' temp label','label']
    
    return annotated