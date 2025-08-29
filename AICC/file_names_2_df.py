import numpy as np
import pandas as pd
import glob

def file_names_2_df(path, split_by='/'):
    ndpi_files = sorted(glob.glob(path + '/*'), reverse=True)

    name_annotations = []
    for ndpi_file in ndpi_files:
        name_annotations.append(ndpi_file.replace('.ndpi','').split(split_by)[-1].split(')_'))

    name_annot_dict = {}
    for i, name_annotation in enumerate(name_annotations):
        if len(name_annotation)== 2 :
            combined_annot = name_annotation[-1]
            if combined_annot in ['1b','1b_JS_1b', ]: final_label = 1
            elif combined_annot in ['1a','1a_1b','1a+1b', '1a_JS_1a']: final_label = 2
            elif combined_annot in ['2','1b_2','2_JS_2'] : final_label = 3
            elif combined_annot in ['3', '2_3','3_JS_3']  : final_label = 4
            elif combined_annot in ['0','0_JS_0']: final_label = 0

            name_annot_dict[name_annotation[0].strip().replace('(', '-')] = [ndpi_files[i], ndpi_files[i].split('\\')[-1], combined_annot, final_label]
        else:
            name_annot_dict[name_annotation[0]] = [ndpi_files[i], ndpi_files[i].split('\\')[-1], np.nan, np.nan]

    annotated= pd.DataFrame.from_dict(name_annot_dict, orient='index')
    annotated.reset_index(inplace=True)

    annotated.columns = ['name', 'label']
    
    return annotated