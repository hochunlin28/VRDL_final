import glob, pylab, pandas as pd
import pydicom, numpy as np
import cv2
import os
from os import listdir
import json
import shutil
from tqdm import tqdm
import shapely.geometry

def bbox(long0, lat0, lat1, long1):
  return Polygon([[long0, lat0],[long1,lat0],[long1,lat1],[long0, lat1]])
  
def convert(o):
    if isinstance(o, np.generic): return o.item()  
    raise TypeError

def get_image_coco_fmt(id, width, height, file_name):
    image = {
        "id": id,
        "width": width,
        "height": height,
        "file_name": file_name,
    }
    return image

def get_category_coco_fmt():
    categories = {
        "id": 0,
        "name": "pneumonia",
        "supercategory": "pneumonia",
    }
    return categories

def get_annotation_coco_fmt(mask_id, image_id, area, bbox, poly):
    annotation = {
        "id": mask_id,
        "image_id": image_id,
        "category_id": 0,
        "area": area,
        "segmentation": poly,
        "bbox": bbox,
        "iscrowd": 0,
    }
    return annotation
    
def combine_coco(images_coco, categories_coco, annotations_coco):
    result = {
        "images": images_coco,
        "categories": categories_coco,
        "annotations": annotations_coco,
    }
    return result
    
def split_train_val():
    src = './train/'
    dst = './val/'
    split_val_image_count = 4
    
    if not os.path.exists(os.path.join('./', 'val')):
        os.makedirs(os.path.join('./', 'val'))

    df = pd.read_csv('./val/val.csv')
    train_filenames = listdir(src)
    
    for n, row in df.iterrows():
        pid = row['patientId']
        if os.path.exists(src + pid + '.png'):
            shutil.move(src + pid + '.png', dst)
            
    '''
    for img_id, file_name in tqdm(enumerate(train_filenames), total=len(train_filenames)):
        # Split 3 data from train to val
        if img_id < split_val_image_count:
            print(f'Move {src + file_name} to {dst}')
            shutil.move(src + file_name, dst)
    '''

def parse_data(df):
    """
    Method to read a CSV file (Pandas dataframe) and parse the 
    data into the following nested dictionary:

      parsed = {
        
        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        }, ...

      }

    """
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['x'], row['y'], row['width'], row['height']]

    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed 
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': 'stage_2_train_images/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed
    
def produce_trainval_coco_json(train_path = './train/', val=False):
    train_filenames = listdir(train_path)
    images_coco, categories_coco, annotations_coco = [], [], []
    mask_counts = 0;
    
    category = get_category_coco_fmt()
    categories_coco.append(category)
    
    for img_id, file_name in tqdm(enumerate(train_filenames), total=len(train_filenames)):
        file_name = file_name[:-4]
        patient = parsed[file_name]
        
        #get image coco format
        pic_path = train_path + file_name + '.png'
        img = cv2.imread(pic_path)[...,::-1]
        img_h, img_w, _ = img.shape
        
        image = get_image_coco_fmt(img_id, img_w, img_h, file_name + '.png')
        images_coco.append(image)
        #print(patient['boxes'])
        
        for box in patient['boxes']:
            area = box[2] * box[3]
            
            polygon=shapely.geometry.box(box[0],box[1],box[0]+box[2], box[1]+box[3], ccw=True)
            K=str(polygon.wkt).split("POLYGON ((")[-1].split("))")[0].split(',')
            poly=[]
            for m in K:
                for p in m.split(" "):
                    if p:
                        poly.append(int(p))

            poly=[poly]
            annotation = get_annotation_coco_fmt(mask_counts, img_id, area, box, poly)
            annotations_coco.append(annotation)
            
            mask_counts  = mask_counts + 1
    
    result = combine_coco(images_coco, categories_coco, annotations_coco)

    json_filename = './val_coco.json' if val else './train_coco.json'
    result_coco = json.dumps(result, indent=4, sort_keys=False, default=convert)
    with open(json_filename, 'w') as fp:
        fp.write(result_coco)

# prepare dataset
df_detailed = pd.read_csv('stage_2_detailed_class_info.csv')
df = pd.read_csv('stage_2_train_labels.csv')
parsed = parse_data(df)

#split_train_val()
train_path = './train/'
val_path = './val/'
produce_trainval_coco_json(val_path, val=True)
produce_trainval_coco_json(train_path, val=False)