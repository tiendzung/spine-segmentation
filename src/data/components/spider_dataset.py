from typing import Optional, List
import monai
import monai.transforms
import SimpleITK as sitk
import numpy as np 
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
import json
import os
import rootutils

rootutils.setup_root(search_from=__file__, indicator="setup.py", pythonpath=True)
from src.data.transforms import array

# always starting with vanilla dataset, like its a norm to me now
class SpiderDataset(Dataset):
    num_class = 15
    def __init__(self, 
                 data = None,
                 data_dir: str = "", 
                 json_path: str = "",
                 ):
        super().__init__()
        self.data = list()
        self.data_dir = data_dir
        if data is not None:
            self.data = data
        else:
            if data_dir == "" or json_path == "":
                raise AssertionError("No dataset ?")
            self.setup(json_path)
    
    def setup(self, json_path):
        json_object = json.load(open(json_path, "r"))
        keys = json_object.keys()
        if "training" in keys:
            for key in keys:
                self.data.extend(json_object[key])
        else:
            try:
                self.data.extend(json_object)
            except:
                raise InsertionError("Something wrong with json file, cannot load or do anything, at all")
    
    def get_item(self, index: int):
        output = dict()
        output["image"] = ""
        output["label"] = ""
        if isinstance(self.data[index]["image"], list):
            output["image"] = [os.path.join(self.data_dir, image) for image in self.data[index]["image"]]
        else:
            # Add for debugging
            path = os.path.join(self.data_dir, self.data[index]["image"])
            output["image"] = [path, path] ##[path, path, path, path] ##Set to 4 because the model_inferr in validation_step
        output["label"] = os.path.join(self.data_dir, self.data[index]["label"])
        return output
    
    # In case they query in list of index
    def __getitem__(self, index):
        output = None
        if not isinstance(index, int):
            output = []
            for id in index:
                output.append(self.get_item(id))
        else:
            output = self.get_item(int(index))
        
        return output
    
    def __len__(self) -> int:
        return len(self.data)
    
    

class SpiderTransformedDataset(Dataset):
    def __init__(self, 
                 dataset: SpiderDataset,
                 transform: monai.transforms.Compose):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, index):
        transformed = self.transform(self.dataset[index])
        if transformed["image"].ndim == 3:
            transformed["image"] = transformed["image"].unsqueeze(0)

        return transformed
    
    def __len__(self) -> int:
        return len(self.dataset)

if __name__=="__main__":
    dataset = SpiderDataset(data_dir = "./data/dataset/spine_nii", json_path="./data/jsons/spine_v3.json")
    # dataset = SpiderDataset(data_dir = "./data/dataset", json_path="/data/hpc/spine/jsons/brats21_folds.json")
    
    transform = monai.transforms.Compose([monai.transforms.LoadImaged(keys=["image", "label"], image_only = False),
                                          array.ConvertToMultiChannelBasedOnSpiderClassesdSemantic(keys=["label"]),
                                        #   monai.transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
                                        #   monai.transforms.Resized(keys=["image", "label"], spatial_size=(250, 250, 155)),
                                          monai.transforms.ToTensord(keys=["image", "label"]),])
    
    transformed = SpiderTransformedDataset(dataset, transform)
    data = dataset[0]
    images = transformed[12]
    print(data)
    print(images["image"].size(), images["image"].dtype)
    print(images["label"].size(), images["label"].dtype)
    print(images.keys())
    print("------------------")
    print(images['image_meta_dict'])
    print("------------------")
    print(images['label_meta_dict'])
    print(images["label"].size())
    
    # res = 0
    # for image in transformed:
    #     res = max(res, image["label"].size(0))
    
    # print(res)