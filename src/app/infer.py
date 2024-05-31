import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
from functools import partial

import nibabel as nib
import numpy as np

from lightning import LightningModule
import torch
from torch.utils.data import DataLoader, Dataset
from src.models.brats21_module import Brats21LitModule

from monai.inferers import sliding_window_inference

class Inferer(object):
    def __init__(
        self,
        exp_name: str,
        checkpoint_path: str,
        data_name: str,
        data_loader: DataLoader,
        net: torch.nn.Module,
        model_inferer: sliding_window_inference,
        device
    ):
        self.output_directory = "./outputs/" + exp_name
        print(os.getcwd())
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            
        self.data_name = data_name
        self.data_loader = data_loader
        net.to(device)
        self.net = net
        model = Brats21LitModule.load_from_checkpoint(checkpoint_path=checkpoint_path, net = net)
        self.model_inferer = partial(model_inferer, predictor=model)
      
    def feed(self):
        with torch.no_grad():
            for i, batch in enumerate(self.data_loader):
                image = batch["image"].cuda()
                print(batch.keys())
                print(image.size())
                affine = batch["image_meta_dict"]["original_affine"][0].numpy()
                num = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1].split("_")[1]
                img_name = self.data_name + "_" + num + ".nii.gz"
                print("Inference on case {}".format(img_name))
                prob = torch.sigmoid(self.model_inferer(image)) ## this or that
                # prob = torch.sigmoid(self.net(image)) ## that
                seg = prob[0].detach().cpu().numpy()
                seg = (seg > 0.5).astype(np.int8)
                print(seg.shape)
                seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
                seg_out[seg[1] == 1] = 2
                seg_out[seg[0] == 1] = 1
                seg_out[seg[2] == 1] = 4
                nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), os.path.join(self.output_directory, img_name))
            print("Finished inference!")
        
        # for i, batch in enumerate(self.data_loader):
        #     print(batch['image'].size())
        #     if (i == 0):
        #         print(batch['image'][0])
            
if __name__ == "__main__":
    import hydra
    from omegaconf import OmegaConf, DictConfig
    
    config_path = os.path.join(os.environ["PROJECT_ROOT"], "configs/app")
    
    @hydra.main(version_base="1.3", config_path=config_path, config_name="infer")
    def main(cfg: DictConfig):
        # print(OmegaConf.to_yaml(cfg))
        inferer: Inferer = hydra.utils.instantiate(cfg)
        print(inferer)
        # print(type(inferer))
        inferer.feed()
        
    main()
