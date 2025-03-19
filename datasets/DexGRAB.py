from typing import Any, Tuple, Dict
import os
import pickle
import torch
import trimesh
import numpy as np
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig, OmegaConf
import transforms3d
from datasets.misc import collate_fn_squeeze_pcd_batch_grasp
from datasets.transforms import make_default_transform
from datasets.base import DATASET
import json
from utils.registry import Registry
from termcolor import cprint
def load_from_json(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data["_train_split"], data["_test_split"], data["_all_split"]


@DATASET.register()
class DexGRAB(Dataset):
    """ Dataset for pose generation, training with DexGraspNet Dataset
    """

    # read json
    _all_split = ['cylindersmall', 'cylindermedium', 'toruslarge', 'camera', 'train', 'mug', 'knife', 'binoculars', 'spheresmall', 
                  'airplane', 'torusmedium', 'rubberduck', 'apple', 'cubesmall', 'wristwatch', 'cylinderlarge', 'flute', 'stamp', 
                  'scissors', 'bowl', 'pyramidlarge', 'toothbrush', 'cubemedium', 'teapot', 'duck', 'gamecontroller', 'hammer', 
                  'flashlight', 'waterbottle', 'torussmall', 'headphones', 'mouse', 'cubemiddle', 'stapler', 'elephant', 'piggybank', 
                  'alarmclock', 'cubelarge', 'cup', 'wineglass', 'lightbulb', 'watch', 'phone', 'eyeglasses', 'spherelarge', 'spheremedium', 
                  'toothpaste', 'doorknob', 'stanfordbunny', 'hand', 'coffeemug', 'pyramidmedium', 'fryingpan', 'table', 'pyramidsmall', 'banana']
    _train_split = ['fryingpan', 'binoculars', 'mug', 'bowl', 'phone', 'hand', 'doorknob', 'spheresmall', 'pyramidlarge', 'cylindersmall', 
                    'spheremedium', 'piggybank',  'alarmclock', 'scissors', 'wineglass', 'knife', 'headphones', 'flashlight', 
                    'cubelarge', 'elephant', 'watch', 'cubesmall', 'stanfordbunny', 'toruslarge', 'teapot', 'airplane', 'flute','train' , 
                    'banana', 'apple', 'toothbrush', 'mouse','spherelarge', 'stapler', 'stamp','toothpaste', 'lightbulb', 'hammer', 'cubemedium', 'waterbottle', 'cup']
    _test_split = ['torusmedium',   'cylinderlarge' ,  'eyeglasses', 'cylindermedium', 'camera', 'pyramidmedium', 'gamecontroller', 
                   'duck', 'pyramidsmall', 'torussmall', 'table', 'wristwatch','coffeemug', 'rubberduck','cubemiddle']



    # joint limits from URDF file
    _joint_angle_lower = torch.tensor([-0.5235988, -0.7853982, -0.43633232, 0., 0., 0., -0.43633232, 0., 0., 0.,
                                       -0.43633232, 0., 0., 0., 0., -0.43633232, 0., 0., 0., -1.047, 0., -0.2618,
                                       -0.5237, 0.])
    _joint_angle_upper = torch.tensor([0.17453292, 0.61086524, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 0.43633232,
                                       1.5707964, 1.5707964, 1.5707964, 0.43633232, 1.5707964, 1.5707964, 1.5707964,
                                       0.6981317, 0.43633232, 1.5707964,  1.5707964, 1.5707964, 1.047, 1.309, 0.2618,
                                       0.5237, 1.])

    _global_trans_lower = torch.tensor([-0.13128923, -0.10665303, -0.45753425])
    _global_trans_upper = torch.tensor([0.12772022, 0.22954416, -0.21764427])


    _NORMALIZE_LOWER = -1.
    _NORMALIZE_UPPER = 1.

    def __init__(self, cfg: DictConfig, phase: str, slurm: bool, case_only: bool=False, **kwargs: Dict) -> None:
        super(DexGRAB, self).__init__()
        self.phase = phase
        self.slurm = slurm
        if self.phase == 'train':
            self.split = self._train_split
        elif self.phase == 'test':
            self.split = self._test_split
        elif self.phase == 'all':
            self.split = self._all_split
        else:
            raise Exception('Unsupported phase.')
        self.datasetname = cfg.name
        self.device = cfg.device
        self.is_downsample = cfg.is_downsample
        self.modeling_keys = cfg.modeling_keys
        self.num_points = cfg.num_points
        self.use_color = cfg.use_color
        self.use_normal = cfg.use_normal
        self.normalize_x = cfg.normalize_x
        self.normalize_x_trans = cfg.normalize_x_trans
        self.obj_dim = int(3 + 3 * self.use_color + 3 * self.use_normal)
        self.transform = make_default_transform(cfg, phase)
        self.use_llm=cfg.use_llm
        ## resource folders
        self.asset_dir = cfg.asset_dir_slurm if self.slurm else cfg.asset_dir
        self.data_dir = self.asset_dir
        self.scene_path = os.path.join(self.asset_dir, 'object_pcds_nors.pkl')
        self._joint_angle_lower = self._joint_angle_lower.cpu()
        self._joint_angle_upper = self._joint_angle_upper.cpu()
        self._global_trans_lower = self._global_trans_lower.cpu()
        self._global_trans_upper = self._global_trans_upper.cpu()
        cprint(
            f"[Dataset]: {self.datasetname} \n"
            f"• Use LLM: {self.use_llm} \n"
            f"• Asset Path: {self.asset_dir} \n",
            "yellow")
        ## load data
        self._pre_load_data(case_only)

    def _pre_load_data(self, case_only: bool) -> None:
        """ Load dataset

        Args:
            case_only: only load single case for testing, if ture, the dataset will be smaller.
                        This is useful in after-training visual evaluation.
        """
        self.frames = []
        self.scene_pcds = {}
        grasp_dataset_file = os.path.join(self.data_dir, 'DexGRAB_shadowhand_downsample.pt')
        grasp_dataset = torch.load(grasp_dataset_file)
        self.scene_pcds = pickle.load(open(self.scene_path, 'rb'))
        self.dataset_info = grasp_dataset['info']
        if self.use_llm:
            # Getting descriptions from LLM
            scene_text_file =os.path.join(self.data_dir,"DexGRAB_gpt4o_mini.json")
            with open(scene_text_file, "r") as jsonfile:
                self.scene_text = json.load(jsonfile)
            # pre-process for tokenizer
            for k, text in self.scene_text.items():
                txtclips = text.split("\n")
                self.scene_text[k] = txtclips[:]
        for mdata in grasp_dataset['metadata']:
            hand_rot_mat = mdata['rotations'].numpy()
            joint_angle = mdata['joint_positions'].clone().detach()##24dof
            global_trans = mdata['translations'].clone().detach()
            if self.normalize_x:
                joint_angle = self.angle_normalize(joint_angle)
            if self.normalize_x_trans:
                global_trans = self.trans_normalize(global_trans)
            mdata_qpos = torch.cat([global_trans, joint_angle], dim=0).requires_grad_(True)
            if mdata['object_name'] in self.split:
                self.frames.append({'robot_name': 'shadowhand',
                                    'object_name': mdata['object_name'],
                                    'object_rot_mat': hand_rot_mat.T,
                                    'qpos': mdata_qpos})

    def trans_normalize(self, global_trans: torch.Tensor):
        global_trans_norm = torch.div((global_trans - self._global_trans_lower), (self._global_trans_upper - self._global_trans_lower))
        global_trans_norm = global_trans_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        return global_trans_norm

    def trans_denormalize(self, global_trans: torch.Tensor):
        global_trans_denorm = global_trans + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        global_trans_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        global_trans_denorm = global_trans_denorm * (self._global_trans_upper - self._global_trans_lower) + self._global_trans_lower
        return global_trans_denorm

    def angle_normalize(self, joint_angle: torch.Tensor):
        joint_angle_norm = torch.div((joint_angle - self._joint_angle_lower), (self._joint_angle_upper - self._joint_angle_lower))
        joint_angle_norm = joint_angle_norm * (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) - (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        return joint_angle_norm

    def angle_denormalize(self, joint_angle: torch.Tensor):
        joint_angle_denorm = joint_angle + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        joint_angle_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        joint_angle_denorm = joint_angle_denorm * (self._joint_angle_upper - self._joint_angle_lower) + self._joint_angle_lower
        return joint_angle_denorm

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, index: Any) -> Tuple:

        frame = self.frames[index]

        ## load data, containing scene point cloud and point pose
        scene_id = frame['object_name']
        scene_rot_mat = frame['object_rot_mat']
        scene_pc = self.scene_pcds[scene_id]
        nor = np.einsum('mn, kn->km', scene_rot_mat, scene_pc[:,3:6])
        scene_pc = np.einsum('mn, kn->km', scene_rot_mat, scene_pc[:,:3])
        cam_tran = None

        ## randomly resample points
        if self.phase != 'train':
            np.random.seed(0) # resample point cloud with a fixed random seed
        resample_indices = np.random.permutation(len(scene_pc))
        scene_pc = scene_pc[resample_indices[:self.num_points]]
        nor = nor[resample_indices[:self.num_points]]
        ## format point cloud xyz and feature
        xyz = scene_pc[:, 0:3]
        nor = nor[:, 0:3]
        if self.use_color:
            color = scene_pc[:, 3:6] / 255.
            feat = np.concatenate([color], axis=-1)



        ## format smplx parameters
        grasp_qpos = (
            frame['qpos']
        )
        
        data = {
            'x': grasp_qpos,
            'pos': xyz,
            'scene_rot_mat': scene_rot_mat,
            'cam_tran': cam_tran, 
            'scene_id': scene_id,
            'normal': nor
        }
        if self.use_normal:
            normal = nor
            feat = np.concatenate([normal], axis=-1)
            data['feat'] = feat
            
        if self.transform is not None:
            data = self.transform(data, modeling_keys=self.modeling_keys)
            
        if self.use_llm:  
            data['text'] = self.scene_text[scene_id]
            
        return data

    def get_dataloader(self, **kwargs):
        return DataLoader(self, **kwargs)


if __name__ == '__main__':
    config_path = "../configs/task/grasp_gen.yaml"
    cfg = OmegaConf.load(config_path)
    dataloader = DexGRAB(cfg.dataset, 'train', False).get_dataloader(batch_size=4,
                                                                    collate_fn=collate_fn_squeeze_pcd_batch_grasp,
                                                                    num_workers=0,
                                                                    pin_memory=True,
                                                                    shuffle=True,)

    print(len(dataloader))