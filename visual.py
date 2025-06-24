import torch
import os
import pickle
import trimesh
from plotly import graph_objects as go
from utils.handmodel import get_handmodel
from utils.plotly_utils import plot_mesh
import numpy as np
from plotly import graph_objects as go
from utils.handmodel import get_handmodel
from utils.plotly_utils import plot_mesh
from utils.rot6d import rot_to_orthod6d
# pt_folder = '/inspurfs/group/mayuexin/datasets/DexGRAB/DexGRAB_shadowhand_downsample.pt'
pt_folder = '/inspurfs/group/mayuexin/datasets/DexGraspNet/dexgraspnet_shadowhand_downsample.pt'
# pt_folder = '/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/unidexgrasp_shadowhand_downsample.pt'
pt_file_path = os.path.join(pt_folder)
grasp_dataset = torch.load(pt_file_path)

# Extract sample_qpos from the data
sample_qpos = grasp_dataset['metadata']

# Define the hand model and set save directory
hand_model = get_handmodel(batch_size=1, device='cpu')
save_dir = '/inspurfs/group/mayuexin/zym/DexGrasp-Anything/visual'
os.makedirs(save_dir, exist_ok=True)

# Loop through each object and visualize its grasp poses
for i, mdata in enumerate(sample_qpos):
    hand_rot_mat = mdata['rotations'].clone().detach().float()  # Use clone().detach() for safety 
    joint_angle = mdata['joint_positions'].clone().detach().float()
    global_trans = mdata['translations'].clone().detach().float()
    object_name = mdata['object_name']
    if  'UniDexGrasp' in pt_folder:
        print(f"UniDexGrasp dataset")
        scale = 1/mdata['scale']
    else:
        scale = mdata['scale']
    # if object_name!='toruslarge':
    #     continue
    # Load object mesh
    # Compute rotation matrix in 6D orthogonal representation
    outputs_3d_rot = rot_to_orthod6d(hand_rot_mat.unsqueeze(0))  # Unsqueeze to add batch dimension
    
    # Initialize outputs tensor
    outputs = torch.zeros((1, 9 + joint_angle.shape[-1]))  # Placeholder for 3 translations + 6D rotation + joint angles

    # Update translations with rotation applied
    outputs[:, :3] = torch.matmul(hand_rot_mat, global_trans)  # Apply rotation to translations

    # Concatenate rotations and joint angles
    outputs[:, 3:9] = outputs_3d_rot.squeeze(0)  # Remove batch dimension after computation
    outputs[:, 9:] = joint_angle 
    outputs[:,11:] =outputs[:,11:]  # Add joint angles
    # mesh_path = os.path.join('/inspurfs/group/mayuexin/datasets/DexGRAB/contact_meshes', f'{object_name}.ply') #DexGRAB
    mesh_path = os.path.join(f'/inspurfs/group/mayuexin/datasets/DexGraspNet/meshdata/{object_name}/coacd', 'decomposed.obj')#DexGraspNet
    # Try to find mesh file in one of these subdirectories  only for Unidexgrasp
    # for dataset_type in {'sem','ddg','core','mujoco'}:
    #     potential_mesh_path = os.path.join(f'/inspurfs/group/mayuexin/datasets/UniDexGrasp/DFCData/meshdatav3/{dataset_type}/{object_name}/coacd', 'decomposed.obj')#UnidexGrasp
    #     if os.path.exists(potential_mesh_path):
    #         mesh_path = potential_mesh_path
    #         break
    
    if os.path.exists(mesh_path):
        obj_mesh = trimesh.load(mesh_path)
        pose_tensor = outputs
        obj_mesh.apply_scale(scale)
        # Update grasp pose using hand_model
        hand_model.update_kinematics(q=pose_tensor)
        # Save object mesh as .ply file
        object_ply_path = os.path.join(save_dir, f'{object_name}_sample-{i}_object.ply')
        obj_mesh.export(object_ply_path)
        print(f"Saved object mesh for {object_name} sample {i} at {object_ply_path}")
        # Get hand mesh data
        hand_mesh_list = hand_model.get_meshes_from_q(q=pose_tensor)

        # Merge hand meshes
        if hand_mesh_list is not None:
            all_vertices = []
            all_faces = []
            vertex_offset = 0

            # Iterate through each hand mesh and merge
            for hand_mesh in hand_mesh_list:
                all_vertices.append(hand_mesh.vertices)
                all_faces.append(hand_mesh.faces + vertex_offset)  # Update face indices with the current vertex offset
                vertex_offset += len(hand_mesh.vertices)  # Update offset

            # Merge all vertices and faces into one large hand mesh
            all_vertices = np.concatenate(all_vertices, axis=0)
            all_faces = np.concatenate(all_faces, axis=0)

            # Create merged Trimesh object
            hand_trimesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)

            # Save merged hand mesh as .ply file
            hand_ply_path = os.path.join(save_dir, f'{i}_hand{object_name}_sample-{i}_hand.ply')
            hand_trimesh.export(hand_ply_path)
            print(f"Saved complete hand mesh for {object_name} sample {i} at {hand_ply_path}")

        # Create plotly data with different colors for object and hand
        vis_data = [plot_mesh(obj_mesh)]  # Object mesh
        vis_data += hand_model.get_plotly_data(opacity=1.0, color='#8799C6')  # Hand mesh

        # Save visualization results as HTML file
        save_html_path = os.path.join(save_dir, f'{object_name}_sample-{i}.html')
        fig = go.Figure(data=vis_data)
        fig.update_layout(scene=dict(bgcolor="white"))  # Set white background
        # Set white background and remove coordinate axes and grid lines
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="white"
            )
        )
        fig.write_html(save_html_path)

        print(f"Saved visualization for {object_name} sample {i} at {save_html_path}")
