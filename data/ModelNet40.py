import numpy as np
import os
import torch
import torch.utils.data as data

type_to_index_map = {
    'night_stand': 0, 'range_hood': 1, 'plant': 2, 'chair': 3, 'tent': 4,
    'curtain': 5, 'piano': 6, 'dresser': 7, 'desk': 8, 'bed': 9,
    'sink': 10,  'laptop':11, 'flower_pot': 12, 'car': 13, 'stool': 14,
    'vase': 15, 'monitor': 16, 'airplane': 17, 'stairs': 18, 'glass_box': 19,
    'bottle': 20, 'guitar': 21, 'cone': 22,  'toilet': 23, 'bathtub': 24,
    'wardrobe': 25, 'radio': 26,  'person': 27, 'xbox': 28, 'bowl': 29,
    'cup': 30, 'door': 31,  'tv_stand': 32,  'mantel': 33, 'sofa': 34,
    'keyboard': 35, 'bookshelf': 36,  'bench': 37, 'table': 38, 'lamp': 39
}

class ModelNet40(data.Dataset):
    def __init__(self, cfg, part='train'):
        self.root = cfg['data_root']
        self.max_faces = cfg['max_faces']
        self.part = part
        self.augment_data = cfg['augment_data']
        if self.augment_data:
            self.jitter_sigma = cfg['jitter_sigma']
            self.jitter_clip = cfg['jitter_clip']

        self.data = []
        for type in os.listdir(self.root):
            if type not in type_to_index_map.keys():
                continue
            type_index = type_to_index_map[type]
            type_root = os.path.join(os.path.join(self.root, type), part)
            for filename in os.listdir(type_root):
                if filename.endswith('.npz') or filename.endswith('.obj'):
                    self.data.append((os.path.join(type_root, filename), type_index))

    def __getitem__(self, i):
        path, type = self.data[i]
        if path.endswith('.npz'):
            data = np.load(path)
            face = data['faces'] # center(vec3), corners(3 vertices * vec3), normal(vec3) --> 15 values per face
            neighbor_index = data['neighbors'] # index of 3 neighborhood faces per face
        else:
            raise Exception('Process for other format is currently ommitted.')
            # face, neighbor_index = process_mesh(path, self.max_faces)
            # if face is None:
            #     return self.__getitem__(0)

        # data augmentation
        if self.augment_data and self.part == 'train':
            # jitter
            jittered_data = np.clip(self.jitter_sigma * np.random.randn(*face[:, :3].shape), -1 * self.jitter_clip,
                                    self.jitter_clip) # jitter center values
            face = np.concatenate((face[:, :3] + jittered_data, face[:, 3:]), 1)

        # fill for n < max_faces with randomly picked faces
        num_point = len(face) # if face is not enough,
        if num_point < self.max_faces:
            fill_face = []
            fill_neighbor_index = []
            for i in range(self.max_faces - num_point):  # add duplicated face data up to max_faces
                index = np.random.randint(0, num_point)
                fill_face.append(face[index])
                fill_neighbor_index.append(neighbor_index[index])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))

        # to tensor
        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()
        target = torch.tensor(type, dtype=torch.long) # target label

        # reorganize
        face = face.permute(1, 0).contiguous() # (15, num_face)
        centers, corners, normals = face[:3], face[3:12], face[12:] # (3, num_face), (9, num_face), (3, num_face)
        corners = corners - torch.cat([centers, centers, centers], 0) # center-relative coord of corner

        return centers, corners, normals, neighbor_index, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    from config.config import get_train_config
    cfg = get_train_config(config_file='../config/train_config.yaml')
    dataset = ModelNet40(cfg=cfg['dataset'], part='train')

    centers, corners, normals, neighbor_index, target = dataset[0]
    print(centers)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    center_array = centers.numpy()
    center_x = center_array[0,:]
    center_y = center_array[1, :]
    center_z = center_array[2, :]
    ax.scatter(center_x, center_y, center_z)
    plt.show()