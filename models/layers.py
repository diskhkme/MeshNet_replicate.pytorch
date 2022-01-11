import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter # Parameter is a tensor that is added as a parameter of a module

class SpatialDescriptor(nn.Module):
    def __init__(self):
        super(SpatialDescriptor, self).__init__()

        self.spatial_mlp = nn.Sequential(
            nn.Conv1d(3,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

    def forward(self, centers):
        # (batch_num, 3, num_face) --> (batch_num, 64, num_face): simple feature encoder for center points
        return self.spatial_mlp(centers)

class FaceRotateConvolution(nn.Module):
    def __init__(self):
        super(FaceRotateConvolution, self).__init__()

        # f( , ) function
        self.rotate_mlp = nn.Sequential(
            nn.Conv1d(6, 32, 1), # (v1,v2) (v2,v3), (v3,v1) pair constitutes 6 dims?
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # g() function
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

    def forward(self, corners):
        # (batch_num(b), 9, num_face) -- (b, 6(v1,v2), num_face) --mlp--> (b, 32, num_face) --|
        #                            |-- (b, 6(v2,v3), num_face) --mlp--> (b, 32, num_face) --|-->mean--> mlp --> (b, 64, num_face)
        #                            |-- (b, 6(v3,v1), num_fave) --mlp--> (b, 32, num_face) --|
        feature = (self.rotate_mlp(corners[:, :6]) + # f(v1,v2)
                   self.rotate_mlp(corners[:, 3:9]) + # f(v2,v3)
                   self.rotate_mlp(torch.cat([corners[:,6:], corners[:,:3]], dim=1))) / 3 # f(v3,v1)

        return self.fusion_mlp(feature)

class FaceKernelCorrelation(nn.Module):
    def __init__(self, num_kernel=64, sigma=0.2):
        super(FaceKernelCorrelation, self).__init__()
        self.num_kernel = num_kernel
        self.sigma = sigma
        self.weight_alpha = Parameter(torch.rand(1, num_kernel, 4) * np.pi)
        self.weight_beta = Parameter(torch.rand(1, num_kernel, 4) * 2 * np.pi)
        self.bn = nn.BatchNorm1d(num_kernel)
        self.relu = nn.ReLU()

    def forward(self, normals, neighbor_index):
        b, _, n = normals.size() # (b, 3, num_face)

        center = normals.unsqueeze(2).expand(-1, -1, self.num_kernel, -1).unsqueeze(4) # (b, 3, num_kernel, num_face, 1)
        neighbor = torch.gather(normals.unsqueeze(3).expand(-1, -1, -1, 3), 2,
                                neighbor_index.unsqueeze(1).expand(-1, 3, -1, -1)) # (b, 3, num_face, 3), 2nd axis is x,y,z of normal vector
        neighbor = neighbor.unsqueeze(2).expand(-1, -1, self.num_kernel, -1, -1) # (b, 3, num_kernel, num_face, 3)

        fea = torch.cat([center, neighbor], 4) # (b, 3, num_kernel, num_face, 3+1), normal of self and neighborhood face is repeating num_kernel times
        fea = fea.unsqueeze(5).expand(-1, -1, -1, -1, -1, 4) # above is repeating four times

        # learnable vector of kernels (theta and phi parameter will be learned)
        weight = torch.cat([torch.sin(self.weight_alpha) * torch.cos(self.weight_beta),
                            torch.sin(self.weight_alpha) * torch.sin(self.weight_beta),
                            torch.cos(self.weight_alpha)], 0) # (3, num_kernel, 4) # why 4?
        weight = weight.unsqueeze(0).expand(b, -1, -1, -1)
        weight = weight.unsqueeze(3).expand(-1, -1, -1, n, -1)
        weight = weight.unsqueeze(4).expand(-1, -1, -1, -1, 4, -1) # (b, 3, num_kernel, num_face, 4, 4)

        dist = torch.sum((fea - weight)**2, 1) # differnce of feat(self and neighbor normal) and kernel vectors
        fea = torch.sum(torch.sum(np.e**(dist / (-2 * self.sigma**2)), 4), 3) / 16 # Gaussian kernel, mean

        # fea: (b, num_kernel, num_point)
        # In summary, vector of kernels(=weight) learned the distribution of normal in training dataset?
        # activation getting stronger if input model is more correlated with learned kernels.
        # While training, weight alpha and beta for kernel is learned to best represent distribution of normals in the dataset

        return self.relu(self.bn(fea))

class StructuralDescriptor(nn.Module):
    def __init__(self, cfg):
        super(StructuralDescriptor, self).__init__()

        self.FRC = FaceRotateConvolution()
        self.FKC = FaceKernelCorrelation(cfg['num_kernel'], cfg['sigma'])
        self.structural_mlp = nn.Sequential(
            nn.Conv1d(64 + 3 + cfg['num_kernel'], 131, 1),
            nn.BatchNorm1d(131),
            nn.ReLU(),
            nn.Conv1d(131, 131, 1),
            nn.BatchNorm1d(131),
            nn.ReLU(),
        )

    def forward(self, corners, normals, neighbor_index):
        structural_fea1 = self.FRC(corners) # (b, 64, num_face) from face rotate conv
        structural_fea2 = self.FKC(normals, neighbor_index) # # (b, num_kernel, num_face) from face kernel corr

        # (b, 64+num_kernel+3(normal), num_face) --mlp--> (b, 131, num_face)
        return self.structural_mlp(torch.cat([structural_fea1, structural_fea2, normals], 1))

if __name__ == '__main__':
    num_batch = 16
    num_point = 1024

    centers = torch.rand(num_batch,3,num_point)
    corners = torch.rand(num_batch,9,num_point)
    normals = torch.rand(num_batch,3,num_point)
    normals = normals/torch.linalg.norm(normals,dim=1).unsqueeze(1) # normalize
    neighbor_indices = torch.randint(0,num_point,(num_batch,num_point,3),dtype=torch.long)

    # spatial desc
    spatial_desc = SpatialDescriptor()
    spatial_feat = spatial_desc(centers)
    print(spatial_feat.shape)

    # # face rotate conv
    # face_rot_conv = FaceRotateConvolution()
    # out = face_rot_conv(corners)
    # print(out.shape)
    #
    # # face kernel corr
    # face_kernel_corr = FaceKernelCorrelation()
    # out = face_kernel_corr(normals, neighbor_indices)
    # print(out.shape)

    # structural desc (face rot + face kernel)
    cfg = {'num_kernel':64, 'sigma':0.2}
    sd = StructuralDescriptor(cfg)
    structural_feat = sd(corners,normals,neighbor_indices)
    print(structural_feat.shape)

