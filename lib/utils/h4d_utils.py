import torch
from collections import deque, defaultdict
from pytorch3d.loss.point_mesh_distance import point_face_distance


def get_loss_weights():
    """Set loss weights"""
    loss_weight = {'chamfer': lambda cst, it: 10. ** 1 * cst * (1 + it),
                   'point2surface': lambda cst, it: 10. ** 1 * cst * (1 + it),
                   'betas': lambda cst, it: 10. ** 0 * cst / (1 + it),
                   'pose_pr': lambda cst, it: 10. ** -1 * cst / (1 + it),
                   }
    return loss_weight


def backward_step(loss_dict, weight_dict, it):
    w_loss = dict()
    for k in loss_dict:
        w_loss[k] = weight_dict[k](loss_dict[k], it)

    tot_loss = list(w_loss.values())
    tot_loss = torch.stack(tot_loss).sum()
    return tot_loss


def compute_p2s_loss(pred_mesh, gt_pcl, seq_len=17):
    ########
    # packed representation for pointclouds
    points = gt_pcl.points_packed()  # (P, 3)
    points_first_idx = gt_pcl.cloud_to_packed_first_idx()
    max_points = gt_pcl.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = pred_mesh.verts_packed()
    faces_packed = pred_mesh.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = pred_mesh.mesh_to_faces_packed_first_idx()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )

    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = gt_pcl.packed_to_cloud_idx()  # (sum(P_i),)
    num_points_per_cloud = gt_pcl.num_points_per_cloud()  # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    weights_p = 1.0 / weights_p.float()
    point_to_face = point_to_face * weights_p
    point_dist = point_to_face.sum() / seq_len
    ########

    return point_dist



class th_Mahalanobis(object):
    def __init__(self, mean, prec, prefix):
        self.mean = torch.tensor(mean.astype('float32'), requires_grad=False).unsqueeze(axis=0).cuda()
        self.prec = torch.tensor(prec.astype('float32'), requires_grad=False).cuda()
        self.prefix = prefix

    def __call__(self, pose, prior_weight=1.):
        '''
        :param pose: Batch x pose_dims
        :return:
        '''
        # return (pose[:, self.prefix:] - self.mean)*self.prec
        temp = pose[:, self.prefix:] - self.mean
        temp2 = torch.matmul(temp, self.prec) * prior_weight
        return (temp2 * temp2).sum(dim=1)


class Prior(object):
    def __init__(self, sm, prefix=3):
        self.prefix = prefix
        if sm is not None:
            # Compute mean and variance based on the provided poses
            self.pose_subjects = sm.pose_subjects
            all_samples = [p[prefix:] for qsub in self.pose_subjects
                           for name, p in zip(qsub['pose_fnames'], qsub[
                    'pose_parms'])]  # if 'CAESAR' in name or 'Tpose' in name or 'ReachUp' in name]
            self.priors = {'Generic': self.create_prior_from_samples(all_samples)}
        else:
            import pickle as pkl
            # Load pre-computed mean and variance
            dat = pkl.load(open('/home/add_disk/jiangboyan/proj/cloth4d/data/pose_prior.pkl', 'rb'))
            self.priors = {'Generic': th_Mahalanobis(dat['mean'],
                                                     dat['precision'],
                                                     self.prefix)}

    def create_prior_from_samples(self, samples):
        from sklearn.covariance import GraphicalLassoCV
        from numpy import asarray, linalg
        model = GraphicalLassoCV()
        model.fit(asarray(samples))
        return th_Mahalanobis(asarray(samples).mean(axis=0),
                              linalg.cholesky(model.precision_),
                              self.prefix)

    def __getitem__(self, pid):
        if pid not in self.priors:
            samples = [p[self.prefix:] for qsub in self.pose_subjects
                       for name, p in zip(qsub['pose_fnames'], qsub['pose_parms'])
                       if pid in name.lower()]
            self.priors[pid] = self.priors['Generic'] if len(samples) < 3 \
                else self.create_prior_from_samples(samples)

        return self.priors[pid]