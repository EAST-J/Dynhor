import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def min_distance_to_point_set(A, B):
    """
    计算点集A中每个点到点集B的所有点的最小欧氏距离。
    
    :param A: 点集A，形状为 (N_A, 3)
    :param B: 点集B，形状为 (N_B, 3)
    :return: 每个点到B点集的最小距离，形状为 (N_A,)
    """
    # 计算每个点集A中的点到点集B中的所有点的距离
    A_expanded = A.unsqueeze(1)  # 扩展A为 (N_A, 1, 3)
    B_expanded = B.unsqueeze(0)  # 扩展B为 (1, N_B, 3)

    # 计算每个点的欧氏距离
    dist = torch.norm(A_expanded - B_expanded, dim=2)  # 计算 (N_A, N_B) 形状的距离矩阵

    # 对每个点，取到B中所有点的最小距离
    min_distances, _ = dist.min(dim=1)  # 选择每一行的最小值

    return min_distances

def to_cartesian(coords, shape):
    i, j = (torch.from_numpy(inds) for inds in np.unravel_index(coords.cpu(), shape=shape))
    return torch.stack([i, j], dim=-1)

def find_semantic_correspondences(img_render_cos, sample_mask, render_mask, img_feat, dino_feat_h, dino_feat_w, correspondence_nums=50):
    '''
    img_render_cos: sample_to_render similarity matrix
    sample_mask: (dino_feat_h * dino_feat_w)
    render_mask: (dino_feat_h * dino_feat_w)
    img_feat: (dino_feat_h * dino_feat_w)
    '''
    sim_1, nn_1 = torch.max(img_render_cos, dim=-1, keepdim=False) # img2render max
    sim_2, nn_2 = torch.max(img_render_cos, dim=-2, keepdim=False) # render2img max

    nn_2[~render_mask.bool()] = 0
    cyclical_idxs = torch.gather(nn_2, dim=-1, index=nn_1)
    image_idxs = torch.arange(dino_feat_h * dino_feat_w)

    cyclical_idxs_ij = to_cartesian(cyclical_idxs, shape=[dino_feat_h, dino_feat_w])
    image_idxs_ij = to_cartesian(image_idxs, shape=[dino_feat_h, dino_feat_w])

    zero_mask = (cyclical_idxs_ij - torch.Tensor([0, 0])[None, :]).sum(dim=1) == 0
    cyclical_idxs_ij[zero_mask] = dino_feat_h ** 2

    hw, ij_dim = cyclical_idxs_ij.size()
    cyclical_dists = -torch.nn.PairwiseDistance(p=2)(cyclical_idxs_ij.view(-1, ij_dim), image_idxs_ij.view(-1, ij_dim))
    cyclical_dists_norm = cyclical_dists - cyclical_dists.min()
    cyclical_dists_norm /= cyclical_dists_norm.max()
    cyclical_dists_norm *= sample_mask.float().to(cyclical_dists_norm.device)

    sorted_vals, topk_candidate_points_image_1 = cyclical_dists_norm.sort(dim=-1, descending=True)
    topk_candidate_points_image_1 = topk_candidate_points_image_1[:correspondence_nums]

    num_pairs_to_return = correspondence_nums // 2
    selected_points_image_1 = []

    feats_b = img_feat[topk_candidate_points_image_1]

    kmeans = KMeans(n_clusters=num_pairs_to_return, random_state=0).fit(feats_b.cpu())
    kmeans_labels = torch.as_tensor(kmeans.labels_)
    final_idxs_chosen_from_image_1_b = []

    for k in range(num_pairs_to_return):
        locations_in_cluster_k = torch.where(kmeans_labels == k)[0]
        saliencies_at_k = sample_mask[topk_candidate_points_image_1][locations_in_cluster_k]
        point_chosen_from_cluster_k = saliencies_at_k.argmax()
        final_idxs_chosen_from_image_1_b.append(topk_candidate_points_image_1[locations_in_cluster_k][point_chosen_from_cluster_k])

    final_idxs_chosen_from_image_1_b = torch.stack(final_idxs_chosen_from_image_1_b)
    selected_points_image_1.append(final_idxs_chosen_from_image_1_b)

    selected_points_image_1 = torch.stack(selected_points_image_1).reshape(-1)
    selected_points_image_2 = torch.gather(nn_1, dim=-1, index=selected_points_image_1)
    return selected_points_image_1, selected_points_image_2

def visualize_correspondences(img1, img2, points1, points2, file_name='2d_corres.jpg'):
    """
    可视化两张图像及其对应点之间的连接线
    :param img1: 第一张图像（numpy array）
    :param img2: 第二张图像（numpy array）
    :param points1: 第一张图像的对应点（numpy array, shape: (n, 2)）
    :param points2: 第二张图像的对应点（numpy array, shape: (n, 2)）
    """
    # 创建一个新的图像区域，图像2放在图像1的右边
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 显示两张图像（img2 会被拼接到 img1 的右边）
    ax.imshow(np.hstack((img1, img2)), cmap='gray')  # 拼接两张图像

    # 绘制第一张图像的点（红色）
    ax.scatter(points1[:, 0], points1[:, 1], color='red', label='Points in img1')

    # 绘制第二张图像的点（蓝色），需要将其 X 坐标加上 img1 的宽度
    ax.scatter(points2[:, 0] + img1.shape[1], points2[:, 1], color='blue', label='Points in img2')

    # 绘制对应点之间的连接线
    for p1, p2 in zip(points1, points2):
        ax.plot([p1[0], p2[0] + img1.shape[1]], [p1[1], p2[1]], 'g-', lw=1)  # 连接两张图像的点

    ax.legend()
    ax.set_title('Correspondences between Images')

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()