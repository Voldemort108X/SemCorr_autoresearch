# Copied from /workspace/SemCorr_proj/Code/SemCorrVLM_cvpr26/semcorr_vfm/SemCorr2D/utils/utils_geoware.py (SPAIR_FLIP, AP10K_FLIP, renumber_indices, permute_indices, flip_keypoints)
import torch
SPAIR_FLIP = {
    'aeroplane': [0,1,2,3,[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19],[20,21],22,23,24],
    # 'bicycle': [0,1,[2,3],4,5,[6,7],8,[9,10],11+2], #11,12 are dummy
    'bicycle': [0,1,[2,3],4,5,[6,7],8,[9,10],11], 
    'bird': [0,[1,2],3,[4,5],6,[7,8],9,[10,11],[12,13],[14,15],16],
    'boat': [0,[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],13],
    'bottle': [[0,1],[2,3],[4,5],[6,7],[8,9]],
    # 'bus': [[0,1],[2,3],[5,6],4,7,[8+2,18+2],[11+2,21+2],[9+2,19+2],[12+2,22+2],[10+2,20+2],[13+2,23+2],[14+2,15+2],[24+2,25+2],[16+2,17+2],[26+2,27+2]],  #8,9 are dummy
    'bus': [[0,1],[2,3],[5,6],4,7,[8,18],[11,21],[9,19],[12,22],[10,20],[13,23],[14,15],[24,25],[16,17],[26,27]], 
    'car':[[0,1],[2,3],4,5,[6,7],8,9,[10,20],[13,23],[11,21],[14,24],[12,22],[15,25],[16,17],[26,27],[18,19],[28,29]],
    'cat':[[0,1],[2,3],[4,5],[6,7],8,[9,10],[11,12],13,14],
    'chair':[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13]],
    'cow':[[0,1],[2,3],[4,5],[6,7],8,[9,10],[11,12],13,14,[15,16],[17,18],[19,20]],
    'dog':[[0,1],[2,3],[4,5],6,7,8,[9,10],[11,12],13,14,15],
    'horse':[[0,1],[2,3],[4,5],[6,7],8,9,[10,11],[12,13],14,15,[16,17],[18,19]],
    'motorbike':[[0,1],[2,3],4,5,6,7,8,9,10,11,12],
    'person':[[0,1],[2,3],4,5,6,7,[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]],
    'pottedplant':[[0,2],1,3,[4,5],[6,8],7],
    'sheep':[[0,1],[2,3],[4,5],[6,7],8,[9,10],[11,12],13,14,[15,16],[17,18],[19,20]],
    'train':[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17]],
    'tvmonitor':[[0,2],[4,6],1,5,[3,7],[8,10],[12,14],9,13,[11,15]]
}

AP10K_FLIP = [
                [0,1], # eye
                2, # nose
                3, # neck
                4, # root of tail
                [5,8], # shoulder
                [6,9], # elbow # knee
                [12, 15], # knee
                [7,10], # front paw 
                [13,16], # back paw
                [11,14], # hip
                              ]

def renumber_indices(lst, counter=[0]):
    """
    input: a nested list
    output: a nested list with all the numbers replaced by the next number in the counter
    """
    new_lst = []
    for item in lst:
        if isinstance(item, list):
            new_lst.append(renumber_indices(item, counter))
        else:
            new_lst.append(counter[0])
            counter[0] += 1
    return new_lst


def permute_indices(flip_list, vis=None):
    """
    Permutes indices based on the rules provided in the flip_list.
    
    Args:
    - flip_list (list): List containing integers or lists of integers.
    - vis (list, optional): A list of boolean values used to determine whether a particular permutation should be applied.
    
    Returns:
    - List[int]: Permuted indices list.

    Sample Input 1:
    flip_list = [1, 2, [0, 3]]
    vis = [True, True, True, False]
    Output:
    [0, 1, 2, 3]

    Sample Input 1:
    flip_list = [1, 2, [0, 3]]
    vis = None
    Output:
    [3, 1, 2, 0]
    """
    # Flatten the list to find the max index
    flat_list = [item for sublist in flip_list for item in (sublist if isinstance(sublist, list) else [sublist])]
    max_idx = max(flat_list)

    # Create a list of indices from 0 to max index
    indices = list(range(max_idx + 1))

    # Permute the indices where necessary
    for item in flip_list:
        if isinstance(item, list):
            # if all elem in item is True in vis, then flip
            if vis is None or all(vis[i] for i in item):
                for i in range(len(item)):
                    indices[item[i]] = item[(i + 1) % len(item)]

    return indices


def flip_keypoints(keypoints, img_size, permute_list=None):
    keypoints_flip = keypoints.detach().clone()
    keypoints_flip[:, 0] = img_size - keypoints_flip[:, 0]
    if permute_list is not None:
        keypoints_flip = keypoints_flip[permute_list]
    return keypoints_flip


