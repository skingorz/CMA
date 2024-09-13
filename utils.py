import math
import torch

def MIR(matrix1, matrix2):
    """
    Matrix Information Ratio
    """
    mie = MIE()
    entropy1 = mie(matrix1)
    entropy2 = mie(matrix2)
    entropy3 = mie(matrix1 * matrix2)
    mi = entropy1 + entropy2 - entropy3

    return mi/torch.min(entropy1, entropy2)


def HDR(matrix1, matrix2):
    """
    Information Entry Difference Ratio
    """
    mie = MIE()
    entropy1 = mie(matrix1)
    entropy2 = mie(matrix2)

    return torch.abs(entropy1-entropy2)/torch.max(entropy1, entropy2)

    
class MIE(torch.nn.Module):
    """
    Matrix Information Entropy
    """

    def __init__(self):
        super().__init__()

    def cal_gram(self, R):
        Z = torch.nn.functional.normalize(R, dim=-1)
        A = torch.matmul(Z, Z.transpose(-1, -2))

        return A
    
    def cal_entropy(self, A):
        if A.dim() == 2:
            A = A.unsqueeze(0)
        assert A.dim() == 3
    
        I = torch.eye(A.shape[1], device=A.device).unsqueeze(0)
        A += (1e-6 * I)
        traces = torch.diagonal(A, dim1=-2, dim2=-1).sum(dim=-1)
        # Ensure trace is not too close to zero to avoid division by zero or very small numbers.
        traces = torch.clamp(traces, min=1e-7)
        A_normalized = A / traces.unsqueeze(-1).unsqueeze(-1)
        # Compute eigenvalues
        eig_val, _ = torch.linalg.eigh(A_normalized)
        # Clamp eigenvalues to avoid log(0). The clamp value can be very small but should be > 0.
        eig_val = torch.clamp(eig_val, min=1e-7)
        # Calculate entropy. Use log1p for better numerical stability.
        entropy = -torch.sum(eig_val * torch.log1p(eig_val - 1), dim=-1)
        # Normalize entropy
        normalized_entropy = entropy / torch.log(torch.tensor(A.shape[1], dtype=torch.float, device=A.device))

        return normalized_entropy


    def forward(self, R1):

        R1 = R1.float()
        R1 = self.cal_gram(R1)
        return torch.mean(self.cal_entropy(R1))
    
def get_visual_prototype(labels, features):
    '''
    calculate visual prototype based on image feature and corresponding labels
    '''

    unique_labels, new_label_indices = labels.unique(return_inverse=True)
    K = unique_labels.size(0)
    D = features.size(1)
    
    class_sums = torch.zeros(K, D, device=features.device, dtype=features.dtype)
    counts = torch.zeros(K, device=features.device, dtype=torch.long)
    
    labels_expanded = new_label_indices.unsqueeze(1).expand(-1, D)
    class_sums.scatter_add_(0, labels_expanded, features)
    counts.scatter_add_(0, new_label_indices, torch.ones_like(new_label_indices))
    
    visual_prototype = class_sums / counts.float().unsqueeze(1)
    
    return visual_prototype


def groupfeaturebyclass(visual_feature, text_feature, label):
    # group visual feature and concat text_feature
    unique_labels = torch.unique(label)
    feature_group = []
    for ul in unique_labels:
        mask = label == ul
        visual_group = visual_feature[mask]
        visual_text_group = torch.cat([visual_group, text_feature[ul].unsqueeze(0)])
        feature_group.append(visual_text_group)
    
    return feature_group