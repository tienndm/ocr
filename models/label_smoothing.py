import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, ignore_index=-100):
        """
        Args:
            classes (int): Số lượng lớp (ở đây là kích thước vocabulary).
            smoothing (float): Hệ số smoothing (ví dụ: 0.1).
            ignore_index (int): Chỉ số của token cần bỏ qua (ví dụ: <PAD>).
        """
        super(LabelSmoothingLoss, self).__init__()
        self.classes = classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, target):
        """
        x: Tensor có shape [N, C] (sau khi view(-1, vocab_size))
        target: Tensor có shape [N] chứa các label.
        """
        # Tính log-probabilities
        log_probs = self.log_softmax(x)
        
        with torch.no_grad():
            # Tạo một phân phối target với label smoothing
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            
            # Đánh dấu các vị trí cần bỏ qua (<PAD>)
            ignore_mask = target == self.ignore_index
            target = target.clone()
            target[ignore_mask] = 0  # thay đổi tạm thời để tránh lỗi scatter_
            
            # Gán xác suất cao nhất cho label đúng
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            # Ở các vị trí cần bỏ qua, đặt lại giá trị bằng 0
            true_dist.masked_fill_(ignore_mask.unsqueeze(1), 0)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))
