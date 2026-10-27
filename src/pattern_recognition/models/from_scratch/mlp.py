import copy
import time

import numpy as np
import torch
import torch.nn as nn

from pattern_recognition.config import SEED

ACTIVATIONS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}


def build_mlp(sizes, activation="relu", p_drop=0.0, init="he", const=0.01, seed=SEED):
    """sizes = [784, k_1, ..., k_{L-1}, 10]. Trả về nn.Sequential:
    Linear -> Act -> (Dropout) -> ... -> Linear (lớp ra không có hàm kích hoạt)."""
    torch.manual_seed(seed)
    layers = []
    L = len(sizes) - 1
    for l in range(L):
        lin = nn.Linear(sizes[l], sizes[l + 1])
        if init == "he":
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")   # std = sqrt(2 / k_in)
        elif init == "xavier":
            nn.init.xavier_uniform_(lin.weight)                         # U(-sqrt(6/(k_in+k_out)), +...)
        elif init == "const":
            nn.init.constant_(lin.weight, const)                        # dùng để minh hoạ tính đối xứng
        else:
            raise ValueError(init)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if l < L - 1:
            layers.append(ACTIVATIONS[activation]())
            if p_drop > 0:
                layers.append(nn.Dropout(p_drop))
    return nn.Sequential(*layers)

def n_params(model):
    """Tính tổng số lượng tham số học được (weights + biases) của mô hình."""
    return sum(p.numel() for p in model.parameters())

def forward_with_cache(model, X, train=True):
    """Lan truyền thuận có lưu cache phục vụ lan truyền ngược.
    cache[i] = (layer, a_in, a_out, extra)."""
    cache = []
    a = X
    for layer in model:
        a_in = a
        extra = None
        if isinstance(layer, nn.Dropout):
            # Cài đặt Inverted Dropout
            if train and layer.p > 0:
                # Sinh mặt nạ Bernoulli M với P(M = 1) = 1 - p
                M = torch.bernoulli(torch.full_like(a_in, 1.0 - layer.p))
                a = a_in * M / (1.0 - layer.p)
                extra = M
            else:
                a = a_in
        else:
            # Lớp Linear hoặc hàm kích hoạt thường
            a = layer(a_in)
        cache.append((layer, a_in, a, extra))
    return a, cache

def softmax_cross_entropy(logits, y):
    """logits: (N, K), y: (N,) nhãn nguyên 0..K-1.
    Trả về (loss, delta):
      loss  = trung bình cross-entropy trên batch (tensor 0 chiều).
      delta = dJ/dlogits, kích thước (N, K), đã chia cho N."""
    N = logits.shape[0]
    z = logits - logits.max(dim=1, keepdim=True).values          # trừ max chống tràn số
    log_probs = z - torch.log(torch.exp(z).sum(dim=1, keepdim=True))
    loss = -log_probs[torch.arange(N), y].mean()
    Y = torch.zeros_like(logits)
    Y[torch.arange(N), y] = 1.0                                    # mã hoá one-hot nhãn đúng
    delta = (torch.exp(log_probs) - Y) / N                         # BP1: (Y_hat - Y) / N
    return loss, delta

def linear_backward(layer, a_in, delta):
    """Trả về (dW, db, grad_in) cho một lớp nn.Linear:
      dW      = dJ/dW^(l), cùng shape với layer.weight (k_l, k_{l-1})
      db      = dJ/db^(l), cùng shape với layer.bias (k_l,)
      grad_in = dJ/dA^(l-1), kích thước (N, k_{l-1}) đẩy ngược về lớp trước."""
    dW = delta.T @ a_in             # BP4: (k_l, N) x (N, k_{l-1}) -> (k_l, k_{l-1})
    db = delta.sum(dim=0)          # BP3: cộng gradient trên trục batch N
    grad_in = delta @ layer.weight # BP2 (nửa đầu): (N, k_l) x (k_l, k_{l-1}) -> (N, k_{l-1})
    return dW, db, grad_in

def relu_backward(a_in, grad):
    """Đạo hàm ReLU: nhận đầu vào a_in (tức Z) và grad theo đầu ra."""
    return grad * (a_in > 0).float()

def sigmoid_backward(a_out, grad):
    """Đạo hàm Sigmoid: tận dụng a_out để tính a_out * (1 - a_out)."""
    return grad * a_out * (1.0 - a_out)

def tanh_backward(a_out, grad):
    """Đạo hàm Tanh: tận dụng a_out để tính (1 - a_out^2)."""
    return grad * (1.0 - a_out ** 2)

def dropout_backward(mask, p, grad):
    """Lan truyền ngược Inverted Dropout: nhân với mask / (1 - p)."""
    if mask is None:
        return grad
    return grad * mask / (1.0 - p)

def backward(cache, grad_out, weight_decay=0.0):
    """Duyệt cache từ cuối về đầu, gán .grad cho mọi tham số nn.Linear.
    Trả về dJ/dX (gradient theo đầu vào của mạng)."""
    grad = grad_out
    for layer, a_in, a_out, extra in reversed(cache):
        if isinstance(layer, nn.Linear):
            dW, db, grad = linear_backward(layer, a_in, grad)
            # Thêm số hạng L2 Regularization vào gradient của trọng số
            layer.weight.grad = dW + weight_decay * layer.weight.data
            layer.bias.grad = db
        elif isinstance(layer, nn.ReLU):
            grad = relu_backward(a_in, grad)
        elif isinstance(layer, nn.Sigmoid):
            grad = sigmoid_backward(a_out, grad)
        elif isinstance(layer, nn.Tanh):
            grad = tanh_backward(a_out, grad)
        elif isinstance(layer, nn.Dropout):
            grad = dropout_backward(extra, layer.p, grad)
    return grad

class SGDMomentum:
    """Bộ tối ưu Stochastic Gradient Descent có Momentum:
    Delta_w(t) = alpha * Delta_w(t-1) - lr * grad
    w(t+1)     = w(t) + Delta_w(t)"""
    def __init__(self, params, lr, momentum=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        # Khởi tạo vector vận tốc ban đầu bằng 0 cho từng tham số
        self.velocity = [torch.zeros_like(p.data) for p in self.params]

    def step(self):
        # Cập nhật Delta_w rồi cộng vào trọng số w (in-place)
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue
            # v = momentum * v - lr * grad
            v.mul_(self.momentum).sub_(self.lr * p.grad)
            # w = w + v
            p.data.add_(v)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

def train_one_epoch(model, X, y, optimizer, batch_size=128, weight_decay=0.0):
    """Huấn luyện mô hình qua 1 epoch theo từng mini-batch.
    Trả về mất mát cross-entropy trung bình trên toàn tập dữ liệu."""
    N = len(X)
    perm = torch.randperm(N)
    total_loss = 0.0
    for s in range(0, N, batch_size):
        batch_idx = perm[s:s + batch_size]
        Xb, yb = X[batch_idx], y[batch_idx]
        logits, cache = forward_with_cache(model, Xb, train=True)
        loss, delta = softmax_cross_entropy(logits, yb)
        backward(cache, delta, weight_decay=weight_decay)
        optimizer.step()
        total_loss += loss.item() * len(Xb)
    return total_loss / N

def evaluate(model, X, y, batch_size=5000):
    """Trả về (cross-entropy trung bình, accuracy). Dropout tắt (train=False)."""
    total, correct = 0.0, 0
    for s in range(0, len(X), batch_size):
        logits, _ = forward_with_cache(model, X[s:s + batch_size], train=False)
        loss, _ = softmax_cross_entropy(logits, y[s:s + batch_size])
        total += loss.item() * len(logits)
        correct += (logits.argmax(dim=1) == y[s:s + batch_size]).sum().item()
    return total / len(X), correct / len(X)

def train_model(model, epochs, lr, momentum=0.0, batch_size=128, weight_decay=0.0,
                X_tr=None, y_tr=None, X_va=None, y_va=None, optimizer=None,
                seed=SEED, verbose=True):
    """Điều phối vòng lặp huấn luyện qua nhiều epoch và ghi lại lịch sử."""
    if any(value is None for value in (X_tr, y_tr, X_va, y_va)):
        raise ValueError("Pass train and validation arrays explicitly.")
    torch.manual_seed(seed)
    opt = optimizer if optimizer is not None else SGDMomentum(model.parameters(), lr, momentum)
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "time"]}
    for ep in range(1, epochs + 1):
        t0 = time.time()
        train_one_epoch(model, X_tr, y_tr, opt, batch_size, weight_decay)
        dt = time.time() - t0
        tl, ta = evaluate(model, X_tr, y_tr)
        vl, va = evaluate(model, X_va, y_va)
        for k, v in zip(hist, [tl, ta, vl, va, dt]):
            hist[k].append(v)
        if verbose:
            print(f"epoch {ep:2d} | {dt:5.1f}s | train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}")
        if not np.isfinite(tl):
            if verbose:
                print("Phân kỳ (loss = nan/inf), dừng sớm.")
            break
    return hist

def layer_grad_norms(model, X, y):
    """Đo chuẩn Frobenius ||dJ/dW^(l)||_F của từng lớp Linear lúc khởi tạo."""
    logits, cache = forward_with_cache(model, X, train=False)
    _, delta = softmax_cross_entropy(logits, y)
    backward(cache, delta)
    return [layer.weight.grad.norm().item() for layer in model if isinstance(layer, nn.Linear)]

def relu_output_stds(model, X):
    _, cache = forward_with_cache(model, X, train=False)
    return [a_out.std().item() for layer, _, a_out, _ in cache if isinstance(layer, nn.ReLU)]

def train_with_early_stopping(model, max_epochs, patience, lr, momentum=0.0, batch_size=128,
                              weight_decay=0.0, X_tr=None, y_tr=None, X_va=None, y_va=None,
                              seed=SEED, verbose=False):
    """Huấn luyện có Early Stopping. Sau khi kết thúc, model mang trọng số của best_epoch."""
    if any(value is None for value in (X_tr, y_tr, X_va, y_va)):
        raise ValueError("Pass train and validation arrays explicitly.")
    torch.manual_seed(seed)
    opt = SGDMomentum(model.parameters(), lr, momentum)
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "time"]}
    
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0

    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        train_one_epoch(model, X_tr, y_tr, opt, batch_size, weight_decay)
        dt = time.time() - t0
        tl, ta = evaluate(model, X_tr, y_tr)
        vl, va = evaluate(model, X_va, y_va)
        for k, v in zip(hist, [tl, ta, vl, va, dt]):
            hist[k].append(v)
        if verbose:
            print(f"epoch {ep:2d} | {dt:5.1f}s | train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}")
        
        # Kiểm tra cải thiện của val_loss
        if vl < best_loss:
            best_loss = vl
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"Dừng sớm (Early Stopping) tại epoch {ep} do val loss không cải thiện qua {patience} epoch liên tiếp.")
                break
                
    # Khôi phục trọng số tối ưu nhất
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist, best_epoch
