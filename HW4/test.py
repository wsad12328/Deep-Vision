import torch

# 假設 x 和 W 是 PyTorch 的 tensor
# x 是大小為 NxD 的 tensor
# W 是大小為 DxF 的 tensor
N = 5
D = 4 
F = 2
x = torch.randn(N, D, requires_grad=False)  # 假設 x 是隨機生成的
W = torch.randn(D, F, requires_grad=True)   # 假設 W 是隨機生成的，並且需要計算其梯度

out = torch.matmul(x, W)  # 計算 out = x * W

# 使用 PyTorch 的自動微分計算 d(out)/d(W)
out.sum().backward()  # 這將計算 dout/dW

# 獲取計算出的梯度值
dout_dW = W.grad
print(x[:,0].sum())
print(x[:,1].sum())
print(x[:,2].sum())
print(x[:,3].sum())
print("d(out)/d(W):", dout_dW)
