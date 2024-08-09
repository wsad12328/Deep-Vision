import torch

# 创建一个形状为 (2, 3, 4) 的张量
x = torch.arange(24).reshape(2, 3, 4)

# 使用 permute 改变维度顺序，然后再用 reshape 调整形状
y_permute = x.permute(0, 2, 1)
print(y_permute.shape)
# 直接使用 reshape 调整形状
y_reshape = x.reshape(2, 4, 3)

# 输出结果比较
print("Original tensor:")
print(x)
print("\nAfter permute and reshape:")
print(y_permute)
print("\nAfter direct reshape:")
print(y_reshape)

# 检查两个结果是否相等
are_equal = torch.equal(y_permute, y_reshape)
print("\nAre the outputs equal?", are_equal)
