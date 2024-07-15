import numpy as np

# 定義形狀相同的數組
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[6, 5, 4], [3, 2, 1]])

# 計算 L1 範數
l1_norm_A_minus_B = np.sum(np.abs(A - B))
l1_norm_B_minus_A = np.sum(np.abs(B - A))

print("L1 norm of A - B:", l1_norm_A_minus_B)  # 結果應該相同
print("L1 norm of B - A:", l1_norm_B_minus_A)  # 結果應該相同
