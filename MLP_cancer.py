from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# 1加载数据
cancer = load_breast_cancer()
# 2切割数据
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
# 3标准化
mean_on_training = X_train.mean(axis=0)  # 求列均值
std_on_training = X_train.std(axis=0)  # 求列标准差
X_train_scaled = (X_train - mean_on_training) / std_on_training
X_test_scaled = (X_test - mean_on_training) / std_on_training
# 4模型训练
mlp = MLPClassifier(max_iter=1000,alpha=1, random_state=0)
mlp.fit(X_train_scaled, y_train)
#5准确率
score_train = mlp.score(X_train_scaled, y_train)
score_test = mlp.score(X_test_scaled, y_test)
print(score_train, score_test)

#6隐层权重热图
plt.figure(figsize=(20,5))
plt.imshow(mlp.coefs_[0],interpolation='none',cmap='viridis')
plt.yticks(range(30),cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()
plt.show()
