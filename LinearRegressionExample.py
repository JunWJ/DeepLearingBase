# ==============================================
# 线性回归（单文件独立运行版）
# 后续新增文件可导入本文件的类/函数复用
# ==============================================

# ----------------------
# 1. 定义数据集（可后续迁移到单独的数据文件）
# ----------------------
x_data = [1, 2, 3]  # 特征
y_data = [2, 4, 6]  # 标签

# ----------------------
# 2. 线性回归模型类（模块化封装，方便后续复用）
# ----------------------
class LinearRegression:
    def __init__(self, init_w=4, learning_rate=0.01):
        """
        初始化模型参数
        :param init_w: 权重初始值
        :param learning_rate: 学习率
        """
        self.w = init_w  # 权重
        self.lr = learning_rate  # 学习率

    def forward(self, x):
        """前向传播（预测）"""
        return x * self.w

    def mse_loss(self, xs, ys):
        """均方误差损失函数"""
        cost_value = 0.0
        for x, y in zip(xs, ys):
            y_pred = self.forward(x)
            cost_value += (y_pred - y) ** 2
        return cost_value / len(xs)

    def compute_gradient(self, xs, ys):
        """计算梯度（批量梯度下降）"""
        grad = 0.0
        for x, y in zip(xs, ys):
            grad += 2 * x * (self.forward(x) - y)
        return grad / len(xs)

    def update_weights(self, grad):
        """更新权重"""
        self.w -= self.lr * grad

    def train(self, xs, ys, epochs=101, print_interval=10):
        """
        模型训练
        :param xs: 输入特征
        :param ys: 标签
        :param epochs: 训练轮数
        :param print_interval: 每隔多少轮打印日志
        """
        for epoch in range(epochs):
            cost = self.mse_loss(xs, ys)
            grad = self.compute_gradient(xs, ys)
            self.update_weights(grad)
            if epoch % print_interval == 0:
                print(f'Epoch {epoch:3d}: w = {self.w:.6f}, loss = {cost:.6f}')

# ----------------------
# 3. 主程序入口（关键！保证单文件可独立运行）
# ----------------------
if __name__ == "__main__":
    # 初始化模型
    model = LinearRegression(init_w=4, learning_rate=0.01)
    # 开始训练
    print("===== 开始训练线性回归模型 =====")
    model.train(xs=x_data, ys=y_data, epochs=101, print_interval=10)
    # 训练后预测示例
    print("\n===== 训练后预测示例 =====")
    test_x = 20  # 测试输入
    test_y_pred = model.forward(test_x)
    print(f'输入 x = {test_x}，预测 y = {test_y_pred:.2f}')  # 预期输出 ~8.0