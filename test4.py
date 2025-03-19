import dataset
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
m = 100
xs, ys = dataset.get_beans_2(m)

# 绘制二维图像
plt.title("Fun", fontsize=12)
plt.xlabel("Size")
plt.ylabel("Toxicity")
plt.xlim(0, 1)
plt.ylim(0, 1.5)
plt.scatter(xs, ys)

w = 0.1
b = 0.1
y_pre = w * xs + b
plt.plot(xs, y_pre)
plt.show()

# 创建三维图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 正确创建 3D 坐标系

# 设置 z 轴范围
ax.set_zlim(0, 2)

# 生成 w 和 b 的网格数据
ws = np.arange(-1, 2, 0.1)
bs = np.arange(-2, 2, 0.1)
ws, bs = np.meshgrid(ws, bs)  # 生成网格数据

# 计算误差
es = np.zeros_like(ws)  # 初始化误差数组
for i in range(ws.shape[0]):
    for j in range(ws.shape[1]):
        y_pre = ws[i, j] * xs + bs[i, j]
        e = np.sum((ys - y_pre) ** 2) * (1 / m)
        es[i, j] = e

# 绘制三维曲面
ax.plot_surface(ws, bs, es, cmap='viridis')

# 设置标签
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Error')

# 显示图像
plt.show()