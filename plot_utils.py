import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from keras.src.models import Sequential#导入keras
from sklearn.decomposition import PCA


def show_scatter_curve(X,Y,pres):
	plt.scatter(X, Y) 
	plt.plot(X, pres) 
	plt.show()

def show_scatter(X,Y):
	if X.ndim>1:
		show_3d_scatter(X,Y)
	else:
		plt.scatter(X, Y) 
		plt.show()


def show_3d_scatter(X,Y):
	x = X[:,0]
	z = X[:,1]
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')  # 正确创建 3D 坐标系
	ax.scatter(x, z, Y)
	plt.show()

def show_surface(x,z,forward_propgation):
	x = np.arange(np.min(x),np.max(x),0.1)
	z = np.arange(np.min(z),np.max(z),0.1)
	x,z = np.meshgrid(x,z)
	y = forward_propgation(x)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')  # 正确创建 3D 坐标系
	ax.plot_surface(x, z, y, cmap='rainbow')
	plt.show()



def show_scatter_surface(X,Y,forward_propgation):
	if type(forward_propgation) == Sequential:
		show_scatter_surface_with_model(X,Y,forward_propgation)
		return
	x = X[:,0]
	z = X[:,1]
	y = Y

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')  # 正确创建 3D 坐标系
	ax.scatter(x, z, y)

	x = np.arange(np.min(x),np.max(x),0.1)
	z = np.arange(np.min(z),np.max(z),0.1)
	x,z = np.meshgrid(x,z)

	X = np.column_stack((x[0],z[0]))
	for j in range(z.shape[0]):
		if j == 0:
			continue
		X = np.vstack((X,np.column_stack((x[0],z[j]))))

	r = forward_propgation
	y = r[0]
	if type(r) == np.ndarray:
		y = r

	
	y = np.array([y])
	y = y.reshape(x.shape[0],z.shape[1])
	ax.plot_surface(x, z, y, cmap='rainbow')
	plt.show()

def show_scatter_surface_with_model(X,Y,model):
	#model.predict(X)

	x = X[:,0]
	z = X[:,1]
	y = Y

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')  # 正确创建 3D 坐标系
	ax.scatter(x, z, y)

	x = np.arange(np.min(x),np.max(x),0.1)
	z = np.arange(np.min(z),np.max(z),0.1)
	x,z = np.meshgrid(x,z)



	X = np.column_stack((x[0],z[0]))

	for j in range(z.shape[0]):
		if j == 0:
			continue
		X = np.vstack((X,np.column_stack((x[0],z[j]))))

	y = model.predict(X)
	
	# return
	# y = model.predcit(X)
	y = np.array([y])
	y = y.reshape(x.shape[0],z.shape[1])
	ax.plot_surface(x, z, y, cmap='rainbow')
	plt.show()

def pre(X,Y,model):
	model.predict(X)

def show_scatter_surface_2(X_test, Y_test, pres):
    """
    可视化模型预测结果。

    参数:
    - X_test: 测试数据，形状为 (n_samples, n_features)
    - Y_test: 真实标签，形状为 (n_samples, n_classes)
    - pres: 模型预测结果，形状为 (n_samples, n_classes)
    """
    # 将 X_test 降维到二维
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_test)

    # 将 Y_test 和 pres 转换为类别标签
    y_true = np.argmax(Y_test, axis=1)  # 真实标签
    y_pred = np.argmax(pres, axis=1)    # 预测标签

    # 创建散点图
    plt.figure(figsize=(12, 6))

    # 绘制真实标签
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_true, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='True Label')
    plt.title('True Labels')

    # 绘制预测标签
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Predicted Label')
    plt.title('Predicted Labels')

    plt.show()