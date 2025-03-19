import tensorflow as tf

# 获取 TensorFlow 的构建信息
build_info = tf.sysconfig.get_build_info()
# 检查 GPU 是否可用
print("GPU 是否可用:", tf.config.list_physical_devices('GPU'))
# 检查 CUDA 和 cuDNN 版本
#print("CUDA 版本:", build_info.get('cuda_version', '未找到'))
#print("cuDNN 版本:", build_info.get('cudnn_version', '未找到'))