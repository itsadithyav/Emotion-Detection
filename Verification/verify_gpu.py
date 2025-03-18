import tensorflow as tf

#Check is GPU is available
print(tf.config.list_physical_devices('GPU'))

#Check if TensorFlow is built with CUDA
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))

#Check if TensorFlow is built with GPU support
print(tf.sysconfig.get_build_info())