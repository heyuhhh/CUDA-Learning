# 矩阵转置

* 主要优化在于下面三点：

1. 通过 shared memory 进行中间缓存

2. 通过 padding 来避免板块冲突 

3. 合并读取以及写入时的访存

这篇文章写得很好: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc

* 可以通过vec4读取、thread coarse 等方法进行优化