# layer norm

* 基于 block reduce 的写法，一般来说是一个 block 处理一行，因为多个 block 处理一行的话会设计到多个 block 之间的通信

* 使用向量化进行访存，因为要多次访问 a，所以存入共享内存中（注意避免 bank conflic）