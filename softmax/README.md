# softmax

* 本质上还是基于 warp reduce 做 block reduce 的事情（计算exp的和），但是有点区别是需要将和计算出来之后再当作分母

* 所以需要网格级的同步，将 exp sum 累加到全局变量中，经过同步最后用来当作分母

* 使用 __threadfence + atomic op 可以实现网格级别的同步，__threadfenc() 保证当前fence的线程内后面的读/写操作一定是在前面的写操作完成之后进行（因为全局写延时较长，同时注意该函数并不是线程同步函数），atomic op 用来计数有多少个 block 完成了写操作，当所有 block 完成写操作之后再进行最后一步除法

* 可以进一步优化的点：1. 根据 N 的大小分情况设计 kernel，可以通过 warp 对寄存器操作，block 对 smem 操作，或者直接对全局内存操作（block设置应尽可能大，利用好缓存）；2. thread coarse 以及向量化访存来减轻 warp scheduler 压力，提升访存效率（因为都是一些重复的优化内容就没继续写了）

* 注意一般的 soft max 操作会减去 max value，可以通过另外一个kernel或者类似于 除法操作 那样再做一遍 减法