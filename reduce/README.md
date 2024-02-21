# reduce 规约

* 主要思路是基于 warp reduce，也就是以线程束为基本粒度进行操作，通过线程束洗牌等操作实现快速的束内寄存器值访问，可以参考[这篇](https://zhuanlan.zhihu.com/p/572820783)文章

* 进行 block reduce 时，先每个 warp 内部进行 warp reduce，然后借助 smem 进行块内同步，之后再对每个 block 进行一次 warp reduce，得到 block 值之后通过原子操作添加到全局结果中

* block 数量越多时间开销越大，可以通过多阶段 reduce 来进行优化