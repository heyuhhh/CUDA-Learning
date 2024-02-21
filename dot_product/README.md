# dot product

* 思路与 block reduce 类似，这类问题本身的性质决定了 计算访存比 不高，优化空间不多

* 利用 vec4 加速访存，同时减少 warp scheduler 压力

* 最后一步仅在 warp0 内部操作即可