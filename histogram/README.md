# histogram

* 最直接的思路每个线程取数加到全局 hist 中，但由于数据竞争，需要原子操作，效率就比较低

* 考虑的一个改进方案是先加入到 shared memory 中，然后再合并到全局内存中去

* 进一步地，代码中的方法是先每个 warp 统计直方图信息，这样只会 warp 直接存在竞争，然后 warp hist 中对于每个 bin 可以通过 reduce 操作合并到 block hist 中。之后再通过一个kernel统计 block hist 到最终的 global hist，思路与上述合并类似

* 可以通过向量化访存来优化读取