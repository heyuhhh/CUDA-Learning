# scan

* 实现 one pass 的 inclusive scan 算法，可以处理多块之间的同步，exclusive scan 是类似的，就是有一个偏移量

* 首先对 block 求 inclusive scan，采用的是 koggle stone 算法，每次会有一个stride，好处是写法简单，如果并行较充分的话时间复杂度也会低一点，缺点就是会有冗余计算，并行性不高的话就有点浪费资源。有另外一个算法 Brent Kung，减少了冗余，但是会有两阶段的计算，在并行性足够高时会有两倍常数。具体原理和实现参考：https://github.com/heyuhhh/Programming-Massively-Parallel-Processors-4th/blob/master/Ch11%20-%20Prefix%20sum%20(scan)%20/scan_kernel.cu

* 这里对 koggle stone 进行了改进，使用 warp level shuffle 来进行规约计算，类似于 block reduce，只是这里需要对 lane 做判断

* 后续使用 domino style 进行 sum 在块之间的传递，通过 flag 来进行标记，有个技巧是对 bid 进行重排，动态分配值，这样更有可能减少等待时间

* 实现的算法还能优化，就是最后传递值时，如果前面一个线程没有处理好，则会一直等待，实际上这一步可以优化，可以继续往回看，如果前面的前缀和计算好了那就ok，否则我们退一步，要求他block sum计算好了就行，再退一步才等待。这样用一点多余的计算量（等着也是等着）来实现更高效的算法