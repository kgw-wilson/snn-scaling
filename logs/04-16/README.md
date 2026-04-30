# 04-16 Experiments

The whole loop had to be wrapped in torch.compile because otherwise multiple graphs would be generated per call. This isn't ideal because although compile can fuse operations into customized OpenMP/Triton kernels, it cannot reduce the overhead from the Python for loop itself.

Other simulators tend to execute in eager mode, so adding support for torch.compile adds complexity and does not directly strengthen the comparison to other SNN simulators.

The GPU code required tensors to not be mutated in ways involving dynamic indices, so operations to zero out the ring buffer like .zero_() on a dynamically-indexed slice would cause graph breaks under torch.compile's reduce-overhead mode (which uses CUDA graphs internally). I couldn't find a workaround for this. Allocating a [num_timesteps x num_neurons] current buffer could fix the problem by eliminating dynamic indexing, but that would consume significantly more memory and add another dimension to the study.

Because the matmul is already handled by cuBLAS/cuDNN in eager mode, and because the matmul dominates with O(n^2), fusing the element-wise tensor ops shouldn't change scaling behavior.

I ended up deciding to drop torch.compile after experimenting with it.
