import cupy as cp

PTX_SRC = r'''
       extern "C" __global__ void hello_cuda_from_gpu() {
              printf("GPU: Hello, CUDA! (Python Version)\n");
       }
'''

if __name__ == "__main__":
    # GPU: Hello, CUDA! (Python Version)"
    hello_cuda_from_gpu = cp.RawKernel(PTX_SRC, 'hello_cuda_from_gpu')
    hello_cuda_from_gpu((2,), (8,), ())  # grid, block and arguments
    cp.cuda.runtime.deviceSynchronize()
    print()
    # CPU: Hello, CUDA! (Python Version)"
    for _ in range(16):
        print("CPU: Hello, CUDA! (Python Version)")
