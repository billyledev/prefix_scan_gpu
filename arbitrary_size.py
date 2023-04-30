import sys
import argparse
import math
import numpy as np
from numba import cuda

RANDOM_SEED = 123

np.random.seed(RANDOM_SEED) # Set the RNG seed for tests

@cuda.jit
def scan_kernel(a: np.ndarray[np.int32]) -> None:
  shared_array = cuda.shared.array(2, dtype=np.int32)

  tid = cuda.threadIdx.x
  bdim = cuda.blockDim.x
  bid = cuda.blockIdx.x
  i = bid * bdim + tid

  m = round(math.log2(shared_array.size))

  shared_array[tid] = a[i]

  for d in range(0, m-1):
    cuda.syncthreads()
    k = tid*2**(d+1)
    if k <= shared_array.size-1:
      shared_array[k+2**(d+1)-1] += shared_array[k+2**d-1]

  if tid == 0: shared_array[shared_array.size-1] = 0

  for d in range(m-1, -1, -1):
    cuda.syncthreads()
    k = tid*2**(d+1)
    if k < shared_array.size-1:
      t = shared_array[k+2**d-1]
      shared_array[k+2**d-1] = shared_array[k+2**(d+1)-1]
      shared_array[k+2**(d+1)-1] += t
  
  cuda.syncthreads()

  a[i] = shared_array[tid]

def scan_gpu(array: np.ndarray[np.int32], threads_per_block: int) -> np.ndarray[np.int32]:
  n = array.size
  blocks = math.ceil(n / threads_per_block)
  print(f"Number of blocks : {blocks}, number of threads per block: {threads_per_block}, total threads: {blocks*threads_per_block}")

  device_array = cuda.to_device(array)

  scan_kernel[blocks, threads_per_block](device_array)
  result = device_array.copy_to_host()

  return result

def main(threads_per_block: int) -> int:
  # array = np.array([2, 3, 4, 6], dtype=np.int32)
  # array = np.random.randint(-100, 100, 6, dtype=np.int32)
  array = np.array([1, 3, 4, 12, 2, 7, 0, 4], dtype=np.int32)
  print(f"Input array: {array}")
  result = scan_gpu(array, threads_per_block)
  print(f"Result array: {result}")

  return 0

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    prog="arbitrary_size.py",
    description="Arbitrary size GPU version of exclusive prefix scan implementation"
  )

  threads_per_block = 2

  sys.exit(main(threads_per_block))
