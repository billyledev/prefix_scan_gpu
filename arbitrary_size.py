import sys
import argparse
import math
import numpy as np
from numba import cuda

RANDOM_SEED = 123
THREADS_PER_BLOCK = 1024

np.random.seed(RANDOM_SEED) # Set the RNG seed for tests

@cuda.jit
def scan_kernel(a: np.ndarray[np.int32], n: int) -> None:
  shared_array = cuda.shared.array(THREADS_PER_BLOCK, dtype=np.int32)

  tid = cuda.threadIdx.x
  m = round(math.log2(shared_array.size))

  shared_array[tid] = a[tid] if tid < n else 0

  for d in range(0, m-1):
    cuda.syncthreads()
    k = tid*2**(d+1)
    if k <= shared_array.size-1:
      shared_array[k+2**(d+1)-1] += shared_array[k+2**d-1]

  if tid == 0: shared_array[n-1] = 0

  for d in range(m-1, -1, -1):
    cuda.syncthreads()
    k = tid*2**(d+1)
    if k < shared_array.size-1:
      t = shared_array[k+2**d-1]
      shared_array[k+2**d-1] = shared_array[k+2**(d+1)-1]
      shared_array[k+2**(d+1)-1] += t
  
  cuda.syncthreads()

  a[tid] = shared_array[tid]

def scan_gpu(array: np.ndarray[np.int32]) -> np.ndarray[np.int32]:
  n = array.size

  device_array = cuda.to_device(array)

  scan_kernel[1, THREADS_PER_BLOCK](device_array, n)
  result = device_array.copy_to_host()

  return result

def main() -> int:
  # array = np.array([2, 3, 4, 6], dtype=np.int32)
  # array = np.random.randint(-100, 100, 6, dtype=np.int32)
  array = np.array([0, 1, 2, 3, 4], dtype=np.int32)
  print(f"Input array: {array}")
  result = scan_gpu(array)
  print(f"Result array: {result}")

  return 0

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    prog="arbitrary_size.py",
    description="Arbitrary size GPU version of exclusive prefix scan implementation"
  )

  sys.exit(main())
