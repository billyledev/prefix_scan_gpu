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
  tid = cuda.threadIdx.x
  m = round(math.log2(n))

  for d in range(0, m-1):
    cuda.syncthreads()
    k = tid*2**(d+1)
    if k <= n-1:
      a[k+2**(d+1)-1] += a[k+2**d-1]

  if tid == 0: a[n-1] = 0

  for d in range(m-1, -1, -1):
    cuda.syncthreads()
    k = tid*2**(d+1)
    if k < n-1:
      t = a[k+2**d-1]
      a[k+2**d-1] = a[k+2**(d+1)-1]
      a[k+2**(d+1)-1] += t

def scan_gpu(array: np.ndarray[np.int32]) -> np.ndarray[np.int32]:
  n = array.size
  m = round(math.log2(n))

  if n != 2**m:
    return [0]

  device_array = cuda.to_device(array)

  scan_kernel[1, THREADS_PER_BLOCK](device_array, n)
  result = device_array.copy_to_host()

  return result

def main() -> int:
  # array = np.array([2, 3, 4, 6], dtype=np.int32)
  # array = np.array([0, 1, 2, 3, 4], dtype=np.int32)
  # array = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
  # array = np.array([i for i in range(0, 32)], dtype=np.int32)
  # array = np.array([i for i in range(0, 64)], dtype=np.int32)
  # array = np.random.randint(-100, 100, 32, dtype=np.int32)
  array = np.random.randint(-100, 100, 128, dtype=np.int32)
  print(f"Input array: {array}")
  result = scan_gpu(array)
  print(f"Result array: {result}")

  return 0

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    prog="single_thread_block.py",
    description="Single thread block GPU version of exclusive prefix scan implementation"
  )

  sys.exit(main())
