import sys
import argparse
import warnings
import math
import numpy as np
from numba import cuda, NumbaPerformanceWarning



MAX_THREADS_PER_BLOCK = 1024
# Disable CUDA performance warnings
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)



# ================== SCAN CPU RELATED CODE ===================

# Find the next power of 2
def next_power(target):
  if target > 1:
    for i in range(1, int(target)):
      if (2**i >= target):
        return 2**i
  else:
    return 1

# Up-sweep algorithm implementation
def up_sweep(a):
  n = a.size
  m = round(math.log2(n))

  for d in range(0, m-1):
    for k in range(0, n-1, 2**(d+1)):
      a[k+2**(d+1)-1] += a[k+2**d-1]

  return a

# Down-sweep algorithm implementation
def down_sweep(a):
  n = a.size
  m = round(math.log2(n))
  a[n-1] = 0

  for d in range(m-1, -1, -1):
    for k in range(0, n-1, 2**(d+1)):
      t = a[k+2**d-1]
      a[k+2**d-1] = a[k+2**(d+1)-1]
      a[k+2**(d+1)-1] += t

  return a

# Prefix scan CPU implementation
def scan_cpu(array):
  n = array.size
  m = round(math.log2(n))

  # Pad with 0 to the next power of 2 if necessary
  if n != 2**m:
    pad_size = next_power(array.size)**2-array.size
    print(f"Padding size : {pad_size}")
    array = np.pad(array, (0,pad_size), 'constant', constant_values=0)
  
  # Copy the array to preserve original values
  a = np.copy(array)

  # Perform up-sweep and down-sweep phases
  up_sweep(a)
  down_sweep(a)

  # Crop the result if necessary
  if n != 2**m:
    return a[:n]

  return a

# ============================================================



# ================== SCAN GPU RELATED CODE ===================

def build_sums(array, blocks, threads_per_block):
  sums = [0]

  for i in range(0, (blocks-1)*threads_per_block, threads_per_block):
    sums += [(sums[-1] + sum(array[i:i+threads_per_block]))]

  return sums

@cuda.jit
def add_sums(sums, array):
  tid = cuda.threadIdx.x
  bdim = cuda.blockDim.x
  bid = cuda.blockIdx.x
  i = bid * bdim + tid

  array[i] += sums[bid]

@cuda.jit
def scan_kernel(a):
  shared_array = cuda.shared.array(MAX_THREADS_PER_BLOCK, dtype=np.int32)

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

def scan_gpu(array, threads_per_block):
  n = array.size
  blocks = math.ceil(n / threads_per_block)
  print(f"Number of blocks : {blocks}, number of threads per block: {threads_per_block}, total threads: {blocks*threads_per_block}")

  device_array = cuda.to_device(array)

  scan_kernel[blocks, threads_per_block](device_array)

  result = device_array.copy_to_host()
  if blocks == 1: return result
  
  sums = build_sums(array, blocks, threads_per_block)

  device_sums = cuda.to_device(sums)
  add_sums[blocks, threads_per_block](device_sums, device_array)

  result = device_array.copy_to_host()
  return result

# ============================================================



# Load and return an aray from a file (comma-separated values)
def load_array(filename):
  file = open(filename, "r")
  values = file.readline().split(",")
  file.close()
  return np.array(values, dtype=np.int32)

# Print all the array values on the standard output using a comma as a separator
def output(array):
  print(",".join(str(value) for value in array))

def main(args):
  output(load_array(args.inputFile))
  return 0



# Program entrypoint
if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    prog="project-gpu.py",
    description="Prefix scan GPU implementation"
  )
  parser.add_argument("inputFile")
  parser.add_argument("--tb")
  parser.add_argument("--cpu", action="store_true")
  parser.add_argument("--independant", action="store_true")
  parser.add_argument("--inclusive", action="store_true")

  args = parser.parse_args()
  res = main(args)

  sys.exit(res)
