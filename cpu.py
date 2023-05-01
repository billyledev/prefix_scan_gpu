import sys
import argparse
import math
import numpy as np

RANDOM_SEED = 123

np.random.seed(RANDOM_SEED) # Set the RNG seed for tests

# Find the next power of 2
def next_power(target):
  if target > 1:
    for i in range(1, int(target)):
      if (2**i >= target):
        return 2**i
  else:
    return 1

# Up-sweep algorithm implementation
def up_sweep(a: np.ndarray[np.int32], verbose: bool) -> np.ndarray[np.int32]:
  n = a.size
  m = round(math.log2(n))

  for d in range(0, m-1):
    for k in range(0, n-1, 2**(d+1)):
      a[k+2**(d+1)-1] += a[k+2**d-1]
      if verbose: print(a)

  return a

# Down-sweep algorithm implementation
def down_sweep(a: np.ndarray[np.int32], verbose: bool) -> np.ndarray[np.int32]:
  n = a.size
  m = round(math.log2(n))
  a[n-1] = 0

  for d in range(m-1, -1, -1):
    for k in range(0, n-1, 2**(d+1)):
      t = a[k+2**d-1]
      a[k+2**d-1] = a[k+2**(d+1)-1]
      if verbose: print(a)
      a[k+2**(d+1)-1] += t
      if verbose: print(a)

  return a

# Prefix scan CPU implementation
def scan_cpu(array: np.ndarray[np.int32], verbose: bool, inclusive: bool) -> np.ndarray[np.int32]:
  n = array.size
  m = round(math.log2(n))

  # Pad with 0 to the next power of 2 if necessary
  if n != 2**m:
    pad_size = next_power(array.size)**2-array.size
    if verbose: print(f"Padding size : {pad_size}")
    array = np.pad(array, (0, pad_size), 'constant', constant_values=0)
  
  # Copy the array to preserve original values
  a = np.copy(array)

  # Perform up-sweep and down-sweep phases
  if verbose: print("Starting up-sweep phase")
  up_sweep(a, verbose)
  if verbose: print("Up-sweep phase ended, starting down-sweep phase")
  down_sweep(a, verbose)
  if verbose: print("Down-sweep phase ended")

  # Inclusive mode
  if inclusive:
    a = a[1:]
    a = np.append(a, a[-1] + array[-1])

  # Crop the result if necessary
  if n != 2**m:
    a = a[:n]

  return a

def main(args) -> int:
  array = np.array([0, 1, 2, 3, 4], dtype=np.int32)
  print(f"Input array: {array}")
  result = scan_cpu(array, args.verbose, args.inclusive)
  print(f"Result array: {result}")

  return 0

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    prog="cpu.py",
    description="CPU version of exclusive prefix scan implementation"
  )
  parser.add_argument("-v", "--verbose", action="store_true")
  parser.add_argument("--inclusive", action="store_true")
  args = parser.parse_args()

  sys.exit(main(args))
