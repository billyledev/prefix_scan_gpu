import sys
import argparse
import math
import numpy as np

RANDOM_SEED = 123

np.random.seed(RANDOM_SEED) # Set the RNG seed for tests

def up_sweep(a: np.ndarray[np.int32], verbose: bool) -> np.ndarray[np.int32]:
  n = a.size
  m = round(math.log2(n))

  for d in range(0, m-1):
    for k in range(0, n-1, 2**(d+1)):
      a[k+2**(d+1)-1] += a[k+2**d-1]
      if verbose: print(a)

  return a

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

def scan_cpu(array: np.ndarray[np.int32], verbose: bool) -> np.ndarray[np.int32]:
  n = array.size
  m = round(math.log2(n))

  if n != 2**m:
    array = np.pad(array, (0,8-array.size), 'constant', constant_values=0)
  
  a = np.copy(array) # Copy the array to preserve original values

  if verbose: print("Starting up-sweep phase")
  up_sweep(a, verbose)
  if verbose: print("Up-sweep phase ended, starting down-sweep phase")
  down_sweep(a, verbose)
  if verbose: print("Down-sweep phase ended")

  if n != 2**m:
    return a[:n]
  return a

def main(verbose: bool) -> int:
  # array = np.array([2, 3, 4, 6], dtype=np.int32)
  # array = np.random.randint(-100, 100, 6, dtype=np.int32)
  # array = np.array([1,2,3], dtype=np.int32)
  array = np.array([0, 1, 2, 3, 4], dtype=np.int32)
  print(f"Input array: {array}")
  result = scan_cpu(array, verbose)
  print(f"Result array: {result}")

  return 0

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    prog="cpu.py",
    description="CPU version of exclusive prefix scan implementation"
  )
  parser.add_argument("-v", "--verbose", action="store_true")
  args = parser.parse_args()

  sys.exit(main(args.verbose))
