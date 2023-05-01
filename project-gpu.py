import sys
import argparse
import numpy as np

# Print all the array values on the standard output using a comma as a separator
def output(array):
  print(",".join(str(value) for value in array))

def main(args):
  output(np.random.randint(-100, 100, 10, dtype=np.int32))

  return 0

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    prog="project-gpu.py",
    description="Prefix scan GPU implementation"
  )
  parser.add_argument('inputFile')
  parser.add_argument("--tb")
  parser.add_argument("--independant", action="store_true")
  parser.add_argument("--inclusive", action="store_true")

  args = parser.parse_args()
  res = main(args)

  sys.exit(res)
