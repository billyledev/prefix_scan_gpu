import sys
import argparse
import numpy as np



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



if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    prog="project-gpu.py",
    description="Prefix scan GPU implementation"
  )
  parser.add_argument("inputFile")
  parser.add_argument("--tb")
  parser.add_argument("--independant", action="store_true")
  parser.add_argument("--inclusive", action="store_true")

  args = parser.parse_args()
  res = main(args)

  sys.exit(res)
