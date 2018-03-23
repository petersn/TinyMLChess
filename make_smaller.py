#!/usr/bin/python

import sys, zlib, multiprocessing
import numpy as np

def process(path):
	print "Opening up", path
	with open(path) as f:
		data = f.read().decode("zlib")
	array = np.fromstring(data, dtype=np.float64)
	print "Read in array of size", len(array)
	del data
	fixed = array.astype(np.int8).tostring()
	del array
	compress = zlib.compressobj(9)
	output_path = path + "-fixed.z"
	with open(output_path, "wb") as f:
		output = compress.compress(fixed)
		f.write(output)
		f.write(compress.flush())
	print "Wrote to:", output_path
	del fixed
	del compress

for path in sys.argv[1:]:
	process(path)

#p = multiprocessing.Pool(4)
#p.map(process, sys.argv[1:])

