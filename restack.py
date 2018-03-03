#!/usr/bin/python

import os, sys, itertools, glob, zlib, multiprocessing
import numpy as np
import build_dataset

if len(sys.argv) != 3:
	print "Usage: %s input-directory/ output-directory/" % (sys.argv[0],)
	exit(1)

input_directory  = sys.argv[1]
output_directory = sys.argv[2]

intermediate = sorted(os.listdir(input_directory))
print "Intermediates:", intermediate
final_outputs = set()
for x in intermediate:
	p = os.path.join(input_directory, x)
	leaves = os.listdir(p)
	for leaf in leaves:
		final_outputs.add(leaf)

block_size = 2**26
print "Output names:", final_outputs

def collate(output):
	print "Collating:", output
	writer = build_dataset.ZlibWriter(os.path.join(output_directory, output))
	for inter in intermediate:
		source = os.path.join(input_directory, inter, output)
		print "   ", source
		total_length = 0
		with open(source) as f:
			decomp = zlib.decompressobj()
			while True:
				block = f.read(block_size)
				if not block:
					break
				expanded_block = decomp.decompress(block)
				total_length += len(expanded_block)
				writer.write(expanded_block)
			expanded_block = decomp.flush()
			total_length += len(expanded_block)
			writer.write(expanded_block)
		print "        Wrote", total_length, "bytes."
	writer.close()

p = multiprocessing.Pool(4)
p.map(collate, final_outputs)

