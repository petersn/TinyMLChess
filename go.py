#!/usr/bin/python

import os, glob, subprocess, multiprocessing, hashlib

def process_path(path):
	assert os.path.isfile(path)
	print "Processing:", path
	name = hashlib.sha256(os.path.basename(path)).hexdigest()[:16]
	scratch_dir = os.path.join("scratch", name)
	try: os.mkdir(scratch_dir)
	except OSError: pass
	command = ["python", "./build_dataset.py", scratch_dir, path]
	print "Executing:", command
	subprocess.check_call(command)
	print "Done writing:", scratch_dir

paths = glob.glob("data/*.pgn")
print "Running on:", paths
p = multiprocessing.Pool(4)
p.map(process_path, paths)

print
print "Completed."

