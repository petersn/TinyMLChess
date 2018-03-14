#!/usr/bin/python

import sys, json, multiprocessing
import chess.pgn

def process(path):
	print "Processing", path
	with open(path) as f:
		spots = []
		while True:
			spot = f.tell()
			g = chess.pgn.read_game(f)
			if g is None:
				break
			spots.append(spot)
	with open(path + ".index", "w") as f:
		json.dump(spots, f)
		f.write("\n")
	print "Wrote %i entries for %s" % (len(spots), path)

p = multiprocessing.Pool(4)
p.map(process, sys.argv[1:])

#for path in sys.argv[1:]:
#	process(path)

