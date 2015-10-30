#!/usr/bin/python2
#
# Remove all features, except for the features specified.
#

import sys

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print "Usage: %s features" % sys.argv[0]
		sys.exit(1)

	features = set()
	for line in open(sys.argv[1]):
		lineParts = line.split()
		features.add(lineParts[0])
	
	for line in sys.stdin:
		lineParts = line.split()
		if len(lineParts) == 1:
			print line,
			continue

		fs = lineParts[2:]
		newFs = []

		for i in range(0, len(fs), 2):
			if fs[i] in features:
				newFs.append(fs[i])
				newFs.append(fs[i+1])

		nNewFs = len(newFs) / 2

		print "%s %d %s" % (lineParts[0], nNewFs, ' '.join(newFs))
