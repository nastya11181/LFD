#!/usr/bin/python3
# this script combines data files from a directory in two tab-separated files
# each file is aimed at a different classification problem

import os
import sys
import string
from pathlib import Path
import xml.etree.ElementTree as ET

# source data file
idir = sys.argv[1]

# output file for the first problem
ofile1 = sys.argv[2]

# output file for the second problem
ofile2 = sys.argv[3]


# list files in the given directory
files = os.listdir(idir)

# remove special directories and files
if "features" in files:
        files.remove("features")
if "results" in files:
        files.remove("results")
if "truth.txt" in files:
        files.remove("truth.txt")
if "truth_pred.txt" in files:
        files.remove("truth_pred.txt")
if not files:
        sys.exit()


with open(ofile1, 'w') as o1:
	with open(ofile2, 'w') as o2:
		o1.write('id\ttweet\tgender\n')
		o2.write('id\ttweet\tage\n')

		# read gender and age data from "truth.txt"
		labels = {}
		if Path(idir + "/truth.txt").is_file():
			with open(idir + "/truth.txt") as f:
				lines = f.readlines()
				# labels is a dictionary in the form
                # { id : [ gender, age ] }
				labels = {l.split(":::")[0]:l.split(":::")[1:3] for l in lines}

		# read tweets from all the files in the given directory
		for f in files:
			with open(idir + "/" + f) as af:
				# parse XML document
				tree = ET.parse(af)
				root = tree.getroot()
				# extract the author id
				auth_id = root.attrib["id"].strip()
				# extract the texts of tweets and write
				for child in root:
					text = child.text.replace('\n', '').strip()
					# write out
					if labels:
					    o1.write(auth_id + '\t' + text + '\t' + labels[auth_id][0] + '\n')
					    o2.write(auth_id + '\t' + text + '\t' + labels[auth_id][1] + '\n')
					else:
					    o1.write(auth_id + '\t' + text + '\t' + 'X' + '\n')
					    o2.write(auth_id + '\t' + text + '\t' + 'XX-XX' + '\n')

