#!/usr/bin/python3
# this script removes stopwords from a file in a certain field
# it assumes that the file contains tab-separated fields
# it assumes that the field of interest has been tokenized

import sys
import string

# source data file
ifile = sys.argv[1]

# list of stopwords
stopwords = set(open(sys.argv[2], 'r').read().split())

# field to filter
filter_field = int(sys.argv[3])

# output data file
ofile = sys.argv[4]


with open(ifile, 'r') as fin:
    with open(ofile, 'w') as fout:
        for line in fin.readlines():
            ind = 0
            fields = line.split('\t')

            for field in fields:
                ind += 1
                # directly output or filter a field
                if ind == filter_field:
                    field2 = list(map(lambda y: '' if y in stopwords else y, field.split()))
                    fout.write(' '.join([w for w in field2 if w]))
                else:
                    fout.write(field)

                # print separators
                if ind < len(fields):
                    fout.write('\t')
                else:
                    fout.write('')

