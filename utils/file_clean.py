#!/usr/bin/python3
# this script deals with certain characters and words from a tab-separated field in a file
# it is meant to be used before tokenization

import re
import sys
import string
import unicodedata

# source data file
ifile = sys.argv[1]

# field to filter
filter_field = int(sys.argv[2])

# output data file
ofile = sys.argv[3]

# matches repeated characters
re_repeat = r'([\D])\1\1+'

# matches twitter mentions @username
re_mention = r'@([A-Za-z0-9_]+)'

# matches urls
re_urls = r'http(s?)://(\S+)'

# matches control characters
re_control = r'(\\[tnrfv])+'

# matches other special characters such as emoticons
re_emoticon = r'([^\w\s' + string.punctuation + r'])'


# open file and read
fin_lines = open(ifile, 'r').readlines()
fout = open(ofile, 'w')

# clean each line
for line in fin_lines:
    ind = 0
    fields = line.split('\t')

    for field in fields:
        ind += 1
        # directly output or filter a field
        if ind != filter_field:
            fout.write(field)
        else:
			# replace repeated letters with only 2 occurrences
            a = re.sub(re_repeat, r'\1\1', field)
            # replace mentions with a generic token
            b = re.sub(re_mention, r'@username ', a)
            # replace urls
            c = re.sub(re_urls, r'@url ', b)
            # replace control characters
            d = re.sub(re_control, r' ', c)
            # replace emoticons
            e = re.sub(re_emoticon, r' \1 ', d)

            fout.write(e.lower())

        # print separators
        if ind < len(fields):
            fout.write('\t')
        else:
            fout.write('')

