#!/bin/bash
# this is a general script to run this project


################  VARIABLES AND DIRECTORY LOCATIONS  ################

# root directory (where this script is located)
ROOTDIR=$PWD

# train directory
DATA_TRAIN=$1

# test directory
DATA_TEST=$2

# data subsets combined
DATA=("$DATA_TRAIN" "$DATA_TEST")

# data subset names
DATANAMES=("train" "test")

# name of each classification subtask (allowed: gender, age)
TASKS=("gender" "age")

# languages to fit and test (allowed: english, spanish, italian, dutch)
LANGS=("italian" "dutch")

# perform svm parameter search (allowed: 0, 1)
OPTIMIZE=0

# tools directory
TOOLSDIR=$ROOTDIR/tools

# utils directory
UTILSDIR=$PWD/utils

# stopwords directory
STOPWORDSDIR="$UTILSDIR/stopwords"

# CMU Tweet NLP directory
TWEETNLPDIR="$TOOLSDIR/ark-tweet-nlp-0.3.2"

#####################################################################


# set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'pipeline error', -x 'print'
set -e
set -u
set -o pipefail
#set -x

mkdir -p $TOOLSDIR
mkdir -p $UTILSDIR

# ensure script runs from the root directory
if ! [ -x "$PWD/run.sh" ]; then
    echo '[INFO] You must run setup.sh from inside its directory'
    exit 1
fi

# prepare CMU Tweet NLP if mising
echo '[INFO] Preparing CMU Tweet NLP...'
if [ ! -d "$TWEETNLPDIR" ]; then
    rm -rf $TWEETNLPDIR
    pushd $TOOLSDIR > /dev/null
    wget -q "https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/ark-tweet-nlp/ark-tweet-nlp-0.3.2.tgz"
    tar zxf "ark-tweet-nlp-0.3.2.tgz"
    rm -f "ark-tweet-nlp-0.3.2.tgz"
    popd > /dev/null
fi
echo '[INFO] CMU Tweet NLP setup finished'
echo ''


# iterate over data subsets
for (( i=0; i<${#DATA[@]}; i++ )); do
    d=${DATA[$i]}
    dn=${DATANAMES[$i]}

    # iterate over languages
    for l in ${LANGS[@]}; do
        rm -f "${DATA[0]}/$l/truth_pred.txt"
        rm -f "${DATA[1]}/$l/truth_pred.txt"

		    # write data into single files
		    FEATSDIR="$d/$l/features"
		    rm -rf $FEATSDIR
		    mkdir -p $FEATSDIR
		    python3 ${UTILSDIR}/dir_extract.py "$d/$l" "$FEATSDIR/$dn-$l-${TASKS[0]}.txt" "$FEATSDIR/$dn-$l-${TASKS[1]}.txt"

		    # for each classification problem
		    for t in ${TASKS[@]}; do
			      if [ -f "$FEATSDIR/$dn-$l-$t.txt" ]; then

				        echo "[INFO] Tokenizing $l $dn $t files..."

				        # clean file
				        python3 ${UTILSDIR}/file_clean.py "$FEATSDIR/$dn-$l-$t.txt" 2 "$FEATSDIR/$dn-$l-$t.cln"

				        # tokenize file
				        pushd ${TWEETNLPDIR} > /dev/null
				        . ${UTILSDIR}/file_tokenize.sh "$FEATSDIR/$dn-$l-$t.cln" 2 "$FEATSDIR/$dn-$l-$t.cmu"
				        popd > /dev/null

				        # remove stopwords
				        python3 ${UTILSDIR}/file_stopwords.py "$FEATSDIR/$dn-$l-$t.cmu" $STOPWORDSDIR/"stopwords_${l}.txt" 2 "$FEATSDIR/$dn-$l-$t.tok"
            fi

            # remove intermediate files
            rm -f "$FEATSDIR/$dn-$l-$t.cln"
            rm -f "$FEATSDIR/$dn-$l-$t.cmu"
		    done

	  done
done


# iterate over languages
for l in ${LANGS[@]}; do

    d0=${DATA[0]}
    d1=${DATA[1]}
    dn0=${DATANAMES[0]}
    dn1=${DATANAMES[1]}

    rm -rf "$d0/$l/results"
    rm -rf "$d1/$l/results"
    mkdir -p "$d0/$l/results"
    mkdir -p "$d1/$l/results"

    # for each classification problem
    for t in ${TASKS[@]}; do

        # optimize/fit an svm model and use it for prediction
        python3 ${ROOTDIR}/fit.py $l $t \
                "$d0/$l/features/$dn0-$l-$t.txt" \
                "$d1/$l/features/$dn1-$l-$t.txt" \
                "$d0/$l/features/$dn0-$l-$t.tok" \
                "$d1/$l/features/$dn1-$l-$t.tok" \
                $OPTIMIZE \
                "$d0/$l/results/$dn0-$l-$t.pred" \
                "$d1/$l/results/$dn1-$l-$t.pred" \
                "$d0/$l/truth_pred.txt" \
                "$d1/$l/truth_pred.txt"

        # evaluate prediction results
        python3 ${ROOTDIR}/eval.py "$d0/$l/features/$dn0-$l-$t.txt" $dn0 $t "$d0/$l/features/$dn0-$l-$t.txt" "$d0/$l/results/$dn0-$l-$t.pred"
        python3 ${ROOTDIR}/eval.py "$d1/$l/features/$dn1-$l-$t.txt" $dn1 $t "$d0/$l/features/$dn0-$l-$t.txt" "$d1/$l/results/$dn1-$l-$t.pred"
    done

done

