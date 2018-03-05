#!/usr/bin/python3
# this script uses the tokenized files to extract features

import re
import sys
import numpy as np
import statistics as stat
import pickle

from wordfreq import word_frequency

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.tag import pos_tag


# obtain features using a tokenized file, the language and a vectorizer object
def get_features(ftoken, lang, vec):

    # map language to lang_code
    langs = {'english':'en', 'spanish':'es', 'italian':'it', 'dutch':'nl'}

    print()
    print('Reading input ' + ftoken + '...')

    tweets = []
    with open(ftoken, 'r', encoding='utf-8') as fi:
        next(fi) # ignore header
        # Just extract the Tweet texts
        for line in fi:
            data = line.strip().split('\t')
            # removing urls
            text = re.sub(r'\shttps?://\S+', '', data[1])
            tweets.append(text)

    '''
    Get basic ngram features
    '''
    print('Getting Ngram features...')
    if vec==None:
        #vec = CountVectorizer(ngram_range=(1,3)) # withou Tfidf weighting
        print("Creating tfidf vectorizer")
        vec = TfidfVectorizer(ngram_range=(1,2))# including Tfidf weighting
        matrix = vec.fit_transform(tweets)
    else:
        print("Using existing tfidf vectorizer")
        matrix = vec.transform(tweets)


    # feat_mat is matrix of shape num_documents x num_ngram_features
    feat_mat = matrix.toarray()

    '''
    Get additional features (partly language-dependent)
    '''
    print('Getting other features...')

    POS = [] # English only
    counts_handle = []
    counts_hashtag = []
    counts_emoticon = []
    contraction = []
    word_freqs = []

    contract_forms = ['n\'t', '\'ll', '\'m', '\'s', '\'ve', '\'d', '\'re'] # to be used for English only

    print('Processing ind. tweets...')
    for t in tweets:
        if lang == 'english':
            tag_list = pos_tag(t.split()) # using (more fine-grained) penn treebank tagset

            # represent tweet as string of POS-tag of each word token
            s = ''
            for items in tag_list:
                if len(items) < 2:
                    items = ['','','']
                s = ' '.join([s, items[1]])
            POS.append(s[1:]) # exclude the first white space

        # GET HANDLE, HASHTAG AND EMOTICON
        counts_handle.append(len(re.findall(r'@username', t)))
        counts_hashtag.append(len(re.findall(r'#\w+', t)))
        counts_emoticon.append(len(re.findall(r'\s>?[:;=xX]-?[\(\)/SoOpPdD]{1,8}|\s<3', t)))

        # GET CONTRACTION COUNTS - only for English. Set to 0 for all other langs
        # GET WORD FREQUENCY
        counts = 0
        freq_list = []
        for tok in t.split():
            if lang == 'english':
            # check if contracted form
                for form in contract_forms:
                    if tok.endswith(form):
                        counts += 1
            else:
                pass # automatically keep 0 in case of all other lang

            # check freq of word token, make it more readable by *1e6, and round to 5 decimals
            freq_list.append(round(word_frequency(tok, langs[lang])*1e6, 5))

        # sort all tokens in tweet by general frequency
        freq_list = sorted(freq_list, reverse=True)
        freq_list[-1] = freq_list[-1] + 0.01 # Give least frequent token (likely with 0 frequency)
        Q1 = freq_list[:max(1, int(len(freq_list)*0.25))] # freq of most frequent 25% of words in tweet
        Q2 = freq_list[int(len(freq_list)*0.25):int(len(freq_list)*0.5)]

        # make sure Q2 and Q3 are not empty lists. Q1 and Q4 will not be unless freq_list is an empty list
        if len(Q2) == 0:
            Q2.append(0)
        Q3 = freq_list[int(len(freq_list)*0.5):int(len(freq_list)*0.75)]
        if len(Q3) == 0:
            Q3.append(0)
        Q4 = freq_list[int(len(freq_list)*0.75):]# freq of least frequent 25% of words in tweet

        # get mean freq for each quardrant, store as list of 4 values
        freqs = [stat.mean(Q1), stat.mean(Q2), stat.mean(Q3), stat.mean(Q4)]

        # store word frequency info and contraction counts
        contraction.append(counts)
        word_freqs.append(freqs)


    # TURN STRINGS OF POS INTO COUNT MATRIX - English Only
    if lang == 'english':
        print('Getting POS-features...')

        vec2 = CountVectorizer()
        pos_mat = vec2.fit_transform(POS)
        feat_mat_pos = pos_mat.toarray()


    '''
    Concatenating all features into one feature matrix
    '''
    print('Concatenating all features...')

    handels = np.array(counts_handle)
    handels = handels.reshape(len(handels),1)

    hashtags = np.array(counts_hashtag)
    hashtags = hashtags.reshape(len(hashtags),1)

    emoticons = np.array(counts_emoticon)
    emoticons = emoticons.reshape(len(emoticons),1)

    contractions = np.array(contraction)
    contractions = contractions.reshape(len(contractions),1)

    word_freq = np.array(word_freqs)

    if lang == 'english':
        all_feat_matrix = np.hstack((handels, hashtags, emoticons, feat_mat))
        #all_feat_matrix = feat_mat# ngram features only
    else:
        all_feat_matrix = np.hstack((handels, hashtags, emoticons, feat_mat))
        #all_feat_matrix = feat_mat# ngram features only

    print('Final feature matrix obtained with shape:', all_feat_matrix.shape)
    print()

    return all_feat_matrix, vec

