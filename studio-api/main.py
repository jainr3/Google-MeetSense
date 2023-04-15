import os

from flask import Flask, request

import re, string

from nltk.corpus import words, wordnet 
import nltk
nltk.download('words')
nltk.download('wordnet')


# https://stackoverflow.com/questions/29381919/importerror-the-enchant-c-library-was-not-found-please-install-it-via-your-o
# !pip install pyenchant
# !sudo apt-get install libenchant1c2a
import enchant

app = Flask(__name__)


@app.route("/metrics", methods = ['POST'])
def metrics():
    json_data = request.get_json()
    text = json_data['transcript']   
  
    word_counts = {}

    a = string.punctuation.replace("'", "")

    text_tokens = re.sub(r'[0-9]', '', text)
    #text_tokens = re.sub(r'[^\w\s]', '', text_tokens)
    text_tokens = re.sub(r'[{}]'.format(a), '', text_tokens)
    text_tokens = text_tokens.strip().split()

    for word in text_tokens:
        lower_word = word.lower()
        if lower_word not in word_counts:
            word_counts[lower_word] = 0
        word_counts[lower_word] += 1

    # start filler word code
    filler_word_list=["um","huh","oh","er","ah","uh","very","really","highly","like","so","and","but","ok","okay","well","now","literally","definitely","actually","basically","totally","seriously","right","just"]
    filler_two_words_list = ["i guess", "i mean", "you know", "i suppose"]

    filler_word_counts = {k:0 for k in filler_word_list}
    for w in filler_word_list:
        try:
            #print(w, word_counts[w])
            filler_word_counts[w] = word_counts[w]
        except:
            pass

    filler_two_words_counts = {k:0 for k in filler_two_words_list}
    for i, t in enumerate(text_tokens):
        try:
            two_t = t + " " + text_tokens[i+1]
            if two_t in filler_two_words_list:
                #print(i, two_t)
                filler_two_words_counts[two_t] += 1
        except:
            pass
        
    overall_filler_words_counts = {**filler_word_counts, **filler_two_words_counts}
    overall_filler_words_counts = dict(sorted(overall_filler_words_counts.items(), key=lambda x: x[1], reverse=True))

    # Start jargon code
    vocabulary = words.words() + list(wordnet.words())
  
    d = enchant.Dict("en_US")

    jargon_counts = {}
    for i, t in enumerate(text_tokens):
        # use a combo of the two instead of just NLTK
        if not d.check(t) and not d.check(t[0].upper() + t[1:]) and t not in vocabulary:
            #print(t)
            if t not in jargon_counts:
                jargon_counts[t] = 0
            jargon_counts[t] += 1

    jargon_counts = dict(sorted(jargon_counts.items(), key=lambda x: x[1], reverse=True))

    return {'filler_words': overall_filler_words_counts, 'jargon': jargon_counts}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))