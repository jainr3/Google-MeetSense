import os

from flask import Flask, request, Response
from transformers import T5Tokenizer, T5ForConditionalGeneration

from torch.utils.data import DataLoader
import re, string
import pandas as pd

from model_code import *

from nltk.corpus import words, wordnet 
import nltk
nltk.download('words')
nltk.download('wordnet')
vocabulary = words.words() + list(wordnet.words())

# https://stackoverflow.com/questions/29381919/importerror-the-enchant-c-library-was-not-found-please-install-it-via-your-o
# !pip install pyenchant
# !sudo apt-get install libenchant1c2a
import enchant

app = Flask(__name__)

import whisper

import torch
import datasets

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

#device = "cuda" if torch.cuda.is_available() else "cpu"
#summarization_model = summarization_model.to(device)

@app.route("/isalive")
def is_alive():
    print("/isalive request")
    status_code = Response(status=200)
    return status_code

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

    return {'filler_words': overall_filler_words_counts, 'jargon': jargon_counts, "filler_words_total": sum(overall_filler_words_counts.values()), "jargon_total": sum(jargon_counts.values())}

@app.route("/transcribe", methods = ["POST"])
def transcribe():
    whisper_model = whisper.load_model("base")
    audio_file = request.files['audio_file']
    audio_file.save("temp.mp3")
    result = whisper_model.transcribe("temp.mp3")

    return result["text"]

@app.route("/summarize", methods= ["POST"])
def summarize():
    json_data = request.get_json()
    my_input_text = json_data['transcript']

    model_name = "t5-base"

    summarization_model = T5ForConditionalGeneration.from_pretrained(model_name)
    summarization_model.load_state_dict(torch.load("./model/t5-model-v3.pth", map_location=torch.device('cpu')))
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    my_data = pd.DataFrame(columns=['ctext', 'text'])

    start = 0
    input_text = []
    while start < len(my_input_text):
        input, start = get_input_sentences(start, my_input_text)
        input_text.append(input)
        #print("input text \n" + input)

    for i, input in enumerate(input_text):
        my_data.loc[i] = ["summarize: " + input, input]

    my_data = CustomDataset(my_data, tokenizer, 512, 50)

    val_params = {
        'batch_size': 4,
        'shuffle': False,
        'num_workers': 0
    }

    my_data_loader = DataLoader(my_data, **val_params)

    predictions, actuals = inference(0, tokenizer, summarization_model, 'cpu', my_data_loader)
    final_df = pd.DataFrame({'Generated Text':predictions,'Actual Text':actuals})

    summary = ""
    for i, x in enumerate(final_df["Generated Text"]):
        x = x[0].upper() + x[1:]
        if x[0] != " " and i != 0:
            summary = summary + " " + x
        else:
            summary += x

    #summary = ' '.join(final_df["Generated Text"].astype(str))

    return summary

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))