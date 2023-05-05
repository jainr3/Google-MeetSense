import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

from torch.utils.data import DataLoader
import re, string
import pandas as pd

from nltk.corpus import words, wordnet 
import nltk
nltk.download('words')
nltk.download('wordnet')
vocabulary = words.words() + list(wordnet.words())

# https://stackoverflow.com/questions/29381919/importerror-the-enchant-c-library-was-not-found-please-install-it-via-your-o
# !pip install pyenchant
# !sudo apt-get install libenchant1c2a
import enchant

import torch
import datasets

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

import random
import librosa
import uuid


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text
        self.ctext = self.data.ctext

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }
    

def inference(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _%100==0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals

def get_input_sentences(start_idx, my_input_text):
  # Get whole sentences up to X number of characters; don't break sentences up
  end_idx = start_idx + 2048
  text = my_input_text[start_idx:end_idx]
  end_sentence_chars = [".", "?", "!"]
  next_start_idx = end_idx
  while text[-1] not in end_sentence_chars and end_idx < len(my_input_text):
      text = text[:-1]
      next_start_idx -= 1
  #print(len(tokenizer.tokenize(text)))
  return text, next_start_idx


def summarize_local(my_input_text):
    # takes a transcript (str) and returns str

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

    return {"summary": summary}

def metrics_local(text):
    # takes a transcript 
  
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

    # return top 3 for each for easy displaying
    temp = [str(x[0]) + ": " + str(x[1]) for x in list(overall_filler_words_counts.items())[:3]]
    top_filler_words = ', '.join(temp)
    temp = [str(x[0]) + ": " + str(x[1]) for x in list(jargon_counts.items())[:3]]
    top_jargon_words = ', '.join(temp)

    return {'filler_words': overall_filler_words_counts, 'jargon': jargon_counts, "filler_words_total": sum(overall_filler_words_counts.values()), "jargon_total": sum(jargon_counts.values()),
            'top_filler_words': top_filler_words, 'top_jargon_words': top_jargon_words}

def synthetic_data_local(meeting_duration):
    possible_meeting_titles = ["Strategy Session", "Brainstorming Meeting", "Project Planning Meeting",
                               "Progress Review Meeting", "Performance Evaluation Meeting", "Team Building Meeting",
                               "Budget Review Meeting", "Sales Forecast Meeting", "Product Development Meeting",
                               "Customer Feedback Meeting", "Risk Assessment Meeting", "Crisis Management Meeting",
                               "Innovation Workshop", "Marketing Campaign Meeting", "Quality Assurance Meeting", 
                               "Talent Acquisition Meeting", "Vendor Negotiation Meeting", "Stakeholder Engagement Meeting",
                               "Quarterly Business Review Meeting", "Employee Training Meeting", "Board of Directors Meeting",
                               "Investor Relations Meeting", "Supply Chain Management Meeting", "Social Media Strategy Meeting",
                               "Partnership Development Meeting"]
    meeting_title = random.choice(possible_meeting_titles)

    if random.randint(0, 1) == 0:
        m_time_hr = random.randint(7, 10)
        meeting_time = str(m_time_hr) + ":00 AM EST" 
    else:
        m_time_hr = random.randint(1, 6)
        meeting_time = str(m_time_hr) + ":00 PM EST"

    meeting_reoccuring = random.choice(["Yes", "No"])

    # TODO: figure out what DB image urls these people will get...
    engineering = {"Olivia Johnson": ("Engineering", ""),
                   "Ethan Smith": ("Engineering", ""),
                   "Chloe Rodriguez": ("Engineering", ""),
                   "William Kim": ("Engineering", ""),
                   "Ava Patel": ("Engineering", ""),}
    product = {"James Lee": ("Product", ""), 
               "Sophia Nguyen": ("Product", ""),  
               "Benjamin Garcia": ("Product", ""), 
               "Mia Brown": ("Product", ""),  
               "Alexander Davis": ("Product", "")}
    design = {"Isabella Robinson": ("Design", ""), 
              "Samuel Wright": ("Design", ""),
              "Charlotte Hernandez": ("Design", ""),
              "Michael Cooper": ("Design", ""),
              "Amelia Green": ("Design", "")}
    management = {"John Smith": ("Management", ""),
                  "Natalie Parker": ("Management", ""),
                  "Christopher Taylor": ("Management", ""),
                  "Grace Martinez": ("Management", ""),
                  "Jonathan Scott": ("Management", ""),}
    
    # select some from each bucket... 1-3 per type
    attendees = {}
    engineering_selected = random.sample(list(engineering.keys()), k=random.randint(1, 3))
    product_selected = random.sample(list(product.keys()), k=random.randint(1, 3))
    design_selected = random.sample(list(design.keys()), k=random.randint(1, 3))
    management_selected = random.sample(list(management.keys()), k=random.randint(1, 3))
    for n in engineering_selected:
        attendees[n] = engineering[n]
    for n in product_selected:
        attendees[n] = product[n]
    for n in design_selected:
        attendees[n] = design[n]
    for n in management_selected:
        attendees[n] = management[n]

    total_participants = len(attendees)
    work_category = 3 # TODO what is this field
    types_of_roles = 4 # always at least 1 of the role

    num_late = random.randint(1, total_participants // 2) - 1
    late_str = "Late by " + str(random.randint(3, 9)) + " mins" if num_late != 0 else "On time"

    attendee_punctuality = (num_late, late_str)

    durs = [10, 15, 20, 30, 45, 60]
    durs_2 = [meeting_duration-x for x in durs]
    durs_3 = 9999
    for x in durs_2:
        if abs(x) < durs_3:
            durs_3 = x

    if meeting_duration in durs:
        meeting_duration_analysis = "Meeting on time"
    elif durs_3 > 0:
        meeting_duration_analysis = "Over by " + str(abs(durs_3)) + " mins"
    else:
        meeting_duration_analysis = "Under by " + str(abs(durs_3)) + " mins"

    # chat log
    example_chats = ["Hi everyone! Glad to see you all here.", 
                     "Hi there! Good to be here.",
                     "Hey everyone, good morning.",
                     "Hi folks!", "Hello", "Hey", "Howdy",
                     "Can you hear me?", "Sorry I'll be right back",
                     "Brb", "All good?", "Good morning"]
    chat_log = []
    m_time_min = 1
    for p in attendees.keys():
        if random.randint(0, 2) == 0:
            chat_log.append("(" + str(m_time_hr) + ":0" + str(m_time_min) + ") " + p + ": " + random.choice(example_chats))
            m_time_min += random.randint(0, 1)
            m_time_min = min(m_time_min, 9)

    # action items
    engineering_action_items = ["Conduct a code review for the new feature implementation.",
                                "Investigate the reported bug and determine the root cause of the issue."
                                "Write unit tests for the new API endpoint.",
                                "Optimize the database query to improve performance.",
                                "Update the API documentation with the latest changes.",
                                "Refactor the legacy codebase to improve maintainability.",
                                "Implement a caching layer to improve the application's performance.",
                                "Investigate the feasibility of integrating a new third-party API.",
                                "Evaluate different open-source libraries for the project's needs.",
                                "Conduct load testing to ensure the application can handle high traffic volumes."]
    design_action_items = ["Create wireframes for the new feature based on the user stories.",
                           "Conduct a usability test with the existing UI design to identify areas of improvement.",
                           "Create a style guide for the design system.",
                           "Design the visual mockup for the landing page.",
                           "Create design assets for the marketing team to use in the campaign.",
                           "Develop a new brand identity for the company.",
                           "Create user personas to better understand the target audience.",
                           "Conduct a design sprint to ideate on new product features.",
                           "Design an onboarding flow for new users.",
                           "Develop a user research plan to gather qualitative data on the product."]
    product_action_items = ["Conduct a competitive analysis to identify potential gaps in the market.",
                            "Gather customer feedback on the existing product and prioritize the feature requests.",
                            "Conduct a market research study to identify new market segments.",
                            "Develop a product roadmap for the upcoming quarter.",
                            "Create a user journey map to identify areas of improvement in the product experience.",
                            "Analyze the customer acquisition funnel and identify areas of dropoff.",
                            "Conduct a pricing analysis and recommend changes to pricing strategy.",
                            "Create a go-to-market plan for the new feature release.",
                            "Develop a user retention plan to increase engagement and reduce churn.",
                            "Conduct user interviews to gather feedback on the new feature's usability."]

    action_items_1 = random.sample(engineering_action_items, k=random.randint(2, 4))
    action_items_2 = random.sample(design_action_items, k=random.randint(2, 4))
    action_items_3 = random.sample(product_action_items, k=random.randint(2, 4))

    action_items_1 = [" " + x if i != 0 else x for i, x in enumerate(action_items_1)]
    action_items_2 = [" " + x if i != 0 else x for i, x in enumerate(action_items_1)]
    action_items_3 = [" " + x if i != 0 else x for i, x in enumerate(action_items_1)]
    
    synthetic_data_dict = {"meeting_title": meeting_title, 
                           "meeting_time": meeting_time,
                           "meeting_reoccuring": meeting_reoccuring,
                           "attendees": attendees,
                           "total_participants": total_participants,
                           "work_category": work_category,
                           "types_of_roles": types_of_roles,
                           "attendee_punctuality": attendee_punctuality,
                           "meeting_duration_analysis": meeting_duration_analysis,
                           "chat_log": chat_log,
                           "engineering_action_items": action_items_1,
                           "design_action_items": action_items_2,
                           "product_action_items": action_items_3}

    return synthetic_data_dict