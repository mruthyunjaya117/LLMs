from transformers import GPT2Tokenizer, GPT2LMHeadModel,GPT2Model
from fastapi import FastAPI, Form, UploadFile, File

from docx import Document
#from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForMaskedLM
import uvicorn
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
nltk.download('punkt')
from pymongo import MongoClient
from docx import Document
from transformers import pipeline
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel
import requests
import json
from bs4 import BeautifulSoup
from pydantic import BaseModel

class UserPrompt(BaseModel):
    prompt: str

app = FastAPI()

final_sentences = []

dgpt2_tokenizer = GPT2Tokenizer.from_pretrained(r'C:\Users\local\Downloads\language_model\question_generation\distilgpt2')
dgpt2_model = GPT2Model.from_pretrained(r'C:\Users\local\Downloads\language_model\question_generation\distilgpt2')
dgpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

gpt2_tokenizer = GPT2Tokenizer.from_pretrained(r'C:\Users\local\Downloads\language_model\semantic_search\gpt2')
gpt2_model = GPT2Model.from_pretrained(r'C:\Users\local\Downloads\language_model\semantic_search\gpt2')
gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# For generating text
gpt2_LM_model = GPT2LMHeadModel.from_pretrained(r'C:\Users\local\Downloads\language_model\semantic_search\gpt2')

roberta_model = RobertaModel.from_pretrained(r'C:\Users\local\Downloads\language_model\semantic_search\roberta-base')
roberta_tokenizer = RobertaTokenizer.from_pretrained(r'C:\Users\local\Downloads\language_model\semantic_search\roberta-base')

client = MongoClient("mongodb://username:password@server:27017/?authSource=admin")

db = client['search']

collection = db['input_data']
collection2 = db['embeddings_gpt2']
collection1 = db['embeddings_roberta']


@app.post("/upload",tags=["Upload the document"],summary="Upload the document")
def create_upload_file(file: UploadFile = File(...)):
    doc = Document(file.file._file)
    data = ""
    for para in doc.paragraphs:
        data += para.text

    document = {
        'name': file.filename,
        'content': data
    }
    # MongoDB file data insertion
    collection.insert_one(document)

    calculate_and_insert_embeddings(document)
    calculate_and_insert_embeddings_roberta(document)

    return {"message": "Document uploaded successfully"}

################### VECTOR EMBEDDINGS WITH GPT2 ####################################
def calculate_and_insert_embeddings(document):
    vector_store = []
    sentences = nltk.sent_tokenize(document['content'])
    final_sentences = [i for i in sentences if len(i) > 30]

    with torch.no_grad():
        for sentence in final_sentences:
            inputs = gpt2_tokenizer.encode(sentence, return_tensors='pt', max_length=512, truncation=True, padding=True)
            outputs = gpt2_model(inputs)[0]
            sentence_embedding = outputs[:, 0, :].squeeze().numpy()
            print("length of sentence embeddings: ", len(sentence_embedding))
            vector_store.append(sentence_embedding.tolist())

            data1 = {'filename': document['name'],
                        'sentence':sentence,
                    'embeddings': sentence_embedding.tolist()}
            
            collection1.insert_one(data1)

######################## VECTOR EMBEDDINGS WITH ROBERTA #########################
def calculate_and_insert_embeddings_roberta(document):
    vector_store = []
    sentences = nltk.sent_tokenize(document['content'])
    final_sentences = [i for i in sentences if len(i) > 30]

    with torch.no_grad():
        for sentence in final_sentences:
            encoded_input = roberta_tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = roberta_model(**encoded_input)
                sentence_embeddings = model_output.last_hidden_state.mean(dim=1)

            # inputs = roberta_tokenizer.encode(sentence, return_tensors='pt', max_length=512, truncation=True, padding=True)
            # outputs = roberta_model(inputs)[0]
            # sentence_embedding = outputs[:, 0, :].squeeze().numpy()
            # print("length of sentence embeddings: ", len(sentence_embedding))
            # vector_store.append(sentence_embedding.tolist())

            data1 = {'filename': document['name'],
                        'sentence':sentence,
                    'embeddings': sentence_embeddings[0].tolist()}
            
            collection2.insert_one(data1)


def generate_text(context):
    max_length = 200
    input_ids = gpt2_tokenizer.encode(context, return_tensors='pt', max_length=max_length, truncation=True)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    output = gpt2_LM_model.generate(input_ids, 
                                  max_length=max_length, 
                                  num_return_sequences=1,
                                  repetition_penalty=1.2,
                                  temperature=0.2,
                                  no_repeat_ngram_size=3,
                                  attention_mask=attention_mask
                                  )
    generated_text = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def cosine_similarity_vectors(vector1, vectors2):
    vectors2 = np.array(vectors2)
    dot_product = np.dot(vector1, vectors2.T)
    norms = np.linalg.norm(vector1) * np.linalg.norm(vectors2, axis=1)
    similarity_scores = dot_product / norms
    return similarity_scores



def find_matched_sentences(prompt_embedding, user_prompt):
    
    userEmbedding = prompt_embedding
    
    result = collection1.aggregate([
    # Project the cosine similarity calculation fields
    {
        '$project': {
        'embeddings': 1,  # The vector embedding field in the database
        'dotProduct': {
            '$reduce': {
            'input': { '$range': [0, {"$size": "$embeddings"}] },
            'initialValue': 0,
            'in': { '$add': ['$$value', { '$multiply': [{ '$arrayElemAt': ['$embeddings', '$$this'] }, { '$arrayElemAt': [userEmbedding, '$$this'] }] }] }
            }
        },
        'magnitudeDb': {
            '$sqrt': {
            '$reduce': {
                'input': { '$map': { 'input': '$embeddings', 'in': { '$pow': ['$$this', 2] } } },
                'initialValue': 0,
                'in': { '$add': ['$$value', '$$this'] }
            }
            }
        },
        'magnitudeUser': {
            '$sqrt': {
            '$reduce': {
                'input': { '$map': { 'input': userEmbedding, 'in': { '$pow': ['$$this', 2] } } },
                'initialValue': 0,
                'in': { '$add': ['$$value', '$$this'] }
            }
            }
        }
        }
    },
    # Calculate cosine similarity
    {
        '$addFields': {
        'similarity': { '$divide': ['$dotProduct', { '$multiply': ['$magnitudeDb', '$magnitudeUser'] }] }
        }
    },
    {
        '$sort': { 'similarity': -1 }
    },
    {
        '$limit':5
    },
    # Group by document _id and return similarity scores
    {
        '$group': {
        '_id': '$_id',
        'dotProduct': { '$first': '$dotProduct' },
        'magnitudeDb': { '$first': '$magnitudeDb' },
        'magnitudeUser': { '$first': '$magnitudeUser' },
        'similarityScores': { '$push': '$similarity' }
        }
    }
    ])

    ids = []
    for doc in result:
        print('Dot Product:', doc['dotProduct'])
        print('Magnitude DB:', doc['magnitudeDb'])
        print('Magnitude User:', doc['magnitudeUser'])
        print('Similarity Scores:', doc['similarityScores'])
        print('_id:', doc['_id'])
        print("**************************************************")
        ids.append(doc['_id'])


    # Find the documents based on the IDs
    documents = collection1.find({'_id': {'$in': ids}},{'sentence':1,'_id':0})

    # Iterate over the retrieved documents
    matched_sentences = []
    for document in documents:
        # Process each document as needed
        print(document)
        matched_sentences.append(document['sentence'])

    searched_sentences = ' '.join(matched_sentences)

    searched_sentences = user_prompt+searched_sentences

    return searched_sentences

    # Finding the summary/context of the documents
def summarizer(input_data: str):
    summary_model_path = r'C:\Users\local\Downloads\language_model\experiments\distilbart_summarizer'
    summary_tokenizer =  BartTokenizer.from_pretrained(summary_model_path)
    summary_model = BartForConditionalGeneration.from_pretrained(summary_model_path)
    summarizer = pipeline("summarization", model=summary_model, tokenizer=summary_tokenizer)
    max_l = 100
    summary = summarizer(input_data, max_length=max_l, min_length=20, do_sample=False)
    summary = summary[0]['summary_text']
    print("The summarized text ------>", summary )

    return summary

@app.post('/search',summary="Perform sematic search and genetate text the document with RoBERTa",tags=["Search and generate text"])
def perform_search(prompt: str):
   
    user_prompt = prompt
    print("user prompt-------------->: ",user_prompt)
    encoded_input = roberta_tokenizer(user_prompt, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = roberta_model(**encoded_input)
        prompt_embedding = model_output.last_hidden_state.mean(dim=1)
    
    prompt_embedding = prompt_embedding[0].tolist()
    
    semantic_sentences = find_matched_sentences(prompt_embedding, user_prompt)

    context = summarizer(semantic_sentences)
    
    generated_text = generate_text(context)

    print("The generated text--->", generated_text)

    return generated_text

@app.post('/search1',summary="Perform sematic search and genetate text the document with GPT2",tags=["Search and generate text"])
def perform_search(prompt: str):
   
    user_prompt = prompt
    print("user prompt-------------->: ",user_prompt)
    prompt_input = gpt2_tokenizer.encode(user_prompt, return_tensors='pt')
    prompt_output = gpt2_model(prompt_input)[0]
    #prompt_embedding = torch.mean(prompt_output, dim=1).squeeze().numpy()
    prompt_embedding = prompt_output[:, 0, :].squeeze().detach().numpy()
    
    context = find_matched_sentences(prompt_embedding, user_prompt)
    
    generated_text = generate_text(context)

    print("The generated text--->", generated_text)

    return generated_text

def has_tags(text1: str):
    soup = BeautifulSoup(text1, 'html.parser')
    tags = soup.find_all()
    return len(tags) > 0

@app.post('/text_gen_unibot', summary="Generates text from the user's response from Unibot", tags=["Generate text from Unibot response"])
def unibot_response(input_prompt: UserPrompt):

    user_prompt = input_prompt.prompt

    # Specify the URL of the API endpoint you want to call
    url = 'http://server:7000/GBot/MainTrainController/GetBotAvailableFAQ'

    # Define the headers
    headers = {
        'Content-Type': 'application/json',
        'USER': 'local'
    }

    # Define the request body as a dictionary
    body = {
        "BotId": "6f88f5cc-f5fb-40af-abc0-b74e82962e34",
        "Question": user_prompt,
        "FAQDataLimit": 2
    }

    # Convert the request body to JSON format
    json_body = json.dumps(body)

    # Make a POST request with headers and body
    response = requests.post(url, headers=headers, data=json_body)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Access the response data
        data = response.json()
        # Process the data as needed
        #print(data)
    else:
        # Print the error status code
        print('Error:', response.status_code)

    data1 = []
        
    for itr in data:
        if has_tags(itr['Answer']):
            soup = BeautifulSoup(itr['Answer'], 'html.parser')
            text_info = soup.get_text()
            data1.append(text_info)
        else:
            data1.append(itr['Answer'])

    unibot_responses = ' '.join(data1)
    print("The Unibot responses ---->",unibot_responses)

    context = summarizer(unibot_responses)

    generated_text_info = generate_text(context)

    return generated_text_info
    
if __name__ == '__main__':
    uvicorn.run("semantic_search:app", host='0.0.0.0', port=5000, reload=True)

