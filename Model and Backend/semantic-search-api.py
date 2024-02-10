from fastapi import FastAPI, Response, status
from fastapi.responses import JSONResponse
import json
from annoy import AnnoyIndex as AI
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pickle

model = SentenceTransformer("sentence-transformers/all-roberta-large-v1")

df = pd.read_csv("Cleaned DB Test.csv")
# Importing encoded DB

with open('encoded_dict.pickle', 'rb') as f:
    combined_dict = pickle.load(f)
    # Importing a dictionary with key being the UniqueID and value being the dimensions of vector corresponding to the UniqueID

u = AI(1024)
u.load('test.ann')

app = FastAPI()

@app.get("/search-results/")
async def search_results(query: str, num : int = 50 , threshold : float = 0.5):
    '''
    Function to perform semantic search
    query: Input User string
    results: DataFrame with Name, College, Department, Area of Interest, Area of Interest Isolated (Highlighted text)
    '''

    query_vector = model.encode(query) # Encoding the input using Sci-BERT model
    ans = u.get_nns_by_vector(query_vector, num)
    # Getting the index of n-nearest-neighbours by checking the query vector in the Annoy Index, for a particular number of neighbours

    nns = []
    print(*ans)
    for i in ans: # Iterating over answer indices
        ls = combined_dict[i] # Retrieving the dimensions of the vector for given UniqueID
        sim = util.cos_sim(query_vector,ls) # Finding the cosine similarity between the user query and
        if sim[0][0] > threshold: # Checking if the cosine similarity is greater than similarity
            nns.append([i, sim[0][0]])

    ans = sorted(nns, key = lambda x:x[1], reverse = True)
    unique_ids = [i for i, _ in ans]  # Extracting unique IDs from the ans list while maintaining order
    results = df[df['uniq_id'].isin(unique_ids)].set_index('uniq_id').loc[unique_ids].reset_index()
    json_ans = results.to_json(orient='records')
    return JSONResponse(content=json.loads(json_ans))