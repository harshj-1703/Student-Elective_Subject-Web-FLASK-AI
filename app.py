from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from numpy.linalg import norm
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['ai_elective_subject']
collection = db['electivesubjectfeedback']

# initializing flask
app = Flask('ElectiveRecommendation')

with open('model.pickle', 'rb') as f:
    data = pickle.load(f)

courses_dataset = data['courses_dataset']
c_des = data['c_des']
model = data['model']
chain1 = data['chain1']
chain2 = data['chain2']
chain3 = data['chain3']
chain4 = data['chain4']
chain5 = data['chain5']
chain6 = data['chain6']
chain7 = data['chain7']
chain8 = data['chain8']
chain9 = data['chain9']
chain10 = data['chain10']
cosine_similarities = []

# function
def checkIndexOrNot(output1,chain_zip):
    value = chain_zip.get(output1, 0)
    return value

@app.route('/',methods=['POST','GET'])
def results():
    # form = request.form
    embeddings = []
    cosine_similarities = []
    if request.args.get('semester') is not None and request.args.get('description') != '':
        # c_des.append(request.args.get('description'))
        # embeddings = model.encode(c_des)
        # embeded = np.array(embeddings)
        # array_len = len(embeded)
        # # print(array_len)
        # for i in range(0,array_len):
        #     cosine = np.dot(embeded[array_len-1],embeded[i])/(norm(embeded[array_len-1])*norm(embeded[i]))
        #     cosine_similarities.append(cosine)
        # cosine_similarities.pop()
        # c_des.pop()
        # index = cosine_similarities.index(max(cosine_similarities))
        
        # output1 = courses_dataset['Course Name'][index]
        # output1_sem = courses_dataset['Semester'][index]

        semester = request.args.get('semester')
        description = request.args.get('description')
        c_des.append(description)
        embeddings = model.encode(c_des)
        embeded = np.array(embeddings)
        array_len = len(embeded)
        # Compute dot product of the last embedding with all other embeddings
        dot_product = np.dot(embeded[array_len-1], embeded.T)
        # Compute norms of all embeddings
        norms = np.linalg.norm(embeded, axis=1)
        # Compute cosine similarities
        cosine_similarities = dot_product / (norms[array_len-1] * norms)
        # Remove the last cosine similarity as it corresponds to the dot product of the last embedding with itself
        cosine_similarities = cosine_similarities[:-1]
        c_des.pop()
        index = np.argmax(cosine_similarities)
        output1 = courses_dataset['Course Name'][index]
        output1_sem = courses_dataset['Semester'][index]


        if(checkIndexOrNot(output1,chain1) != 0):
            final_chain = chain1
        elif(checkIndexOrNot(output1,chain2) != 0):
            final_chain = chain2
        elif(checkIndexOrNot(output1,chain3) != 0):
            final_chain = chain3
        elif(checkIndexOrNot(output1,chain4) != 0):
            final_chain = chain4
        elif(checkIndexOrNot(output1,chain5) != 0):
            final_chain = chain5
        elif(checkIndexOrNot(output1,chain6) != 0):
            final_chain = chain6
        elif(checkIndexOrNot(output1,chain7) != 0):
            final_chain = chain7
        elif(checkIndexOrNot(output1,chain8) != 0):
            final_chain = chain8
        elif(checkIndexOrNot(output1,chain9) != 0):
            final_chain = chain9
        else:
            final_chain = chain10

        user_sem = int(request.args.get('semester'))
        final = [{k:v} for k,v in final_chain.items() if v >= user_sem]
        output_dict = {k: v for d in final for k, v in d.items()}
        if(output1 in output_dict):
            del output_dict[output1]

        else:
            for new_s, new_val in output_dict.items():
                output1 = new_s
                output1_sem = new_val
                break
            if(output1 in output_dict):
                del output_dict[output1]
        return render_template('index.html', semester=semester, description=description, output_des=output1, output_sem=output1_sem,output_dict=output_dict)
    
    else:
        return render_template('index.html')

# add feedback
@app.route('/addfeedback', methods=['POST','GET'])
def add_addfeedback():
    if request.method == "POST":
        feedbackname = request.form['feedbackname']
        collection.insert_one({'name' : feedbackname})
        message = "Feedback added successfully"
        return render_template('index.html',message_feedback=message)
    if request.method == "GET":
        return render_template('index.html')

# run file
app.run("localhost", "9999", debug=True)