from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from numpy.linalg import norm
from json import dump

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
cosine_similarities = []

# function
def checkIndexOrNot(output1,chain_zip):
    value = chain_zip.get(output1, 0)
    return value

# initializing flask
app = Flask('ElectiveRecommendation')

# @app.route('/')
# def show_predict_stock_form():
#     return render_template('index.html')

@app.route('/results',methods=['POST','GET'])
def results():
    # form = request.form
    cosine_similarities = []
    embeddings = []
    if request.args.get('semester') is not None and request.args.get('description') != '':
        c_des.append(request.args.get('description'))
        embeddings = model.encode(c_des)
        array_len = len(np.array(embeddings))
        # print(array_len)
        for i in range(0,array_len):
            cosine = np.dot(np.array(embeddings)[array_len-1],np.array(embeddings)[i])/(norm(np.array(embeddings)[array_len-1])*norm(np.array(embeddings)[i]))
            cosine_similarities.append(cosine)
        cosine_similarities.pop()
        c_des.pop()
        index = cosine_similarities.index(max(cosine_similarities))
        
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
        # if(user_sem > output1_sem):
        if(output1 in output_dict):
            del output_dict[output1]
            output = sorted(output_dict)
            # output.pop()
            # output1 = 
            # print("\n---------------------------------\nYou can select subject : ",output1," in sem : ",output1_sem,"th")
            # if(output_dict != {}):
            #     for i,j in output_dict.items():
            #         print("Also you can select similer subject like :",i,"from semester : ",j,"th")

        else:
            output = sorted(output_dict)
            for new_s, new_val in output_dict.items():
                output1 = new_s
                output1_sem = new_val
                break
            output.pop()
            # print("\n---------------------------------\nYou can select subject : ",output1," in sem : ",output1_sem,"th")
            # if(output_dict != {}):
            #     for i,j in output_dict.items():
            #         print("Also you can select similer subject like :",i,"from semester : ",j,"th")

        #   #write your function that loads the model
        #   model = get_model() #you can use pickle to load the trained model
        semester = request.args.get('semester')
        description = request.args.get('description')
        # predicted_stock_price = model.predict(year)
        return render_template('index.html', semester=semester, description=description, output_des=output1, output_sem=output1_sem,output_dict=output_dict)
    
    else:
        return render_template('index.html')

# run file
app.run("localhost", "9999", debug=True)