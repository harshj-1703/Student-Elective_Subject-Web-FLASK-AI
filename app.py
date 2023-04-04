# inport library
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

courses_dataset = pd.read_excel('./model train files/Elective Courses.xlsx')
c_des = list(courses_dataset['Description'])
model = SentenceTransformer('all-MiniLM-L6-v2')

# initializing flask
app = Flask('ElectiveRecommendation')

@app.route('/')
def show_predict_stock_form():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    form = request.form
    c_des.append(request.form['description'])
    embeddings = model.encode(c_des)
    array_len = len(np.array(embeddings))
    cosine_similarities = []
    for i in range(0,array_len):
        cosine = np.dot(np.array(embeddings)[array_len-1],np.array(embeddings)[i])/(norm(np.array(embeddings)[array_len-1])*norm(np.array(embeddings)[i]))
        cosine_similarities.append(cosine)
    cosine_similarities.pop()
    index = cosine_similarities.index(max(cosine_similarities))

    chain1={"Advanced Microprocessor" : 5,"Embedded System Design" : 6,"VLSI Designs" : 6,"Digital Design using Verilog" : 7,"VLSI Physical Design" :8,"FPGA Based System Design" :8,"Embedded Linux" :8,"Internet of Things":7}
    chain2={"Optical Communication" : 5,"RF and Microwave Communication" : 6,"Satellite Communication" : 6,"Wireless system Design" : 7,"Multimedia computing" : 7,"Spread Spectrum communications" : 8}
    chain3={"Linux Administration" : 5,"Cloud Computing" : 6,"Embedded Operating System" : 7,"Introduction to DevOps Tools" : 8,"Network Administration" : 8,"Advanced Computer Networks" : 6}
    chain4={"Applied Linear algebra" : 5,"Machine learning" : 6,"Data Warehousing and Data mining" : 6,"Big Data Analytics" : 7,"Computer Vision" : 7,"Human computer interaction" : 7,"Data Visualization" : 8,"Deep Learning for Computer Vision" : 8}
    chain5={"Theory of Computation" : 5,"Compiler Design" : 6}
    chain6={".NET Technology" : 6,"Advance Web Technology" : 8}
    chain7={"Programming for Application Development" : 7,"Cross Platform Mobile Development" : 8}
    chain8={"Game Development" : 8}
    chain9={"Advance Java" : 7,"Advance Database" : 8,"Advance C++ Programming" : 8}
    chain10={"SEO And Digital Marketing" : 8}

    output1 = courses_dataset['Course Name'][index]
    output1_sem = courses_dataset['Semester'][index]

    def checkIndexOrNot(output1,chain_zip):
        value = chain_zip.get(output1, 0)
        return value

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

    user_sem = int(request.form['semester'])
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

    # if request.method == 'POST':
    #   #write your function that loads the model
    #   model = get_model() #you can use pickle to load the trained model
    semester = request.form['semester']
    description = request.form['description']
    # predicted_stock_price = model.predict(year)
    return render_template('result.html', semester=semester, description=description, output_des=output1, output_sem=output1_sem)

# run file
app.run("localhost", "9999", debug=True)