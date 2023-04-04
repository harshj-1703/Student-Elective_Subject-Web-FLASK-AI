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
    chain1=["Advanced Microprocessor","Embedded System Design","VLSI Designs","Digital Design using Verilog","VLSI Physical Design","FPGA Based System Design","Embedded Linux","Internet of Things"]
    chain1_sem=[5,6,6,7,8,8,8,7]
    chain2=["Optical Communication","RF and Microwave Communication","Satellite Communication","Wireless system Design","Multimedia computing","Spread Spectrum communications"]
    chain2_sem=[5,6,6,7,7,8]
    chain3=["Linux Administration","Cloud Computing","Embedded Operating System","Introduction to DevOps Tools","Network Administration","Advanced Computer Networks"]
    chain3_sem=[5,6,7,8,8,6]
    chain4=["Applied Linear algebra","Machine learning","Data Warehousing and Data mining","Big Data Analytics","Computer Vision","Human computer interaction","Data Visualization","Deep Learning for Computer Vision"]
    chain4_sem=[5,6,6,7,7,7,8,8]
    chain5=["Theory of Computation","Compiler Design"]
    chain5_sem=[5,6]
    chain6=[".NET Technology","Advance Web Technology"]
    chain6_sem=[6,8]
    chain7=["Programming for Application Development","Cross Platform Mobile Development"]
    chain7_sem=[7,8]
    chain8=["Game Development"]
    chain8_sem=[8]
    chain9=["Advance Java","Advance Database","Advance C++ Programming"]
    chain9_sem=[7,8,8]
    chain10=["SEO And Digital Marketing"]
    chain10_sem=[8]
    chain1_zip = dict(zip(chain1,chain1_sem))
    chain2_zip = dict(zip(chain2,chain2_sem))
    chain3_zip = dict(zip(chain3,chain3_sem))
    chain4_zip = dict(zip(chain4,chain4_sem))
    chain5_zip = dict(zip(chain5,chain5_sem))
    chain6_zip = dict(zip(chain6,chain6_sem))
    chain7_zip = dict(zip(chain7,chain7_sem))
    chain8_zip = dict(zip(chain8,chain8_sem))
    chain9_zip = dict(zip(chain9,chain9_sem))
    chain10_zip = dict(zip(chain10,chain10_sem)) 
    output1 = courses_dataset['Course Name'][index]
    output1_sem = courses_dataset['Semester'][index]

    def checkIndexOrNot(output1,chain_zip):
        value = chain_zip.get(output1, 0)
        return value

    if(checkIndexOrNot(output1,chain1_zip) != 0):
        final_chain = chain1_zip
    elif(checkIndexOrNot(output1,chain2_zip) != 0):
        final_chain = chain2_zip
    elif(checkIndexOrNot(output1,chain3_zip) != 0):
        final_chain = chain3_zip
    elif(checkIndexOrNot(output1,chain4_zip) != 0):
        final_chain = chain4_zip
    elif(checkIndexOrNot(output1,chain5_zip) != 0):
        final_chain = chain5_zip
    elif(checkIndexOrNot(output1,chain6_zip) != 0):
        final_chain = chain6_zip
    elif(checkIndexOrNot(output1,chain7_zip) != 0):
        final_chain = chain7_zip
    elif(checkIndexOrNot(output1,chain8_zip) != 0):
        final_chain = chain8_zip
    elif(checkIndexOrNot(output1,chain9_zip) != 0):
        final_chain = chain9_zip
    else:
        final_chain = chain10_zip

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
    return render_template('result.html', semester=semester, description=description, output_des=output1, output_sem=output1_sem, output_dict = output)


# run file
app.run("localhost", "9999", debug=True)