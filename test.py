# # inport library
# from flask import Flask, render_template, request
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sentence_transformers import SentenceTransformer
# from numpy.linalg import norm
# import pickle

# courses_dataset = pd.read_excel('./model train files/Elective Courses.xlsx')
# c_des = list(courses_dataset['Description'])
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # similar subjects
# chain1={"Advanced Microprocessor" : 5,"Embedded System Design" : 6,"VLSI Designs" : 6,"Digital Design using Verilog" : 7,"VLSI Physical Design" :8,"FPGA Based System Design" :8,"Embedded Linux" :8,"Internet of Things":7}
# chain2={"Optical Communication" : 5,"RF and Microwave Communication" : 6,"Satellite Communication" : 6,"Wireless system Design" : 7,"Multimedia computing" : 7,"Spread Spectrum communications" : 8}
# chain3={"Linux Administration" : 5,"Cloud Computing" : 6,"Embedded Operating System" : 7,"Introduction to DevOps Tools" : 8,"Network Administration" : 8,"Advanced Computer Networks" : 6}
# chain4={"Applied Linear algebra" : 5,"Machine learning" : 6,"Data Warehousing and Data mining" : 6,"Big Data Analytics" : 7,"Computer Vision" : 7,"Human computer interaction" : 7,"Data Visualization" : 8,"Deep Learning for Computer Vision" : 8}
# chain5={"Theory of Computation" : 5,"Compiler Design" : 6}
# chain6={".NET Technology" : 6,"Advance Web Technology" : 8}
# chain7={"Programming for Application Development" : 7,"Cross Platform Mobile Development" : 8}
# chain8={"Game Development" : 8}
# chain9={"Advance Java" : 7,"Advance Database" : 8,"Advance C++ Programming" : 8}
# chain10={"SEO And Digital Marketing" : 8}

# data = {'courses_dataset': courses_dataset,
#         'c_des': c_des,
#         'model': model,
#         'chain1': chain1,
#         'chain2': chain2,
#         'chain3': chain3,
#         'chain4': chain4,
#         'chain5': chain5,
#         'chain6': chain6,
#         'chain7': chain7,
#         'chain8': chain8,
#         'chain9': chain9,
#         'chain10': chain10}

# with open('model.pickle', 'wb') as f:
#   pickle.dump(data, f)

# inport library
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer,util
from numpy.linalg import norm
import pickle

courses_dataset = pd.read_excel('./model train files/Elective Courses.xlsx')
c_des = list(courses_dataset['Description'])
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

trained_des = model.encode(c_des)

# similar subjects
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

data = {'courses_dataset': courses_dataset,
        'c_des': c_des,
        'trained_des' : trained_des,
        'model': model,
        'chain1': chain1,
        'chain2': chain2,
        'chain3': chain3,
        'chain4': chain4,
        'chain5': chain5,
        'chain6': chain6,
        'chain7': chain7,
        'chain8': chain8,
        'chain9': chain9,
        'chain10': chain10}

with open('model.pickle', 'wb') as f:
  pickle.dump(data, f)