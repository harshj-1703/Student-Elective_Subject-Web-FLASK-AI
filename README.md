# Student-Elective_Subject-Web-FLASK-AI

---

# Youtube Link : https://youtu.be/83CbzukWJz4

---

# Objective Of Project:

Elective subject recommendation is use to give student recommendation about which elective student can join based on their interest.

---

# Basic details:

Their are two types for sentence matching in natural language processing
1)Lexical
2)Semantic

in laxical it will match words with word to word matching
in semantic it will match words based on their semantic review

in laxical first i have geted keywords from sentence input with keybert library with stop words 0.7, here stop words is thing that removes english language extra words from sentence input.
then in laxical i have done keybert for all the datasets of description and finded the unique words from datasets.
then i have used tfidf(term frequency inverse transform frequency) for word matching instead of bag of words.
the problem in laxical analysis is that it will match based on words so for some inputs it will shows wrong outputs or elective recommend.

in semantic analysis i have used library sentence-transformers and in that i have used 'multi-qa-MiniLM-L6-cos-v1' model.
it will shows the correct output because it matches words based on semantic search

here in next step i have used flask framework of python for making web.
i have used html for templates.
first i have faced problem with flask that it takes much high time for input transformation to recommend elective subject
but after that i have used test.py output store in trained model output as a model.pkl(pickle) file that stores train model and after that i am getting very faster speed for user input.

---

# Requirements for run my project:

currently we need python and some libraries installed but after web deployment we not need anything just website should be open.

---

# Libraries should be installed for this project without deployment:

sentence transformers -> uses for models for semantic analysis(here i have used multi-qa-MiniLM-L6-cos-v1' model),
flask -> uses for website api with python,
pickle -> uses for serialize and deserialize python object or store trained model data in that,
numpy -> used for scientific computing and data analysis,
pymongo -> for mongo database connection,
pandas -> for data manipulation and analysis,

---

# Credits

KeyBERT: https://github.com/MaartenGr/KeyBERT
sentence-transformers: https://github.com/UKPLab/sentence-transformers
