# Amazon Fake Review Detection

Group Project for CSE 6242 @ GeorgiaTech 

This project was created as a part of the graduate course CSE 6242 - Data and Visual Analytics (Fall 2021). <br/>
Authors: Sittun Swayam Prakash, Atrima Ghosh, Parth Iramani, Zoe Masood, Jenna Gottschalk, Mugundhan Murugesan

Fake Review Detection Website Demo <br/>
![Fake Review Detection Website Demo](https://media.giphy.com/media/lojgb9uYmTqVQrriiH/giphy-downsized-large.gif)

Project Report Summary: <br/>
![Summary](https://github.com/sittunswayam/Amazon-Fake-Review-Detection/blob/main/REPORT/Report.png)

### DESCRIPTION:

The aim of our project is to detect fake reviews on Amazon using the review text. Our approach combines semi-supervised learning and transformer models to identify fake Amazon reviews.
The end product of our project is a web application which will allow the user to predict whether a review is fake or real.
The Datasets used in our projects are:
Labelled Dataset from Kaggle - https://www.kaggle.com/lievgarcia/amazon-reviews
Unlabelled dataset from Amazon - https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_multilingual_US_v1_00.tsv.gz

Following are the steps involved in creating and evaluation of the Model to predict fake reviews using review text: <br/>
(1) We split the original labeled dataset into four parts: 70% training set, 10% first validation set to compare initial supervised classification models, 10% second validation set to compare the updated classification models, and 10% test set to evaluate the final classifier’s performance. <br/>
(2) We trained the initial model on the training set of labeled data. <br/> 
(3) We generated pseudo-labels by using the initial model to classify the unlabeled data set. <br/>
(4) The most confidently predicted pseudo-labels above a specific threshold became training data for the next step. <br/>
(5) We updated the initial model using the pseudo-labels as training data. <br/>
(6) Finally, we tested the final classifier on the test set of the original labeled data. <br/>

Modules used: <br/>
Pandas, Numpy, scikit-learn - Commonly used Machine Learning python libraries <br/>
transformers - Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction, question answering, summarization, translation, text generation and more in over 100 languages. <br/>
datasets - Huggingface Datasets is a library for easily accessing and sharing datasets, and evaluation metrics for Natural Language Processing (NLP), computer vision, and audio tasks. <br/>

The following are the steps and respective code (.py files) invloved in obtaining the final model file that is used for prediction on the website. <br/>
1. Modeling - Sentence Transformer and getting embeddings for review text <br/>
	|___ 11initial_cleaning.py <br/>
2. Logistic Regression with Labelled Dataset <br/>
	|___ 21initial_modelling.py <br/>
3. Generation of Pseudo Labels <br/>
	|___ 31clean_final_dataset.py <br/>
	|___ 32get_partitions.py <br/>
	|___ 33get_embeddings_for_partition.py <br/>
	|___ 34pseudo_labels.py <br/>
4. Unsupervised Learning using Stochastic Gradient Descent (SGD) <br/>
	|___ 41SGD_log.py <br/>

Analysis using BERT (Bidirectional Encoder Representations from Transformers) model: <br/>
1. Supervised Learning <br/>
	|___ 1_6242_GROUP_PROJECT_BERT_training_for_a.py <br/>
2. Generation of Pseudo_labels <br/>
	|___ 2_6242_GROUP_PROJECT_Generate_BERT_Pseudo_Labels.py <br/>
3. Unsupervised Learning <br/>
	|___ 3_6242_GROUP_PROJECT_BERT_training_for_b.py <br/>
4. Compilation of final dataset for visualization <br/>
	|___ 4_6242_GROUP_PROJECT_compiling_final_dataset <br/>


### Description of the website:

The web-application hosted on (http://machinelearner.eastus2.cloudapp.azure.com/) allows the user to enter the URL of the review or the text of the review and get a prediction. 

The public Tableau dashboard (https://public.tableau.com/app/profile/sittun.prakash/viz/Book1_16398770354100/Dashboard1) is also accessible from the website. 

Directory Structure: <br/>
static <br/>
  |___ style.css (Style Sheet for html files) <br/>
templates <br/>
  |___ home.html (Renders the default home page) <br/>
  |___ manual.html (Renders the manual entry page for review data) <br/>
  |___ result.html (Renders the result page after model prediction) <br/>
LogReg.pkl (The saved Logistic Regression model that generates the predictions under the hood <br/>
download_sentence_transformer.py (Commands used to download and save st_model folder) <br/>
run.py (The main python file running the Flask application which carries out all the computation) <br/>
gibberish_classifier.py (The Python classifier which checks if review text entered is gibberish - if yes, it asks user to re-enter review data) <br/>
st_model (The Sentence Transformer model that is used to generate paragraph embeddings from text data in run.py) <br/>

Modules used in run.py: <br/>
flask - to build the Python web application <br/>
pandas - to process data as dataframes <br/>
joblib - to load the LogReg.pkl saved model <br/>
sklearn - to generate the prediction output and prediction report <br/>
requests - to get html data from amazon URL <br/>
bs4 - BeautifulSoup package to scrape relevant data off html page returned by requests <br/>
sentence_transformers - to get the paragraph embeddings on text data from the saved st_model <br/> 

### INSTALLATION:

The python code files include commands that will install all the required libraries.

### EXECUTION:

First execute download_sentence_transformer.py to create the st_model folder needed for the website.
Execute run.py to run the website on your local server. 
