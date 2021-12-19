README
------

Directory Structure:

static
  |___ style.css (Style Sheet for html files)
templates
  |___ home.html (Renders the default home page)
  |___ manual.html (Renders the manual entry page for review data)
  |___ result.html (Renders the result page after model prediction)
LogReg.pkl (The saved Logistic Regression model that generates the predictions under the hood
run.py (The main python file running the Flask application which carries out all the computation)
gibberish_classifier.py (The Python classifier which checks if review text entered is gibberish - if yes, it asks user to re-enter review data)
st_model (The Sentence Transformer model that is used to generate paragraph embeddings from text data in run.py)

Modules used in run.py:
flask - to build the Python web application
pandas
joblib - to load the LogReg.pkl saved model
sklearn - to generate the prediction output and prediction report
requests - to get html data from amazon URL
bs4 - BeautifulSoup package to scrape relevant data off html page returned by requests
sentence_transformers - to get the paragraph embeddings on text data from the saved st_model 

