# study notes
 
01_NLP_twitter_diseaster

This is my first project learning NLP. The project goal is to classify if a twitter message is indicating a diseaster situation or not. Project details and source data can be found here:
https://www.kaggle.com/competitions/nlp-getting-started

I applied the following techniques to gradually improve the model performance:
- text vectorization using glove embedding
- apply early stopping to prevent model from overfitting;
- apply flatten layer and global pooling layer to improve model structure
- Use RNN/LSTM model, on top of glove embedding
- Remove special characters, stopping words
- Bert model (did not finish)
  
This project was done before LLM. If we apply LLM / transformer, the performance would be even better  
