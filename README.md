Project Summary / Study Notes
 
**01_NLP_twitter_diseaster**

This is my first project learning NLP. The project goal is to classify if a twitter message is indicating a diseaster situation or not. Project details and source data can be found [here] (https://www.kaggle.com/competitions/nlp-getting-started)

I applied the following techniques to gradually improve the model performance:
- text vectorization using glove embedding
- apply early stopping to prevent model from overfitting;
- apply flatten layer and global pooling layer to improve model structure
- Use RNN/LSTM model, on top of glove embedding
- Remove special characters, stopping words
- Bert model (did not finish)
  
This project was done before LLM. If we apply LLM / transformer, the performance would be even better  


**02_CV_chest_X_ray**

This is my first project learning computer vision. The project goal is to classify Chest X-ray images as "Pneumonia / Normal". Project details can be found [here](https://www.kaggle.com/competitions/pneumonia-chest-x-ray-class-classification/overview)

The following techniques are practices / applied:
- Simple convolutional neural network （two convolutional layers)
- Complex convolutional neural network (five convolutional layers)
- Transfer leraning using Inception Network
- Apply dropout layer to mitigate overfitting
- Train model on TPU and GPU
- Train models with both Tensorflow and Pytorch


**03_CV_RSNA_ATD_Competition (Radiological Society of North America: Abdominal Trauma Detection)** 

I participated in this the Kaggle competition, where the goal is to detect and classify traumatic abdominal injuries. The project requirement is [here](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)
The prediction targets (multi-class classification) include:

Bowel: healthy/injury
Extravasation: healthy/injury

Kidney: Healthy/low/high
Liver: healthy/low/high
Spleen: healthy/low/high


In this competition, I primary tried two techniques:
- Transfer learning of Inception Network
- Restnet backbone

I did submit my solution before the competition deadline but it was far from a top solution. I later learned from the competition winner a top notch solution would involve two stage of modeling: stage 1: segmenting the area of the image and identify the object area and 2) feed the results into a CNN/RNN network for final prediction. 


**04_XGB_Regression_Housing_Prediction**

The project goal is to predict housing price based on Tabular data. The project detail is [here](https://www.kaggle.com/competitions/housing-price-prediction-isq/data)

Work in this project primarily involves classic ML techinques:
- Data exploration
- Missing value handling (remove records, fill with mean, etc)
- One hot encoding
- XGboost, lightBoost, Ridge Regression, Stacked Ensemble models










