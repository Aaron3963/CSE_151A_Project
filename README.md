# CSE 151A Project
## Milestone 3: Pre-Processing

### Data Preprocessing
For the preprocessing process we have 5 main steps:
1. Clean out the nulls: We have found null data within the CNBC network dataset. We will be dropping these rows since we cannot fill the data with random words. 
2. Removing extraneous columns: Two of the datasets have a description column which we will be dropping since we are only going to analyze the headlines
3. Combine all the datasets: This step will clean and concatenate three news datasets by aligning each row with the appropriate stock date, and then combine them into a unified dataset with features [date, headline, stock price].
4. Tokenize: This step will turn each word in the dataset into an integer based off a dictionary created by scanning all the headlines
5. Embeddings: This will be a randomly generated embeddings table that will allow the model to get an embeddings for each word. 

### First Model: Logistic Regression with TF-IDF  
**Model Summary**
1. Features: TF-IDF vectorized text with max_features varying between 50 and 1200.
2. Training Algorithm: Logistic Regression with 1500 maximum iterations.
3. Evaluation Metrics: Training and test errors (1 - accuracy rate) were used to analyze model performance and identify underfitting or overfitting.  
The model achieved 63.8% accuracy on test set and 76.2% on training set.


**Model Evaluation and Future Models**

Based on the fitting graph:

1. Optimal Complexity: The model performed best at max_features = 800, achieving the lowest test error. This suggests that the model is most balanced at this level of feature complexity.
2. Overfitting Zone: When max_features exceeded 800, the test error began to increase while training error continued to decrease, indicating overfitting.
3. Underfitting Zone: At lower values of max_features (e.g., 50–200), both training and test errors were high, reflecting insufficient model complexity.

For future models, we plan to focus on natural language processing. The first model we want to try is the basic transformer model where the use of attention could find words that help dictate a positive or negative trend. We also want to explore the use of FinBert which is a model that is specifically trained to analyze sentiment of financial texts. With this model we can hybridize it by attaching an RNN in order to make this problem a regression problem that predicts exactly how much the S&P500 will change by. 

**Model Conclusion**

Overall our model did alright. It had a binary classification task of predicting whether the S&P500 will rise or fall and it did slightly better than just guessing. This was expected since this type of model does not perform any sort of sentiment analysis. Some hyperparameter tuning can be done to improve this model; however, it is very limited in how much it can accomplish given the data. Nonetheless, this model gives us a good baseline for how a model should perform and we hope the future models we implement can outperform it.  


### Current Files
**Preprocessed Dataset.** [View Dataset](dataset/final_dataset.csv)

**Exploratory Data Analysis.** [View EDA](EDA.ipynb)

**First model.** [View Model](Model1.ipynb)
