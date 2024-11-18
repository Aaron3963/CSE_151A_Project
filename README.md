# CSE 151A Project
## Milestone 3: Pre-Processing

### Data Preprocessing
For the preprocessing process we have 5 main steps:
1. Clean out the nulls: We have found null data within the CNBC network dataset. We will be dropping these rows since we cannot fill the data with random words. 
2. Removing extraneous columns: Two of the datasets have a description column which we will be dropping since we are only going to analyze the headlines
3. Combine all the datasets: This step will clean and concatenate three news datasets by aligning each row with the appropriate stock date, and then combine them into a unified dataset with features [date, headline, stock price].
4. Tokenize: This step will turn each word in the dataset into an integer based off a dictionary created by scanning all the headlines
5. Embeddings: This will be a randomly generated embeddings table that will allow the model to get an embeddings for each word. 

First Model: Logistic Regression with TF-IDF  
Model Summary  
1. Features: TF-IDF vectorized text with max_features varying between 50 and 1200.
2. Training Algorithm: Logistic Regression with 1500 maximum iterations.
3. Evaluation Metrics: Training and test errors (1 - accuracy rate) were used to analyze model performance and identify underfitting or overfitting.  
The model achieved 63.8% accuracy on test set and 76.2% on training set.


### Where does your model fit in the fitting graph? and What are the next models you are thinking of and why?

Based on the fitting graph:

1. Optimal Complexity: The model performed best at max_features = 800, achieving the lowest test error. This suggests that the model is most balanced at this level of feature complexity.
2. Overfitting Zone: When max_features exceeded 800, the test error began to increase while training error continued to decrease, indicating overfitting.
3. Underfitting Zone: At lower values of max_features (e.g., 50–200), both training and test errors were high, reflecting insufficient model complexity.


Things we did:
Cleaned and combined the dataset. [View Dataset](dataset/final_dataset.csv)

Exploratory Data Analysis. [View EDA](EDA.ipynb)

First model. [View Moddel](Model1.ipynb)
