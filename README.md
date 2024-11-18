# CSE 151A Project
## Milestone 3: Pre-Processing

### Data Preprocessing
For the preprocessing process we have 5 main steps:
1. Clean out the nulls: We have found null data within the CNBC network dataset. We will be dropping these rows since we cannot fill the data with random words. 
2. Removing extraneous columns: Two of the datasets have a description column which we will be dropping since we are only going to analyze the headlines
3. Combine all the datasets: This step will clean and concatenate three news datasets by aligning each row with the appropriate stock date, and then combine them into a unified dataset with features [date, headline, stock price].
5. Tokenize: This step will turn each word in the dataset into an integer based off a dictionary created by scanning all the headlines
6. Embeddings: This will be a randomly generated embeddings table that will allow the model to get an embeddings for each word. 

### Train your first model

### Evaluate your model and compare training vs. test error

### Answer the questions: Where does your model fit in the fitting graph? and What are the next models you are thinking of and why?

### Update your README.md to include your new work and updates you have all added. Make sure to upload all code and notebooks. Provide links in your README.md

### Conclusion section: What is the conclusion of your 1st model? What can be done to possibly improve it?
