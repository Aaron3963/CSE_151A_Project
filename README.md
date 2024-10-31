# CSE 151A Project
## Milestone 2: Data Exploration & Initial Preprocessing
clean nulls
remove extraneous columns (description)
concat dataset with the appropriate stock date and mash 3 datasets together
tokenize
create emeddings

For the preprocessing process we have 5 main steps:
1. Clean out the nulls: We have found null data within the CNBC network dataset. We will be dropping these rows since we cannot fill the data with random words. 
2. Removing extraneous columns: Two of the datasets have a description column which we will be dropping since we are only going to analyze the headlines
3. Concat dataset with stock date: After cleaning the dataset, we need to concat each row in all three datasets with the appropriate stock date. The final feature format for the dataset should be [date, headline, stock price].
4. Combine all three datasets: We will combine the datasets of all three news stations into one large dataset following the features from step 3. We are combining all the data since there is a large discrepancy between the dataset by the Guardian and CNBC. 
5. Tokenize: This step will turn each word in the dataset into an integer based off a dictionary created by scanning all the headlines
6. Embeddings: This will be a randomly generated embeddings table that will allow the model to get an embeddings for each word. 


I