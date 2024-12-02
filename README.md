# CSE 151A Project
## Milestone 4: Second Model + more

### Transformer Classifier
Generally throughout the different iterations of our Transformer model, we found that it finished fitting within the first 10 epochs. This could most likely be occuring do to our large batch sizes and how big the data is. Our best Transformer model is one where we implemented the following improvements: we did hyperparameter tuning to find the ideal number of heads, layers, etc, change the number of outputs in the final layer of the Transformer model from 2 to 1, and had no learning rate decay. This model obtained a test accuracy of 59.2% and a train accuracy of 69.8%. Although this is better than our first model, TFIDF, we feel that there is still room for improvement with other models.

**Note:** Please see the end of our [Project Notebook](project.ipynb) to see the final model's fitting graph, train vs. test error, and predictive stats on the test data.

### Conclusion
Overall our second model performed worse than our expectations. Althrough it beat out the first model, it was very marginal compared to the hyperparameter tuning that was done. Some improvements that can be done to this model is to possibly create and ensemble with a time series model and transform the problem into a regression problem. We believe that there is a lot of noise in the data to do a simple classification, so reworking the problem may utilize the model the best. These changes could improve the model; however, we believe that the improvements would be limited due to the nature of the model and the data itself. 

### Future Models
Another that we plan to look into are LSTMs. These types of models perform in both NLP tasks and time series tasks. Since our problem is heavily dependent on those two things, LSTMs could be the perfect model.
We have also already began working with BERT because it is bidirectional and it is specifically built for sentiment analysis. We believe that the combination of these facts along with attention can boost the accuracy on this dataset. 

**Things we did:**
New grouped dataset for the Transformer Classifier. [View Dataset](grouped_dataset.csv)

Transformer Classifier Model. [View Model](project.ipynb) (Under the Transformer Classifier header)
