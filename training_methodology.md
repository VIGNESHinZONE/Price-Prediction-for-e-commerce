# Training Approaches - 

  1. Exploratory Data Analysis performed in this [notebook](https://github.com/m-rec/547fc32ae0e26fbd6ebc5b4193c93468965a80dd/blob/master/notebooks/Exploratory_Data_Analysis.ipynb).  
  
  2. Baseline Model [notebook](https://github.com/m-rec/547fc32ae0e26fbd6ebc5b4193c93468965a80dd/blob/master/notebooks/Simple_Ridge_LGBM.ipynb) - 
      * We simplified bag of words models over the ['name', 'category'] features.
      * Created a list of popular brands that had existed more than 5 times in training data and converted
      the rest of the features into 'missing' value.
      * One hot encoded ['item_condition_id', 'shipping'] features.
      * Build a tf-idf encoder for ['item_description'] feature.
      * Used an ensemble of two LGBM models and one Ridge model to build the final classifier.
      
      <B>RMSLE</B> - 0.503 using 3-fold cross validation.
  
 3. Expanded Ridge [notebook](https://github.com/m-rec/547fc32ae0e26fbd6ebc5b4193c93468965a80dd/blob/master/notebooks/Expanded_Ridge.ipynb) - 
 
    The motivation was to build a large feature matrix, to further investigate the performance of the Ridge model. 
    LGBM models were removed because Boosting models don't perform well in large features.
 
      * Created new feature columns called ['brand_missing', 'description_missing'] which would indicate the presence of missing value.
      * Converted ['category'] feature into ['primary_category', 'secondary_category'] such that, we split the text on `/` token. Further converted into one-hot encoding.
      * We expanded bag of words models for ['name', 'item_description'] features with a very large vocabulary.
      * Created a list of popular brands that had existed more than 3 times in training data and converted
      the rest of the features into 'missing' value.
      * One hot encoded ['item_condition_id', 'shipping'] features.
      * Converted the entire table using Tf-idf transformer as the whole table was similar to the sparse matrix of bag-of-words.
      * Used a Ridge model with a large parameter size.
      
      <B>RMSLE</B> - 0.523 using 3-fold cross validation 
      
  4. Doc2vec embeddings [notebook](https://github.com/m-rec/547fc32ae0e26fbd6ebc5b4193c93468965a80dd/blob/master/notebooks/Ridge_Doc2vec.ipynb) - 
  
    The motivation was to decrease the size of the feature matrix to improve the accuracy of LGBM model.
  
      * Built doc2vec embeddings for ['name', 'item_description'] features.
      * Created a list of popular brands that had existed more than 10 times in training data and converted
      the rest of the features into 'missing' value. 
      * Created new feature columns called ['brand_missing', 'description_missing'] which would indicate the presence of missing value.
      * Converted ['category'] feature into ['primary_category', 'secondary_category'] such that, we split the text on `/` token. Further converted into one-hot encoding.
      * One hot encoded ['item_condition_id', 'shipping'] features.
      * Used an ensemble of two LGBM models and one Ridge model to build the final classifier.
      
      <B>RMSLE</B> - 0.5763 using 3-fold cross validation 