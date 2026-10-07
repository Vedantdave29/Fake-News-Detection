Step 1: Load the Data (step1_load.py)

-> Purpose: Load the news dataset and prepare it for use.

    Step 1.Read Fake.csv and True.csv using pandas.
    Step 2.Add a label column (0 = Fake, 1 = Real).
    Step 3.Merge both files into one table.
    Step 4.Remove duplicate articles.
    Step 5.Shuffle the rows so fake and real news are mixed.
    Step 6.Print the shape and label counts to check the data.

    Output: A single dataset of about 44,000 articles.


-----------------------------------------------------------------------------------------------------


Step 2: Clean the Text (step2_clean.py)

-> Purpose: Convert raw news text into clean words the model can learn from.

    Step 1.Load and merge the dataset (same as Step 1).
    Step 2.Remove the "(Reuters)" tag so the model doesn't cheat by recognizing the source.
    Step 3.Convert text to lowercase.
    Step 4.Remove links, numbers, and punctuation.
    Step 5.Remove stopwords (common words like the, is, and).   
    Step 6.Lemmatize words to their base form (cars becomes car).
    Step 7.Combine the title and text into one column called clean.
    Step 8.Save the result as data/clean_data.csv.


----------------------------------------------------------------------------------------------------

Step 3: Train the Models (step3_train.py)

    Step 1 :Load clean_data.csv.
    Step 2.Split into 80% training and 20% testing data.
    Step 3:Convert text to numbers using TF-IDF.
    Step 4:Train 4 models: Logistic Regression, Naive Bayes, Passive Aggressive, Linear SVM.
    Step 5:Compare accuracy and F1-score.
    Step 6:Save the best model as model.pkl and vectorizer.pkl.



----------------------------------------------------------------------------------------------------

Step 4: Web App (app.py)

    Step 1:Load the saved model and vectorizer.
    Step 2:User pastes a news article.
    Step 3:The text is cleaned the same way as in Step 2.
    Step 4:The model predicts Real or Fake with a confidence score.