# YouTube_Comment_Sentiment_Analyser

This project is a sophisticated YouTube Comment Sentiment Analyzer designed to decode the sentiment of individual comments. It scrapes comments from YouTube videos, preprocesses the text data, and performs sentiment analysis using machine learning techniques.

 ## Features

- Scrapes comments from YouTube videos using the YouTube API
- Preprocesses text data by removing emojis, tokenizing, and stemming
- Performs sentiment analysis using TextBlob library and Random Forest algorithm
- Trains a Random Forest classifier on the preprocessed data
- Evaluates model performance using accuracy, classification report, ROC-AUC curves, and confusion matrix
- Implements hyperparameter tuning using GridSearchCV and RandomizedSearchCV
- Employs stratified sampling and cross-validation techniques
- Explores ensemble methods like Bagging and AdaBoost classifiers
- Compares performance with other classifiers like SVM, Logistic Regression, and Naive Bayes
- Provides a user-friendly web interface to analyze sentiment distribution for YouTube videos

  ## Webpage

  Our project encompasses a user-friendly webpage designed for analyzing sentiments within YouTube video comments. Users can seamlessly share the YouTube video link on our platform to receive sentiment scores reflecting the overall sentiment of the video's comments. Leveraging React for frontend development and Flask for backend development, our webpage ensures a smooth and responsive user experience. The sentiment scores encompass three categories: positive, neutral, and negative. Upon analysis, our webpage dynamically changes its color scheme to visually represent the sentiment distribution.
  
  
The interface displays red dynamically when there’s a greater number of negative comments.
![Screenshot 2024-05-11 121033](https://github.com/shreyas066/IML_youtube_comment/assets/131354922/bde0538a-df4e-464d-b4b8-eaa4276eb55e) 

The interface displays grey dynamically when there’s a greater number of neutral comments.
![Screenshot 2024-05-11 131528](https://github.com/shreyas066/IML_youtube_comment/assets/131354922/5d45780f-be8d-493e-b684-04a90b5a2747)

The interface displays green dynamically when there’s a greater number of positive comments.
![WhatsApp Image 2024-05-11 at 15 23 18_abf173e6](https://github.com/shreyas066/IML_youtube_comment/assets/131354922/68a52db5-3d00-474f-878a-6886cd76a4bc) 




## Installation

1. Clone the repository
2. Install the required dependencies: pip install -r requirements.txt
3. Obtain a YouTube API key and replace it in the code
4. Run the youtube_comment_sentiment_analysis_finalcode.py script

## Usage

1. Enter the YouTube video ID when prompted
2. The script will scrape the comments, prepr
ocess the data, and train the sentiment analysis model
3. Model performance metrics and visualizations will be displayed
4. The trained model and vectorizer will be serialized for future use
5. Access the web interface to analyze sentiment distribution for YouTube videos

## License

This project is licensed under the [MIT License](LICENSE).

 ## Acknowledgments

- [NLTK](https://www.nltk.org/) for text processing
- [scikit-learn](https://scikit-learn.org/) for machine learning models and metrics
- [TextBlob](https://textblob.readthedocs.io/en/dev/) for sentiment analysis
- [Google API Client Library for Python](https://github.com/googleapis/google-api-python-client) for YouTube API integration
