# Sentiment Analysis Web Application

This Flask-based web app analyzes tweets' sentiment using a pre-trained machine learning model. The application provides a simple user interface to input tweets and get sentiment predictions.

Check out the live deployment of the application on Heroku: [Sentiment Analysis Web App](https://p07badbuzz-de1a8332d456.herokuapp.com/)

## Project Structure

The project consists of the following main files and directories:

- `static/style.css`: CSS file for styling the frontend.
- `templates/index.html`: HTML template for the web interface.
- `app.py`: The main Flask application containing the sentiment analysis logic.
- `lr_model.pkl`: Pre-trained machine learning model (Linear Regression) for sentiment analysis.
- `Procfile`: Configuration file for Heroku deployment.
- `requirements.txt`: List of required Python packages for the project.
- `test_app.py`: Unit tests for the application.
- `tfidf.pickle`: Pickled TF-IDF vectorizer used for text preprocessing.

## Getting Started

Follow these steps to run the application on your local machine:

1. Clone the repository.
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
3. Run the application:
	```bash
	python app.py
	```
4. Access the application in your web browser at http://127.0.0.1:5000/

## Usage

1. Enter a tweet in the input field and submit it.
2. The application will process the input and display the sentiment classification result as either 'Positive' or 'Bad Buzz !!!'.

## Customization
You can customize the application in the following ways:

* Model and Vectorizer: If needed, replace the pre-trained machine learning model (lr_model.pkl) and vectorizer (tfidf.pickle) with your own trained models.

* Emojis and Preprocessing: Modify the EMOJIS dictionary and the related functions in app.py to change emoji descriptions and preprocessing steps.

* Frontend: Modify the templates/index.html file to adjust the appearance and layout of the application.

## Testing
Run unit tests to ensure the application's functionality:
	```bash
	pytest test_app.py
	```
	
## Deployment
This application is deployed on Heroku using the provided `Procfile` and `requirements.txt`.

## License
This project is licensed under the MIT License. Feel free to fork the repository and adapt it to your needs.