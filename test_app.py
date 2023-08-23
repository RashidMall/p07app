import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_positive_sentiment(client):
    response = client.post('/', data={'tweet': 'I love this product! It\'s amazing.'})
    assert b'Positive' in response.data

def test_bad_buzz_sentiment(client):
    response = client.post('/', data={'tweet': 'This product is terrible. I hate buying it.'})
    assert b'Bad Buzz !!!' in response.data