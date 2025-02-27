import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def emotion_emotion(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This model tries to determine the emotion conveyed by the writer on a text, using Ekman's list of emotions as potential results.
		
		Although detecting emotion can be very subjective, this model could find patterns after analyzing over 210,000 samples of labeled text of comments made on social media."
    
    """
    url = f"https://detect4.p.rapidapi.com/detect/emotion/"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "detect4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sentiment_detector(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Similarly to the Emotions API, this model tries to determine the general sentiment of a given text based on how it was written."
    
    """
    url = f"https://detect4.p.rapidapi.com/detect/sentiment/"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "detect4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gibberish_detector(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We define as gibberish any text that has the particularity of not being intelligible in the target language of the reader.
		
		Unintelligible can be anything from a random sequence of characters, like `asdasqweqdaczc`, to a series of words that may be valid when analyzed one by one, but that in combination make no sense. For example: `dog boat the yes`.
		
		The former is somewhat more easily detectable by computers but the latter is much harder as they are existing words that just happen to make no sense in combination.
		
		Because of this, we introduced the concept of  `mild-gibberish`, which covers the case of sentences having valid words with occurrences of gibberish in it."
    
    """
    url = f"https://detect4.p.rapidapi.com/detect/gibberish/"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "detect4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def language_detector(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Our language detector uses SpaCy and FastText to predict what language the input text is written in.
		
		As many of our models, the longer the input the better the predictions are, especially if the text is written in a language that shares a recent common ancestor with another, like Latin languages do with each other. For example: Spanish, Portuguese and Catalan."
    
    """
    url = f"https://detect4.p.rapidapi.com/detect/language/"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "detect4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def names_detector(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API tries to determine what words in a given input correspond to a first-name and/or last-name.
		
		Although this task may seem simple, in many cultures some first names can also be last names, which can cause some confusion. For example, the name Franco can be both a first name as a family name, making it hard to label.
		
		We recommend using this API in combination with the *Gender API*."
    
    """
    url = f"https://detect4.p.rapidapi.com/detect/names/"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "detect4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gender_detector(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API tries to determine what the gender of a person is based on their name and a database of over 90,000 entries tagged with their reported genders.
		
		We recommend using this API in combination with the _Names API_."
    
    """
    url = f"https://detect4.p.rapidapi.com/detect/gender/"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "detect4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

