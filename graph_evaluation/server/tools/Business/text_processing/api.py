import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def detect_language(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detect the language of a natural language text."
    text: The text for which the language should be determined.
        
    """
    url = f"https://webknox-text-processing.p.rapidapi.com/text/language"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-text-processing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def detect_part_of_speech(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Tag a given text with part-of-speech tags (POS tags)."
    text: The text for which the POS tags should be found.
        
    """
    url = f"https://webknox-text-processing.p.rapidapi.com/text/posTags"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-text-processing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def extract_locations(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract names of locations in the given text."
    text: The text from which locations should be extracted.
        
    """
    url = f"https://webknox-text-processing.p.rapidapi.com/text/locations"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-text-processing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def extract_dates(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract dates from a given text."
    text: The text from which dates should be extracted.
        
    """
    url = f"https://webknox-text-processing.p.rapidapi.com/text/dates"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-text-processing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def extract_entities(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract entities such as movies or people in a given text."
    text: The text in which entities should be detected.
        
    """
    url = f"https://webknox-text-processing.p.rapidapi.com/text/entities"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-text-processing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def detect_errors(text: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detect spelling errors in the text."
    text: The text that should be auto-corrected.
        language: The language of the given text.
        
    """
    url = f"https://webknox-text-processing.p.rapidapi.com/text/correction"
    querystring = {'text': text, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-text-processing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def split_sentences(text: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Split a given text into sentences."
    text: The text to split into sentences.
        language: The language of the text. Currently English (en) and German (de) are supported.
        
    """
    url = f"https://webknox-text-processing.p.rapidapi.com/text/sentences"
    querystring = {'text': text, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-text-processing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def summarize_text(text: str, keepratio: str='0.5', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract a summary of a text."
    text: The text which should be summarized.
        keepratio: The ratio of the text that should be kept.
        
    """
    url = f"https://webknox-text-processing.p.rapidapi.com/text/summary"
    querystring = {'text': text, }
    if keepratio:
        querystring['keepRatio'] = keepratio
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-text-processing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def detect_sentiment(text: str, language: str, contextwords: str='John Hiatt', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detect the sentiment (positive or negative) of a given text. The detection works for English and German texts."
    text: The text for which the sentiment should be detected.
        language: The language of the given text (en=English, de=German).
        contextwords: Optionally consider only sentences that contain a context word for the sentiment detection.
        
    """
    url = f"https://webknox-text-processing.p.rapidapi.com/text/sentiment"
    querystring = {'text': text, 'language': language, }
    if contextwords:
        querystring['contextWords'] = contextwords
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-text-processing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

