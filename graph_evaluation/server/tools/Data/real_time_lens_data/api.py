import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def object_detection(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detect landmarks, places, plants, animals, products, and other objects in an image, including bounding boxes, labels / types and confidence score."
    url: URL of an image to perform Google Lens object detection.
        
    """
    url = f"https://real-time-lens-data.p.rapidapi.com/object-detection"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-lens-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_to_text_ocr(url: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract text from an image and get paragraph, sentence and word level text detections from Google Lens."
    url: URL of an image from which to extract text.
        language: Set the language of the results.

**Allowed values:** 2-letter language code, see [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
**Default:** en.
        
    """
    url = f"https://real-time-lens-data.p.rapidapi.com/ocr"
    querystring = {'url': url, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-lens-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_search(url: str, country: str='us', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search image by url and get visual matches, knowledge graph, products and prices, text and object detections. See it in on [Google Lens](https://lens.google.com/uploadbyurl?url=https%3A%2F%2Fi.imgur.com%2FHBrB8p0.png)."
    url: URL of an image to perform Google Lens search.
        country: Set the country for the search.

**Allowed values:** 2-letter country code, see [ISO 3166](https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes).
**Default:** us.
        language: Set the language of the results (Google's **hl** parameter).

**Allowed values:** 2-letter language code, see [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
**Default:** en.
        
    """
    url = f"https://real-time-lens-data.p.rapidapi.com/search"
    querystring = {'url': url, }
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-lens-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_image_sources(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send an image URL and get the top 400 source web pages (Reverse Image Search).
		
		**Note:** Requests for this endpoint will be charged 2x (as two requests)."
    url: URL of the image for which to find sources
        
    """
    url = f"https://real-time-lens-data.p.rapidapi.com/find-image-sources"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-lens-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

