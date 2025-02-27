import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_loremipsum(max_length: int=None, start_with_lorem_ipsum: str=None, random: str=None, paragraphs: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Lorem Ipsum API endpoint. Returns one or more paragraphs of lorem ipsum placeholder text."
    max_length: Maximum character length.
        start_with_lorem_ipsum: Whether to begin the text with the words \\\"Lorem ipsum\\\". Must be either true or false. If unset, a default value of true will be used.
        random: Whether to randomly generate paragraphs. Must be either true or false. If unset, a default value of true will be used.
        paragraphs: Number of paragraphs to generate. If unset, a default value of 1 will be used.
        
    """
    url = f"https://lorem-ipsum-by-api-ninjas.p.rapidapi.com/v1/loremipsum"
    querystring = {}
    if max_length:
        querystring['max_length'] = max_length
    if start_with_lorem_ipsum:
        querystring['start_with_lorem_ipsum'] = start_with_lorem_ipsum
    if random:
        querystring['random'] = random
    if paragraphs:
        querystring['paragraphs'] = paragraphs
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lorem-ipsum-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

