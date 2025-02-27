import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_number_to_speech_provide_any_integer_number(number: int, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converting any integer number to Speech"
    number: Pass Only Integer Number!
        lang: Select Language , By default  ( en-us ) is supported, 
Hit GET Languages Endpoint to get a list of all supported languages.
        
    """
    url = f"https://all-purpose-complex-converter.p.rapidapi.com/numbers_to_speech/{number}/{lang}/"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-purpose-complex-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_text_to_speech_provide_any_text(text: str, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert Any Text To Speech."
    text: Support Both Text and Numbers. 
Can Pass Text Such as (\"any Thing to convert\")
or 
Can Pass Any Number such as 1029383833384474
        lang: Select Language , By default  ( en-us ) is supported, 
Hit GET Languages Endpoint to get a list of all supported languages.
        
    """
    url = f"https://all-purpose-complex-converter.p.rapidapi.com/text_to_speech/{text}/{lang}/"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-purpose-complex-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_numbers_to_words_provide_any_integer_number(number: int, to_convert: str='ordinal', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert any number to words."
    number: Pass only Integer Numbers! 
        to_convert: By Default the number will be converted to cardinal, if you wan to convert it into specific format such as ordinal, ordinal_num, year or currency than you can specify it here. 
        
    """
    url = f"https://all-purpose-complex-converter.p.rapidapi.com/numbers_to_words/{number}/{to_convert}"
    querystring = {}
    if to_convert:
        querystring['to_convert'] = to_convert
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-purpose-complex-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_languages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get All The Supported Languages."
    
    """
    url = f"https://all-purpose-complex-converter.p.rapidapi.com/get_languages/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-purpose-complex-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

