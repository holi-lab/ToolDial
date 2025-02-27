import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def create_an_ar_code(text: str, backgroundheight: int, backgroundcolor: str, backgroundwidth: int, api_key: str, textcolor: str, textsize: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To create an AR Code please fill all the parameters.
		
		To edit this AR Code later with a premium plan keep its referenceID."
    text: Text to display
        backgroundheight: Plane behind the text height (Example: 2)
        backgroundcolor: Plane behind the text Hex color (Example: FFFFFF)
        backgroundwidth: Plane behind the text width (Example: 4)
        api_key: Get your API key for FREE on : https://ar-code.com/
        textcolor: Hex text color (Example: ff0000)
        textsize: Text size (Example: 8)
        
    """
    url = f"https://ar-code-augmented-reality-codes-generator.p.rapidapi.com/arcode_api"
    querystring = {'text': text, 'backgroundheight': backgroundheight, 'backgroundcolor': backgroundcolor, 'backgroundwidth': backgroundwidth, 'api_key': api_key, 'textcolor': textcolor, 'textsize': textsize, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ar-code-augmented-reality-codes-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scans_statistics(api_key: str, referenceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scans number for an AR Code with its referenceID and password."
    api_key: Get your API key for FREE on : https://ar-code.com/
        
    """
    url = f"https://ar-code-augmented-reality-codes-generator.p.rapidapi.com/arcode_stats_api"
    querystring = {'api_key': api_key, 'referenceid': referenceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ar-code-augmented-reality-codes-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def edit_an_ar_code(textcolor: str, text: str, backgroundwidth: int, backgroundheight: int, referenceid: str, api_key: str, textsize: int, backgroundcolor: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To edit an AR Code please indicate its referenceID.
		
		Editable data: text, textcolor, textsize, backgroundwidth, backgroundheight, backgroundcolor"
    textcolor: Hex text color (Example: ff0000)
        text: Text to display
        backgroundwidth: Plane behind the text width (Example: 4)
        backgroundheight: Plane behind the text height (Example: 2)
        referenceid: To edit an AR Code please indicate its referenceid.
        api_key: Get your API key for FREE on : https://ar-code.com/
        textsize: Text size (Example: 8)
        backgroundcolor: Plane behind the text Hex color (Example: FFFFFF)
        
    """
    url = f"https://ar-code-augmented-reality-codes-generator.p.rapidapi.com/arcode_edit_api"
    querystring = {'textcolor': textcolor, 'text': text, 'backgroundwidth': backgroundwidth, 'backgroundheight': backgroundheight, 'referenceid': referenceid, 'api_key': api_key, 'textsize': textsize, 'backgroundcolor': backgroundcolor, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ar-code-augmented-reality-codes-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

