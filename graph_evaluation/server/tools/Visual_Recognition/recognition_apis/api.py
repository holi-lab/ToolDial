import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def recognize_objects_by_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "HTTP GET API which recognizes objects inside an image provided by URL value in the query string."
    
    """
    url = f"https://recognition-apis1.p.rapidapi.com/objects/vbeta"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "recognition-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recognize_plate_by_url(url: str, language: str=None, ocrengine: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "HTTP GET API which recognizes vehicle plate standards inside an image provided by URL value in the query string."
    url: An image containing the vehicle plate which wants to be identified.
        language: Select the language you want OCR uses to analyze the image content. By default the English ('eng') language is selected.
        ocrengine: Select the engine you want for OCR engine. By default the 'TesseractLstmCombined' engine is selected.
        
    """
    url = f"https://recognition-apis1.p.rapidapi.com/plate/vbeta"
    querystring = {'url': url, }
    if language:
        querystring['language'] = language
    if ocrengine:
        querystring['ocrEngine'] = ocrengine
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "recognition-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recognize_text_by_url(url: str, ocroutput: str=None, language: str=None, ocrengine: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "HTTP GET API which recognizes text standards inside an image provided by URL value in the query string."
    url: An text image which wants to be identified.
        ocroutput: Select the OCR output text type you want. By default the 'UTF8Text' type is selected.
        language: Select the language you want OCR uses to analyze the image content. By default the English ('eng') language is selected.
        ocrengine: Select the engine you want for OCR engine. By default the 'TesseractLstmCombined' engine is selected.
        
    """
    url = f"https://recognition-apis1.p.rapidapi.com/text/vbeta"
    querystring = {'url': url, }
    if ocroutput:
        querystring['ocrOutput'] = ocroutput
    if language:
        querystring['language'] = language
    if ocrengine:
        querystring['ocrEngine'] = ocrengine
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "recognition-apis1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

