import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def clean_text(text: str, models: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is Safe-Text main API. 
		
		Use the **clean_text** to make cleansing operations for supported languages .
		It only accept **GET** action and it is as simple as:
		`/api/clean_text?text=hello%20world&models=FixMojibak,Punctuate` To run the *FixMojibak* and *Punctuate* models for instance.
		
		**List of available models**
		
		> "FixHTML", "Linkify", "FixMojibak", "Punctuate", "Decancer", "BadWords", "Sensitive", "StripTags", "DetectLanguage"
		
		You can pass a subset of these models. Please not that models are executed always in the same order.
		
		Example:
		`curl -X 'GET'   'https://ns514514.ip-142-4-215.net/api/clean_text?text=hello%20world&models=Punctuate,FixMojibak'`
		
		Please pass models simply as in the example. We couldn't have this format ( array of enum) working using the RapidAPI UI."
    
    """
    url = f"https://safe-text.p.rapidapi.com/clean_text"
    querystring = {'text': text, 'models': models, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "safe-text.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def serve_health(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get server status from every once in a while. This way you don't pay for a dead endpoint :)
		You parse results to can check:
		
		- Current version 
		- Deployed models
		- Supported languages"
    
    """
    url = f"https://safe-text.p.rapidapi.com/meta"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "safe-text.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

