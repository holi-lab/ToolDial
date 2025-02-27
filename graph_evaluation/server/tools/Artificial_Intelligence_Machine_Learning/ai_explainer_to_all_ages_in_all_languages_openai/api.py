import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def explain_a_topic(prompt: str, language: str='English', age: int=6, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint:
		/explain
		
		HTTP Method:
		GET
		
		Description:
		Generates an explanation for a specified topic in a specified language and at a specified level of complexity suitable for a specified age group.
		
		Request Parameters:
		
		prompt (string): The topic that needs explanation.
		age (integer): The age of the target audience for the explanation, in years.
		language (string): The language in which the explanation should be provided.
		Response:
		The response is a JSON object that includes the generated explanation in the specified language.
		
		Note:
		Please use HTTP GET method to access the /explain endpoint."
    
    """
    url = f"https://ai-explainer-to-all-ages-in-all-languages-openai.p.rapidapi.com/explain"
    querystring = {'prompt': prompt, }
    if language:
        querystring['language'] = language
    if age:
        querystring['age'] = age
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-explainer-to-all-ages-in-all-languages-openai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

