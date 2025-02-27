import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_infringement_v1_infringement_research_predict_get(patent_number: str, text: str, model_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create a request item with the following information:
		
		- **text**: could be the description of a company asset or a product explanation. It should be minimum length of `20` words for better performance. Note that GET queries are limited to `2048` characters (_required_).
		- **patent_number**: should start with a country code and end with the _Kind_ _Code_. You can enter as many as `5` patent numbers, each separated with a comma (_required_).
		- **model_name**: is the name of the model empowering Stacks Patent Similarity engine. Defaults to `stk_nova`."
    
    """
    url = f"https://stacks-patent-similarity1.p.rapidapi.com/v1/infringement-research/predict"
    querystring = {'patent_number': patent_number, 'text': text, }
    if model_name:
        querystring['model_name'] = model_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stacks-patent-similarity1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

