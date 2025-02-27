import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def people_search(keywords: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of Linkedin profiles"
    
    """
    url = f"https://linkedin9.p.rapidapi.com/search_people"
    querystring = {'keywords': keywords, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linkedin9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def companies_search(keywords: str, schematype: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of LinkedIn companies"
    schematype: You can choose schemaType by your preference
        
    """
    url = f"https://linkedin9.p.rapidapi.com/search_companies"
    querystring = {'keywords': keywords, }
    if schematype:
        querystring['schemaType'] = schematype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linkedin9.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

