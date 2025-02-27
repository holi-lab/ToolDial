import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_list_of_definitions(hashtag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the top definitions, currently max 5. Ordered by popularity. This popularity is based on user votes, but is weighed using an intern algorithm on Tagdef.com"
    hashtag: The name of the hashtag
        
    """
    url = f"https://tagdef.p.rapidapi.com/{hashtag}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tagdef.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_top_definition(hashtag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the top definition for given hashtag. The top definition is calculated based on user votes, and weighed using an intern algorithm on Tagdef.com"
    hashtag: The name of the hashtag
        
    """
    url = f"https://tagdef.p.rapidapi.com/one.{hashtag}.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tagdef.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

