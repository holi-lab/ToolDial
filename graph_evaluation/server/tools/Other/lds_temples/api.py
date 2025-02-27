import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_information_by_temple(templenameid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return all data for a specific temple found on its detail page on The Church of Jesus Christ of Latter-day Saints website. For example, https://www.churchofjesuschrist.org/temples/details/aba-nigeria-temple?lang=eng
		
		Get the templeNameId from the temples endpoint."
    
    """
    url = f"https://lds-temples.p.rapidapi.com/temples/{templenameid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lds-temples.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_temples(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return data from all temples"
    
    """
    url = f"https://lds-temples.p.rapidapi.com/temples"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lds-temples.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

