import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_record_details(href: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the record details for a given href which will be returned from the "Get Records" request
		
		
		Returns details like who/what/where/when and an array of strings indicating the different paragraphs as displayed in guinnessworldrecords.com"
    
    """
    url = f"https://guinness-world-records-api.p.rapidapi.com/guinness/recordDetails"
    querystring = {'href': href, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "guinness-world-records-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_records(term: str, pagenum: int=1, maxresults: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Guinness World Records for a given term"
    
    """
    url = f"https://guinness-world-records-api.p.rapidapi.com/guinness/records/{term}"
    querystring = {}
    if pagenum:
        querystring['pageNum'] = pagenum
    if maxresults:
        querystring['maxResults'] = maxresults
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "guinness-world-records-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

