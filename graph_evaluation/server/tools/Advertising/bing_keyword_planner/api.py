import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def url_seed(target_domain: str, location_name: str='United States', device: str='all', language_name: str='English', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Based on your provided keywords, this endpoint will return up to 3000 suitable keyword recommendations. Set up to 200 keywords and obtain the results that Bing Ads suggests for your query. Using this tool, you may obtain up to 3000 keyword ideas."
    target_domain: The domain name of the target website.
        location_name: Name of location.

Can be a Country, Region or City. 

Examples: \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"United States\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"Germany\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"London,England,United Kingdom\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
        device: Limits data for a particular device type.

Possible values: *all*, *mobile*, *desktop*, *tablet*
Default value: *all*
        language_name: Name of language.

Examples: \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"English\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"German\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\", \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"Spanish\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
        
    """
    url = f"https://bing-keyword-planner.p.rapidapi.com/urlseed"
    querystring = {'target_domain': target_domain, }
    if location_name:
        querystring['location_name'] = location_name
    if device:
        querystring['device'] = device
    if language_name:
        querystring['language_name'] = language_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bing-keyword-planner.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keyword_ideas(keywords: str, device: str='all', location_name: str='United States', language_name: str='English', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Based on your provided keywords, this endpoint will return up to 3000 suitable keyword recommendations. Set up to 200 keywords and obtain the results that Bing Ads suggests for your query. Using this tool, you may obtain up to 3000 keyword ideas."
    keywords: Comma-separated list of input keywords.

Maximum number of input keywords: 200
        device: Limits data for a particular device type.

Possible values: *all*, *mobile*, *desktop*, *tablet*
Default value: *all*
        location_name: Name of location.

Can be a Country, Region or City. 

Examples: \\\\\\\"United States\\\\\\\", \\\\\\\"Germany\\\\\\\", \\\\\\\"London,England,United Kingdom\\\\\\\"
        language_name: Name of language.

Examples: \\\\\\\"English\\\\\\\", \\\\\\\"German\\\\\\\", \\\\\\\"Spanish\\\\\\\"
        
    """
    url = f"https://bing-keyword-planner.p.rapidapi.com/keywordideas"
    querystring = {'keywords': keywords, }
    if device:
        querystring['device'] = device
    if location_name:
        querystring['location_name'] = location_name
    if language_name:
        querystring['language_name'] = language_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bing-keyword-planner.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

