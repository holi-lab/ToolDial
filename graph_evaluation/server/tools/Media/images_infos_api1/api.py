import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getwebsitesscreenshot(url: str, fullpage: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Capture a screenshot from website"
    
    """
    url = f"https://images-infos-api1.p.rapidapi.com/getWebsitesScreenshot"
    querystring = {'url': url, }
    if fullpage:
        querystring['fullpage'] = fullpage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "images-infos-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwebsitespdf(url: str, fullpage: str='true', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a PDF from a website"
    
    """
    url = f"https://images-infos-api1.p.rapidapi.com/getWebsitesPDF"
    querystring = {'url': url, }
    if fullpage:
        querystring['fullpage'] = fullpage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "images-infos-api1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

