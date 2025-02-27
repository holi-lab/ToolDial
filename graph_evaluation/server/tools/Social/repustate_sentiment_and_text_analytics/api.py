import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def clean_html(url: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API call will extract out the most important part of a web page, removing all tags and any common header or footer content. For those familiar with Readability.js, this API call replicates Readability's functionality. You can also pass in a URL to a PDF and obtain just the text from this PDF.  If the article has a main image, the URL to this image will also be returned."
    url: Any valid URL. If the URL specified is a URL-shortened one (e.g. bit.ly), Repustate will follow the redirects properly until the final page is found.
        
    """
    url = f"https://repustate.p.rapidapi.com/v2/{apikey}/clean-html.json"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "repustate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

