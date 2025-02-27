import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v2_api_call_php(url: str, filter: str, options: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Process Image from URL"
    url: URL of image to process.  Note: max size is 1500x1500 pix.
        filter: Which LunaPic image filter to use.  Full list of filters at http://lunapic.com/?filters
        options: Optional Parameters.  ie: keepcolors
        
    """
    url = f"https://lunapic-photo-effects.p.rapidapi.com/v2/api-call.php"
    querystring = {'url': url, 'filter': filter, }
    if options:
        querystring['options'] = options
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lunapic-photo-effects.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

