import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v19_json(lookup: str, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get technology usage for a domain"
    lookup: The domain you want to lookup.

**Multi-Lookup Option**
When doing RAW domain lookups, you can parse up to 16 domains at once as a CSV. For example cnn.com,about.com,builtwith.com - this dramatically improves throughput.
        key: This is your key, use this for lookups. Get this key from https://api.builtwith.com
        
    """
    url = f"https://builtwith-domain.p.rapidapi.com/v19/api.json"
    querystring = {'LOOKUP': lookup, 'KEY': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "builtwith-domain.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

