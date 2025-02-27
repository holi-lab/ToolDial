import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def collection(q: str, culture: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET /api/[culture]/collection gives the full collection with brief information about each work. This results are split up in result pages. By using the p and ps parameters you can fetch more results. All of the other parameters are identical to the search page on the Rijksmuseum website. You can use that to find out what's the best query to use"
    q: The search terms that need to occur in one of the fields of the artwork data
        culture: nl / en		The language to search in (and of the results)
        
    """
    url = f"https://community-rijksmuseum.p.rapidapi.com/{culture}/collection"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-rijksmuseum.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

