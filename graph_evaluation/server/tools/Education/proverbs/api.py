import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_proverb(includesources: str='rumi,emerson', excludesources: str=None, searchterm: str='love', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve one proverb and the source it came from.  There are three optional parameters - includeSources, excludeSources, and searchTerm.  If a value is provided in the includeSources parameter it is considered a comma separated list of sources.  The same goes for the excludeSources parameter.  The available sources to EITHER include OR exclude are Rumi, Emerson, Thoreau, Shakespeare, Tao Te Ching, or Proverbs.  If includeSources is populated then excludeSources is ignored.
		
		A third optional parameter called searchTerm can be used to make sure the resulting proverb contains that term.  Only letters, numbers, and spaces are allowed in the searchTerm parameter (other characters will be ignored).  All parameters are case insensitive."
    
    """
    url = f"https://proverbs.p.rapidapi.com/proverbs-api.php"
    querystring = {}
    if includesources:
        querystring['includeSources'] = includesources
    if excludesources:
        querystring['excludeSources'] = excludesources
    if searchterm:
        querystring['searchTerm'] = searchterm
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "proverbs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

