import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def extract_open_graph_data(url: str, force: bool=None, tailgraph: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will scrape an URL and extract all Open Graph related information."
    force: If 1, forces refreshing the website URL (slower than returning the monthly cached data).
        tailgraph: If equals to 1, then the API will generate an Open Graph image using TailGraph's free API in case the website does not provide an image.
        
    """
    url = f"https://opengraphr.p.rapidapi.com/og"
    querystring = {'url': url, }
    if force:
        querystring['force'] = force
    if tailgraph:
        querystring['tailgraph'] = tailgraph
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opengraphr.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

