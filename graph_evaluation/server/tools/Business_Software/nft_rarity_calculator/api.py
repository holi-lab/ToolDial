import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def calculate_scores_from_metadata_url(source: str, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The metadata.json should be presented as combined files of metadata."
    source: Provide the full source path of the metadata.json file
        limit: Use the `limit` parameter, to get a limited result or skip it to get full data.
        
    """
    url = f"https://nft-rarity-calculator.p.rapidapi.com/v1/api/scores"
    querystring = {'source': source, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nft-rarity-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

