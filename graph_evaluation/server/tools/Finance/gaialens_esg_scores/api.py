import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getesgscores(isin: str=None, sedol: str=None, companyname: str='Apple Inc.', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API provides real-time Environmental, Social, Governance and Overall scores for companies on a scale of 0 to 100. In addition to this, the API also provides other relevant metrics like Global Rank, Industry Rank and more."
    
    """
    url = f"https://gaialens-esg-scores.p.rapidapi.com/scores"
    querystring = {}
    if isin:
        querystring['isin'] = isin
    if sedol:
        querystring['sedol'] = sedol
    if companyname:
        querystring['companyname'] = companyname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gaialens-esg-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

