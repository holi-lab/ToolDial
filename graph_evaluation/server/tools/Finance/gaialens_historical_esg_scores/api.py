import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gethistoricalscoresbyyear(year: str, content_type: str=None, sedol: str=None, isin: str=None, companyname: str='Apple Inc.', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns historical Environmental, Social, Governance and Overall scores for companies on a scale of 0 to 100 based on the year provided in the request.  In addition to this, the API also provides other relevant metrics like Global Rank, Industry Rank and more."
    year: <= 2020
        
    """
    url = f"https://gaialens-historical-esg-scores.p.rapidapi.com/scores/historical/{year}"
    querystring = {}
    if content_type:
        querystring['Content-Type'] = content_type
    if sedol:
        querystring['sedol'] = sedol
    if isin:
        querystring['isin'] = isin
    if companyname:
        querystring['companyname'] = companyname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gaialens-historical-esg-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

