import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_by_abn(searchstring: int, authenticationguid: str, includehistoricaldetails: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the Australian company register by Australian Business Number (ABN)"
    
    """
    url = f"https://australian-company-data.p.rapidapi.com/ABRXMLSearch/AbrXmlSearch.asmx/SearchByABNv202001"
    querystring = {'searchString': searchstring, 'authenticationGuid': authenticationguid, 'includeHistoricalDetails': includehistoricaldetails, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "australian-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

