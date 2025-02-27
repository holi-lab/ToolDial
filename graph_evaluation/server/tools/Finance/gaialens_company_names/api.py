import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getcompanynames(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns a list of all company names available to the user. It can be used as a pre-step before invoking other GaiaLens APIs such as GaiaLens ESG Scores, Gaialens Historical ESG Scores and GaiaLens ESG News."
    
    """
    url = f"https://gaialens-company-names.p.rapidapi.com/companynames"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gaialens-company-names.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

