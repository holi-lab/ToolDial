import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def esg_risk_rating(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The final ESG Risk Rating is a measure of unmanaged risk on a grading scale of A through F, with A signaling less ESG Risk. Controversy Risk is rated on a grading scale A through F, with F denoting the most serious controversy with the largest potential impact."
    
    """
    url = f"https://esg-risk-ratings-for-stocks.p.rapidapi.com/api/v1/resources/esg"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "esg-risk-ratings-for-stocks.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

