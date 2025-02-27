import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getesgnews(content_type: str=None, date: str=None, companyname: str='Apple Inc.', sedol: str=None, isin: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ESG news API with a state of the art algorithm tracking thousands of news sources, social media and NGOs in real-time. The results include ESG specific topics including SASB and UN SDGs and a few other important ESG themes such as Modern Slavery and Gender Equality."
    date: Please provide date in DD/MM/YYYY format.
        
    """
    url = f"https://gaialens-esg-news.p.rapidapi.com/news"
    querystring = {}
    if content_type:
        querystring['Content-Type'] = content_type
    if date:
        querystring['date'] = date
    if companyname:
        querystring['companyname'] = companyname
    if sedol:
        querystring['sedol'] = sedol
    if isin:
        querystring['isin'] = isin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gaialens-esg-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

