import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_forex_article_details(requestid: str, uniquefxarticletoken: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides details for a given Forex article token.
		Sample request:
		
		    GET 
		    {        
		      /api/v1/ForexArticleDetails/3bbfaafd-dd32-4e79-a65f-50730b8ffeb1:AUDUSD-20201013
		    }"
    requestid: User Request Id - Ref
        uniquefxarticletoken: Unique article token
        
    """
    url = f"https://forex-news-alerts.p.rapidapi.com/api/v1/ForexArticleDetails/{requestid}/{uniquefxarticletoken}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forex-news-alerts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_latest_forex_news_updates(requestid: str, date: str, indexcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides a list of Forex News headlines for a given currency pair index and date. 
		
		Please contact us if you would like to request historical data which are older than a month.
		Sample request:
		
		    GET 
		    {        
		      /api/v1/ForexNewsUpdates/AUDUSD/20200921
		    }
		Valid Index Types:
		LatestNews,EURUSD,GBPUSD,USDJPY,AUDUSD,NZDUSD,USDCAD,USDCHF,EURGBP,Gold,SP500,DollarIndex,Commodities,Bonds,Equities"
    requestid: User Request Id - Ref
        date: Date
        indexcode: Forex Currency Code
        
    """
    url = f"https://forex-news-alerts.p.rapidapi.com/api/v1/ForexNewsUpdates/{requestid}/{indexcode}/{date}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forex-news-alerts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

