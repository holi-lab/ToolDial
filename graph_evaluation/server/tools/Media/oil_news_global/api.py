import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_individual_oil_company_news_outlet(endpoint: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets news/press-releases from an individual source, for example:
		/news/bbc  GETs news from BBC.com
		/aramco  GETs news and press-releases from Saudi Aramco"
    
    """
    url = f"https://oil-news-global.p.rapidapi.com/news/{endpoint}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "oil-news-global.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_oil_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns articles from the following endpoints:
		Endpoint => GETs Data From (Country of Origin if not US of GB)
		/oilprice => "news"
		/reuters => "energy"
		/yahoo => "finance"
		/bbc =>"news; topics; oil"
		/fox => "foxbusiness; oil"
		/cnbc => "energy"
		/cnn => "business"
		/guardian => "business"
		/nytimes => "energy-environment"
		/rt => "oil-prices-news-economy" (Russia)
		/rosneft => "news" (Russia)
		/tatneft => "news, press-releases" (Russia; Tatarstan)
		/aramco => "news" (Saudi Arabia)
		/aljazeera => "economy" (Qatar)"
    
    """
    url = f"https://oil-news-global.p.rapidapi.com/news"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "oil-news-global.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

