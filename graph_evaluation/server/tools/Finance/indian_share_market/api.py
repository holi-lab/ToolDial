import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_stock_data_of_category_specified(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Indian Share Market categorized into 
		trending_stocks, 52_week_high, 52_week_low, most_active_stocks, top_stock_gainers, top_stock_losers
		use these values for category parameter in query"
    
    """
    url = f"https://indian-share-market.p.rapidapi.com/s/new/"
    querystring = {'category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indian-share-market.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

