import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_stocks_by_mentions(filter: str, pagenum: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List stocks by mentions given a filter.
		
		filters:
		all  # All subreddits combined
		all-stocks #  Only subreddits focusing on stocks such as r/wallstreetbets or r/stocks
		all-crypto #  Only subreddits focusing on cryptocurrencies such as r/CryptoCurrency or r/SatoshiStreetBets
		
		CryptoCurrency
		CryptoCurrencies
		Bitcoin
		SatoshiStreetBets
		CryptoMoonShots
		CryptoMarkets
		stocks
		wallstreetbets
		options
		WallStreetbetsELITE
		Wallstreetbetsnew
		SPACs
		investing
		Daytrading
		
		Each page contains 100 results."
    
    """
    url = f"https://reddit-stock-and-crypto-sentiment-tracker.p.rapidapi.com/filter/{filter}/page/{pagenum}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reddit-stock-and-crypto-sentiment-tracker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

