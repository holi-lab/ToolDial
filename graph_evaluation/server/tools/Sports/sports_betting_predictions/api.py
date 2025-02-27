import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_sports_predictions(sport_id: int, date: str, competition: str=None, country: str=None, market: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to get predictions. This endpoint provides different filters passed as POST Form parameters. This API requires you to supply sport_id use Get Sports endpoint to get a list of available sport IDs"
    sport_id: Filter by Sport
        date: Filter by Date, date format should be YYYY-MM-DD
        competition: Specify competition or leave blank to get prediction for all competitions
        country: Specify country or leave blank to get prediction for all countries. You can also provide multiple countries by separating them with comma ( , ) e.g ENG,ITA 

        market: Specify market or leave blank to get prediction for all markets

        
    """
    url = f"https://sports-betting-predictions.p.rapidapi.com/v1/prediction"
    querystring = {'sport_id': sport_id, 'date': date, }
    if competition:
        querystring['competition'] = competition
    if country:
        querystring['country'] = country
    if market:
        querystring['market'] = market
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-betting-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_soccer_3_way_predictions(country: str=None, competition: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to get predictions. This endpoint provides different filters passed as POST Form parameters. This API retrieves ONLY predictions for 3 Way Soccer Sport"
    country: Specify country or leave blank to get prediction for all countries. You can also provide multiple countries by separating them with comma ( , ) e.g ENG,ITA 

        competition: Specify competition or leave blank to get prediction for all competition

        
    """
    url = f"https://sports-betting-predictions.p.rapidapi.com/v3/soccer/prediction"
    querystring = {}
    if country:
        querystring['country'] = country
    if competition:
        querystring['competition'] = competition
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-betting-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_soccer_prediction(date: str, market: str=None, competition: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get prediction for soccer sports. This endpoint allows you get prediction for soccer ONLY"
    date: Specify date in the format YYYY-MM-DD
        market: Filter prediction by market, to get a list of available markets use GetMarket endpoint
        competition: Filter prediction by competition
        country: Filter prediction by Country, to get a list of available countries use GetCountry endpoint. 
You can also provide multiple countries by separating them with comma ( , ) e.g ENG,ITA 
        
    """
    url = f"https://sports-betting-predictions.p.rapidapi.com/v2/soccer/prediction"
    querystring = {'date': date, }
    if market:
        querystring['market'] = market
    if competition:
        querystring['competition'] = competition
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-betting-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfixtures(date: str, sport_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoints get all the  fixture information. The endpoint provides the following filters"
    date: Filter fixture by date, allowed date format YYYY-MM-DD
        sport_id: Filter by Sport
        
    """
    url = f"https://sports-betting-predictions.p.rapidapi.com/v1/fixture"
    querystring = {'date': date, 'sport_id': sport_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-betting-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmarkets(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves all the available markets, use the returned market name  to filter predictions by market"
    
    """
    url = f"https://sports-betting-predictions.p.rapidapi.com/v1/markets"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-betting-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcountries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve all active countries. You can use the returned country names to filter predictions by country"
    
    """
    url = f"https://sports-betting-predictions.p.rapidapi.com/v1/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-betting-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstatistics(sport_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoints get all the  statistics information."
    sport_id: Filter statistics by Sport
        
    """
    url = f"https://sports-betting-predictions.p.rapidapi.com/v1/statistics"
    querystring = {'sport_id': sport_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-betting-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsports(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returned a list of all active sports, the return payload contains sport_id which should be used to filter prediction by sport. Every sport has a unique identifier labelled sport_id"
    
    """
    url = f"https://sports-betting-predictions.p.rapidapi.com/v1/sports"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-betting-predictions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

