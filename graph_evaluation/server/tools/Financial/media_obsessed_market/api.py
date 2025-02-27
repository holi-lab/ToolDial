import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def predict(stop_loss: int=0, shares2buy: int=10, sell_same_day: int=1, integrate_nn_results: int=1, percent_of_change: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives a list of companies that are predicted to move the specified percentage up within daily trading. Allowed percentages are 0.4%(0.004) . Use with no optional parameters to get full data back. It is not large.
		Verify the performance using predict_evaluation endpoint or on the website mbaassist.co (navigate to Dashboards)"
    
    """
    url = f"https://media-obsessed-market.p.rapidapi.com/predict/"
    querystring = {}
    if stop_loss:
        querystring['stop_loss'] = stop_loss
    if shares2buy:
        querystring['shares2buy'] = shares2buy
    if sell_same_day:
        querystring['sell_same_day'] = sell_same_day
    if integrate_nn_results:
        querystring['integrate_NN_results'] = integrate_nn_results
    if percent_of_change:
        querystring['percent_of_change'] = percent_of_change
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "media-obsessed-market.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transactions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Transactions list"
    
    """
    url = f"https://media-obsessed-market.p.rapidapi.com/transaction/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "media-obsessed-market.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def predict_evaluation(integrate_nn_results: int=1, days_ago_started: int=30, percent_of_change: int=0, sell_same_day: str='True', stop_loss: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides evaluation of using predictions for up to 60 days earlier"
    
    """
    url = f"https://media-obsessed-market.p.rapidapi.com/results/"
    querystring = {}
    if integrate_nn_results:
        querystring['integrate_NN_results'] = integrate_nn_results
    if days_ago_started:
        querystring['days_ago_started'] = days_ago_started
    if percent_of_change:
        querystring['percent_of_change'] = percent_of_change
    if sell_same_day:
        querystring['sell_same_day'] = sell_same_day
    if stop_loss:
        querystring['stop_loss'] = stop_loss
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "media-obsessed-market.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

