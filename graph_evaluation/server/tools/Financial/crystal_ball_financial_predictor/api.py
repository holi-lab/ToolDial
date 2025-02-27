import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def predictcrypto(predictionlength: int, cryptosymbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint deals with digital currencies.
		
		All predictions are based off of the daily closing price. 
		
		It returns json data with future "x" days closing prices and their respective timestamps  (EST)"
    predictionlength: How many days into the future you'd like to predict
        cryptosymbol: Crypto Symbol for the Coin  you'd like to predict [Example: BTC for Bitcoin or LTC for Litecoin]
        
    """
    url = f"https://crystal-ball-financial-predictor.p.rapidapi.com/predict_crypto"
    querystring = {'predictionLength': predictionlength, 'cryptoSymbol': cryptosymbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crystal-ball-financial-predictor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def predictstock(tickersymbol: str, predictionlength: int, stock: str='f', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint deals with traditional stock assets.
		
		All predictions are based off of the daily closing price. 
		
		It returns json data with future "x" days closing prices and their respective timestamps  (EST)"
    tickersymbol: Ticker for Stock you'd like to predict [Example: SPY for the S&P500 or APPL for Apple]
        predictionlength: How many days into the future you'd like to predict. [Max: 30]
        
    """
    url = f"https://crystal-ball-financial-predictor.p.rapidapi.com/predict_stock"
    querystring = {'tickerSymbol': tickersymbol, 'predictionLength': predictionlength, }
    if stock:
        querystring['stock'] = stock
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crystal-ball-financial-predictor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

