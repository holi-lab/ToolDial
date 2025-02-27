import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def financial_statement_summarization(frequency: str, financial_statement: str, ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint leverages GPT-3.5 engine to summarize the latest financial statements of the company entered by the user. The users can enter any US Stock Ticker Symbol along with the financial statement they would like to analyze - balance_sheet, income_statement or cashflow. Within a few seconds the endpoint will generate a quick summary for the user to give them an overview of the company's financial positioning and health."
    
    """
    url = f"https://altapi1.p.rapidapi.com/summarize-financial-statements/{ticker}/{financial_statement}/{frequency}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "altapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def social_sentiment_analysis(asset: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides the sentiment score along with the most popular URL page for the company for the current date. It requires the user to put in the company name as the parameter.
		
		It may happen that no results are provided, this will be because the company you've searched for either is not much talked about in the news or other social media platforms or there might not be any related news for that company on that specific date."
    asset: Enter the asset name you want the sentiment analysis for: 
        
    """
    url = f"https://altapi1.p.rapidapi.com/sentiment/{asset}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "altapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insider_trades(us_stock_ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The **insider-ratings endpoint** will retrieve the trades made by insider participants of a company. The required parameter is a **US Ticker Symbol**."
    us_stock_ticker: Enter the TICKER of the company: 
        
    """
    url = f"https://altapi1.p.rapidapi.com/insider-trades/{us_stock_ticker}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "altapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def esg_scores(us_stock_ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ESG score stands for a company's Enviornmental, Social and Governance score. It is one of the most recent Alternative Data. This GET request provides a JSON response which includes :
		1. Overall ESG Score
		2. Governance Score
		3. Environmental Score
		4. Social Score
		The user is provided to input a company ticker as the parameter."
    us_stock_ticker: Enter the TICKER of the company: 
        
    """
    url = f"https://altapi1.p.rapidapi.com/esg/{us_stock_ticker}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "altapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_playstore_app_reviews_analysis(company: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint enables users to analyse the reviews of an app on Google PlayStore. This endpoint uses textual emotional classification to differentiate between reviews that are negative and positive and also labels them with certain positive and negative scores."
    
    """
    url = f"https://altapi1.p.rapidapi.com/playstore-reviews-analysis/{company}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "altapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

