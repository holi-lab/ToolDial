import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def all_companies_listed(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Currently, we have over 4000 companies that have been reviewed and assessed using the principles set out by Ben Graham in The Intelligent Investor.
		
		The return will include Ticker (GENERAL_CODE) and Company name (GENERAL_NAME). The ticker is used to find a company full report. Also, you can is if the company you are interested in, is part of the `database.`"
    
    """
    url = f"https://market-fluent-value-investing-research-uk.p.rapidapi.com/list/all"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "market-fluent-value-investing-research-uk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_uk_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This list contains all the UK companies we currently hold in our database. We update our database every month.
		
		The return will include Ticker (GENERAL_CODE) and Company name (GENERAL_NAME). The ticker is used to find a company full report. Also, you can is if the company you are interested in, is part of the database."
    
    """
    url = f"https://market-fluent-value-investing-research-uk.p.rapidapi.com/list/uk"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "market-fluent-value-investing-research-uk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def report_on_a_uk_company(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### How to use API overview
		
		Using the ticker (General Code) for a company, this endpoint will bring back all the information we hold on that business.
		
		This includes (but not limited to):
		
		- Sector and Industry
		
		- Description of their business
		
		- Website details
		
		- The number of years worth of data we have used in our model. This will be outlined as Growth Percentage (GP) Year.
			GP Year 1 with be for the last year reported i.e. 1 year.
			GP Year 2 will normally be for the last 3 years.
			GP Year 3 will be for the last 5 years.
			GP year 4 will be for the last 7 years.
			GP year 5 will be for the last 9 years.
			If any of these timeframes are different, the output from the API will be given a number, that is the number of years that we hold data on that company. Example GP Year 5 = 8, which means we hold data for the last 8 years and that is what the model has used.
		
		- An indication on growth percentage for the following areas of the business:
		
		  - Equity (%)
		  - Free Cash Flow (%)
		  - Revenue (%)
		  - Earnings per Share (EPS) (%)
		  - Return on Invested Capital (ROIC) (%)
		
		  Included is our Price to Earnings (PE) ratio and an average of the PE Ratios published online, using the most conservative figure to make our predictions and valuations.
		
		- Our algorithm takes all of the factors mentioned above and creates the following predictions:
		
		  - Future EPS in 10 years time (£)
		  - Future Share Price in 10 years time (£)
		  - A fair price for the share today (£)
		  - A margin of safety price, allowing for mistakes in the valuation (£)
		
		- The number of years debt when compared to the last know free cash flow is included for comparison of how much debt a company is holding.
		
		**To ensure the endpoint works correctly, you must put the companies ticker in capital letters**"
    
    """
    url = f"https://market-fluent-value-investing-research-uk.p.rapidapi.com/markets/uk/{ticker}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "market-fluent-value-investing-research-uk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

