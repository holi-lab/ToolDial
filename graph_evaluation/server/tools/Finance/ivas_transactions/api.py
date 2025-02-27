import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_many(x_rapidapi_user: str, userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows users to fetch all their saved Transactions."
    x_rapidapi_user: This should match with your RapidAPI Username. This header is used to create a UNIQUE identifier in the database.
        userid: The Id of the user from YOUR system.

Default is 1.
        
    """
    url = f"https://ivas-transactions.p.rapidapi.com/transactions"
    querystring = {'X-RapidAPI-User': x_rapidapi_user, 'userId': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ivas-transactions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_summary(x_rapidapi_user: str, userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides a summary of the portfolio of the provided User."
    
    """
    url = f"https://ivas-transactions.p.rapidapi.com/portfolios"
    querystring = {'X-RapidAPI-User': x_rapidapi_user, 'userId': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ivas-transactions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

