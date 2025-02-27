import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def batch(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "batch process to retrieve payout transaction data"
    
    """
    url = f"https://sigue-payout.p.rapidapi.com/payout/transactions/batch/query?agentId=7737&destinationPaymentMethodCode=wp&queryType=Nr&maxTransactions=5"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sigue-payout.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

