import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_fancy_settle_result(authorization: str, mid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Single bets with a score less than or equal to over_line will win if placed on over and lose if placed on under.
		Single bets with a score greater than or equal to under_line will lose if placed on over and win if placed on under.
		If VoidCase has a total of -404, all bets will be voided. This situation can only occur during resettlement.
		If the total of a bet is equal to the total of VoidCase, it will be voided.
		If the total of a bet is equal to CancelDt total, it will also be voided and adjusted for bets placed within the time interval."
    authorization: type bearer
        mid: market id
        
    """
    url = f"https://oddsbeta-fancy-api-cricket-all-sports.p.rapidapi.com/fancy/merchant/settle/result/"
    querystring = {'authorization': authorization, 'mid': mid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "oddsbeta-fancy-api-cricket-all-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fancy_re_settle_market(authorization: str, settle_dt_end: str=None, settle_dt_start: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The data appearing here are the markets that had a resettlement within the given time frame"
    authorization: type bearer
        settle_dt_end: settle_dt_end
        settle_dt_start: By default, it is 72 hours before the current time
        
    """
    url = f"https://oddsbeta-fancy-api-cricket-all-sports.p.rapidapi.com/fancy/merchant/re-settle/market/"
    querystring = {'authorization': authorization, }
    if settle_dt_end:
        querystring['settle_dt_end'] = settle_dt_end
    if settle_dt_start:
        querystring['settle_dt_start'] = settle_dt_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "oddsbeta-fancy-api-cricket-all-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

