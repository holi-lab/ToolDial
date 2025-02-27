import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_your_all_qr_code_list(limit: int=5, sort: int=1, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all custom Visual QR code with this API. This will return all the campaign of Dynamic QR which you can easily track for visitors."
    limit: Query limit for No. of campaign per page (Default 5)
        sort: List all campaign by Date(1) or Name(2)
        page: Request Query campaign by page (Default 1)
        
    """
    url = f"https://qrzebra-qr-ze-v1.p.rapidapi.com/campaign/"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qrzebra-qr-ze-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_track_data(period: str='month', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Track your Custom QR code Data by No. Of scans, Country, City, Device type"
    period: GET Tracking Data by “day” (Day viz data of current day/ today) “week” (Day viz data of current week) “month” (Day viz data of current month) “months” (Month viz data of the current year) “year” (Yearly data)
        
    """
    url = f"https://qrzebra-qr-ze-v1.p.rapidapi.com/data/qrId"
    querystring = {}
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qrzebra-qr-ze-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

