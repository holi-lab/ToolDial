import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def whois_history_v1(domainname: str, createddateto: str=None, createddatefrom: str=None, outputformat: str=None, expireddateto: str=None, updateddateto: str=None, mode: str=None, updateddatefrom: str=None, expireddatefrom: str=None, sincedate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check Whois  record's history"
    domainname: The domain for which historic WHOIS data is requested.
        createddateto: If present, search through domains created before the given date. Acceptable values: date in "YYYY-MM-DD" format.
        createddatefrom: If present, search through domains created after the given date. Acceptable values: date in "YYYY-MM-DD" format.
        outputformat: Response output format. Acceptable values: XML or JSON. Defaults to JSON.
        expireddateto: If present, search through domains expired before the given date. Acceptable values: date in "YYYY-MM-DD" format.
        updateddateto: If present, search through domains updated before the given date. Acceptable values: date in "YYYY-MM-DD" format.
        mode: preview – return only the number of domains. No credits deducted. purchase – return the actual list of domains (limited to 10,000). 1 request costs 50 DRS credits. Default: preview
        updateddatefrom: If present, search through domains updated after the given date. Acceptable values: date in "YYYY-MM-DD" format.
        expireddatefrom: If present, search through domains expired after the given date. Acceptable values: date in "YYYY-MM-DD" format.
        sincedate: If present, search through activities discovered since the given date. Sometimes there is a latency between the actual added/renewal/expired date and the date when our system detected this change. Acceptable values: date in "YYYY-MM-DD" format.
        
    """
    url = f"https://whois-history1.p.rapidapi.com/api/v1"
    querystring = {'domainName': domainname, }
    if createddateto:
        querystring['createdDateTo'] = createddateto
    if createddatefrom:
        querystring['createdDateFrom'] = createddatefrom
    if outputformat:
        querystring['outputFormat'] = outputformat
    if expireddateto:
        querystring['expiredDateTo'] = expireddateto
    if updateddateto:
        querystring['updatedDateTo'] = updateddateto
    if mode:
        querystring['mode'] = mode
    if updateddatefrom:
        querystring['updatedDateFrom'] = updateddatefrom
    if expireddatefrom:
        querystring['expiredDateFrom'] = expireddatefrom
    if sincedate:
        querystring['sinceDate'] = sincedate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whois-history1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

