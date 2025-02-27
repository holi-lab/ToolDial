import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def dashboard(updatefrequency: str, updatefrequencyunit: str, region: str, role: str, externalid: str, dashboard: str, infolinktext: str=None, infolink: str=None, periodthree: str=None, forceupdate: bool=None, periodone: str=None, periodtwo: str=None, companyname: str=None, companyurl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get private signed url to dashboard"
    updatefrequency: Update Frequency, combined with updatefrequencyunit, determines how often data in the dashboard is refreshed)
        updatefrequencyunit: Time Unit for the updatefrequency parameter
        region: The region in which the dashboard that you want to expose is defined in.
        role: The IAM role to be assumed when reading Dashboard and Metrics
        externalid: External ID used to Assume access to your IAM role
        dashboard: The name of the Cloudwatch Dashboard
        infolinktext: The text to be used for the info link on the dashboard
        infolink: The link to be used when clicking the info text on the dashboard
        periodthree: The third of the three time periods each metric on the dashboard is displayed for.
        forceupdate: Update even if according to the updatefrequency, and update is not required
        periodone: The first of the three time periods each metric on the dashboard is displayed for. 
        periodtwo: The second of the three time periods each metric on the dashboard is displayed for.
        companyname: Provide the name of your company to be displayed on the dashboard
        companyurl: The link to be used when clicking the company on the Dashboard
        
    """
    url = f"https://publiccloudwatchdashboard.p.rapidapi.com/dashboard"
    querystring = {'updatefrequency': updatefrequency, 'updatefrequencyunit': updatefrequencyunit, 'region': region, 'role': role, 'externalid': externalid, 'dashboard': dashboard, }
    if infolinktext:
        querystring['infolinktext'] = infolinktext
    if infolink:
        querystring['infolink'] = infolink
    if periodthree:
        querystring['periodthree'] = periodthree
    if forceupdate:
        querystring['forceupdate'] = forceupdate
    if periodone:
        querystring['periodone'] = periodone
    if periodtwo:
        querystring['periodtwo'] = periodtwo
    if companyname:
        querystring['companyname'] = companyname
    if companyurl:
        querystring['companyurl'] = companyurl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "publiccloudwatchdashboard.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

