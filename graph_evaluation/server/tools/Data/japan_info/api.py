import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def request_data(indicatorcode: str, regioncode: str='01000', regionlevel: str='1', modifiedfrom: str='20001010', modifiedto: str='20201010', cycle: str='1', timefrom: str='20001000', valuecondition: str='>100', regionalrank: str='1', isseasonaladjustment: str='2', timeto: str='20201000', lang: str='JP', parentregioncode: str='00000', time: str='20101000', statname: str='人工', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API reference"
    
    """
    url = f"https://japan-info1.p.rapidapi.com/indicatorCode/{indicatorcode}"
    querystring = {}
    if regioncode:
        querystring['RegionCode'] = regioncode
    if regionlevel:
        querystring['RegionLevel'] = regionlevel
    if modifiedfrom:
        querystring['modifiedFrom'] = modifiedfrom
    if modifiedto:
        querystring['modifiedTo'] = modifiedto
    if cycle:
        querystring['Cycle'] = cycle
    if timefrom:
        querystring['TimeFrom'] = timefrom
    if valuecondition:
        querystring['ValueCondition'] = valuecondition
    if regionalrank:
        querystring['RegionalRank'] = regionalrank
    if isseasonaladjustment:
        querystring['IsSeasonalAdjustment'] = isseasonaladjustment
    if timeto:
        querystring['TimeTo'] = timeto
    if lang:
        querystring['Lang'] = lang
    if parentregioncode:
        querystring['ParentRegionCode'] = parentregioncode
    if time:
        querystring['Time'] = time
    if statname:
        querystring['StatName'] = statname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "japan-info1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

