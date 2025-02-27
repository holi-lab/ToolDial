import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def elec_ev_station_for_ca(orderby: str, print: str='"pretty"', limittofirst: int=5, equalto: str='"Airdrie"', endat: str=None, startat: str=None, limittolast: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find electric vehicle charging stations in the United States"
    orderby: you can  **orderBy** city name or postcode 
all STRING in Tutorials:
[Tutorials:Order By](https://rapidapi.com/zoal21301/api/electric-vehicle-charging-station-and-point/tutorials/order-by)
        
    """
    url = f"https://electric-vehicle-charging-station-and-point.p.rapidapi.com/ca/elec.json"
    querystring = {'orderBy': orderby, }
    if print:
        querystring['print'] = print
    if limittofirst:
        querystring['limitToFirst'] = limittofirst
    if equalto:
        querystring['equalTo'] = equalto
    if endat:
        querystring['endAt'] = endat
    if startat:
        querystring['startAt'] = startat
    if limittolast:
        querystring['limitToLast'] = limittolast
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "electric-vehicle-charging-station-and-point.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def elec_ev_station_for_us(orderby: str, equalto: str='"Sun Valley"', limittolast: int=None, startat: str=None, endat: str=None, limittofirst: int=5, print: str='"pretty"', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find electric vehicle charging stations in the United States"
    orderby: you can  **orderBy** city name or postcode 
all STRING in Tutorials:
[Tutorials:Order By](https://rapidapi.com/zoal21301/api/electric-vehicle-charging-station-and-point/tutorials/order-by)
        
    """
    url = f"https://electric-vehicle-charging-station-and-point.p.rapidapi.com/us/elec.json"
    querystring = {'orderBy': orderby, }
    if equalto:
        querystring['equalTo'] = equalto
    if limittolast:
        querystring['limitToLast'] = limittolast
    if startat:
        querystring['startAt'] = startat
    if endat:
        querystring['endAt'] = endat
    if limittofirst:
        querystring['limitToFirst'] = limittofirst
    if print:
        querystring['print'] = print
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "electric-vehicle-charging-station-and-point.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

