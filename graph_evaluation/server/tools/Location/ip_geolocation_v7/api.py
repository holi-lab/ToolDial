import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def geo_location_endpoint(fields: str=None, ips: str=None, callback: str=None, output: str=None, ip: str='5.197.252.85', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get geolocation endpoint"
    fields: To filter response fields.
Example query: `/ipgeo?fields=country_name,country_code2`

Comma separated fields to query only the requested fields. You would get all the fields if you don't specify this parameter.
        ips: To query bulk IPs.
Example query: `/ipgeo?ips[]=5.197.252.85&ips[]=5.197.252.86`
Example query: `/ipgeo?ips=[\"5.197.252.85\",\"5.197.252.86\"]`
        callback: To specify the JSONP callback namespace.
        output: To specify the output format. 
Options are: **json** or **xml**. 
Default: **json**
        ip: To query for specific IP address.
Default: **Visitor IP / Server IP which the request sent from.**

For client side use you don't need to send this parameter, it means the visitor IP would be queried. But for server side use you need to send this parameter as visitor IP, otherwise it would query for server IP which the request is sent from.
        
    """
    url = f"https://ip-geolocation16.p.rapidapi.com/ipgeo"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if ips:
        querystring['ips'] = ips
    if callback:
        querystring['callback'] = callback
    if output:
        querystring['output'] = output
    if ip:
        querystring['ip'] = ip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-geolocation16.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

