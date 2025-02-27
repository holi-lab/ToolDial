import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def check_endpoint(ipaddress: str, maxageindays: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The check endpoint accepts a single IP address (v4 or v6). Optionally you may set the maxAgeInDays parameter to only return reports within the last x amount of days.
		
		The desired data is stored in the data property. Here you can inspect details regarding the IP address queried, such as version, country of origin, usage type, ISP, and domain name. And of course, there is the valuable abusive reports.
		
		Geolocation, usage type, ISP, and domain name are sourced from the IP2Location™ IP Address Geolocation Database. If you're looking for a performant IP database for geolocation, then use their product directly.
		
		The isWhitelisted property reflects whether the IP is spotted in any of our whitelists. Our whitelists give the benefit of the doubt to many IPs, so it generally should not be used as a basis for action. The abuseConfidenceScore is a better basis for action, because it is nonbinary and allows for nuance. The isWhitelisted property may be null if a whitelist lookup was not performed.
		
		The usageType is a string that describes the general usage of this address. Possible values are:
		
		Commercial
		Organization
		Government
		Military
		University/College/School
		Library
		Content Delivery Network
		Fixed Line ISP
		Mobile ISP
		Data Center/Web Hosting/Transit
		Search Engine Spider
		Reserved
		The maxAgeInDays parameter determines how far back in time we go to fetch reports. In this example, we ask for reports no older than 90 days. The default is 30.
		
		The totalReports property is a sum of the reports within maxAgeInDays.
		
		Reports are included in this response because the verbose flag was added. Omitting the verbose flag will exclude reports and the country name field. If you want to keep your response payloads light, this is recommended. The reports array is limited to 10,000 elements. Only reports within the timeframe of maxAgeInDays are considered.
		
		The IP address should be url-encoded, because IPv6 addresses use colons, which are reserved characters in URIs."
    maxageindays: default : 30
min : 1
max : 365
        
    """
    url = f"https://abuse-ip-check.p.rapidapi.com/api/v2/check"
    querystring = {'ipAddress': ipaddress, }
    if maxageindays:
        querystring['maxAgeInDays'] = maxageindays
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abuse-ip-check.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

