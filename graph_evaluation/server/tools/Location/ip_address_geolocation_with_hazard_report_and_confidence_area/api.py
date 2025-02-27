import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ip_address_geolocation_with_hazard_report_and_confidence_area(ip: str, localitylanguage: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "IP address intelligence plays a vital role in cyber security by allowing businesses to perform various checks on IP addresses. With the rise in VPN/proxy services, online scammers have become comfortable hiding their identities when committing fraud. The technology used by hackers is getting refined, and it is harder to detect with just the reactive measures of blocklisting IP addresses. BigDatacloud IP geolocation API services can provide more proactive metrics to tackle global online cyber crimes and reduce their impact.
		
		This API endpoint provides a critical security report of an IP address. It is an extension of the [IP Address Geolocation with Confidence Area API](https://www.bigdatacloud.com/docs/api/ip-address-geolocation-with-confidence-area-api), making it the most comprehensive IP address intelligence API offering.
		
		The additional data object is called hazardReport, which contains 15 security signals. The hazard report provides conventional reactive and our highly innovative proprietary proactive metrics. The API also offers a textual summary of these 15 security vulnerabilities associated with the IP address, called a securityThreat.
		
		You will find more information about the hazard report on our dedicated [Hazard Report API](https://www.bigdatacloud.com/docs/api/hazard-report-api) page.
		
		You may use our User Risk API page if you are looking for a more simplified security API. 
		
		- Unprecedented Update Rate
		- Geolocation data is partially updated every 2 hours and fully updated at least once a day
		- BGP data is updated every 2 hours
		- Registry data is updated at least once a day
		- Country object data is usually updated at least once a month
		- Hazard report data is updated every hour"
    
    """
    url = f"https://ip-address-geolocation-with-hazard-report-and-confidence-area.p.rapidapi.com/data/ip-geolocation-full"
    querystring = {'ip': ip, }
    if localitylanguage:
        querystring['localityLanguage'] = localitylanguage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-address-geolocation-with-hazard-report-and-confidence-area.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

