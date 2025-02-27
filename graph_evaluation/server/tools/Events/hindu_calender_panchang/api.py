import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def panchang(tz: str='Asia/Kolkata', alt: str='0.007', lng: str='69.6092900', lat: str='21.6421900', date: str='09-03-2023', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Endpoint URL**
		/panchang
		
		**Method**
		GET
		
		**Query Parameters**
		date (optional) - The date for which the panchang details are requested. Format: DD-MM-YYYY.    Default : today's date
		lat (optional) - The latitude of the location for which the panchang details are requested.   Default :   23.1823900
		lng (optional) - The longitude of the location for which the panchang details are requested.   Default : 75.7764300
		tz (optional) - The timezone of the location for which the panchang details are requested. Format: Area/Location.  Default : Asia/Kolkata
		alt (optional) - The altitude of the location for which the panchang details are requested. Default: 0.494
		
		**Response**
		The API returns a JSON object with the following keys:
		
		sunrise - The time of sunrise in local time and Unix timestamp format.
		sunset - The time of sunset in local time and Unix timestamp format.
		next_sunrise - The time of next sunrise in local time and Unix timestamp format.
		prev_sunset - The time of previous sunset in local time and Unix timestamp format.
		dinamana - The duration of the day in hours and minutes.
		ratrimana - The duration of the night in hours and minutes.
		madhyahna - The time of madhyahna in local time and Unix timestamp format.
		moonrise - The time of moonrise in local time and Unix timestamp format.
		moonset - The time of moonset in local time and Unix timestamp format.
		weekday - The day of the week in English.
		bramhaMuhrat - The time of brahma muhurta in local time and Unix timestamp format.
		abhijit - The time of abhijit muhurta in local time and Unix timestamp format.
		godhuli - The time of godhuli muhurta in local time and Unix timestamp format.
		pratahSandhya - The time of pratah sandhya in local time and Unix timestamp format.
		vijayMuhurat - The time of vijay muhurta in local time and Unix timestamp format.
		sayahnaSandhya - The time of sayahna sandhya in local time and Unix timestamp format.
		nishitaMuhurta - The time of nishita muhurta in local time and Unix timestamp format.
		amritKal - The time of amrit kal in local time and Unix timestamp format.
		rahuKal - The time of rahu kal in local time and Unix timestamp format.
		gulikaiKal - The time of gulikai kal in local time and Unix timestamp format.
		yamaganda - The time of yamaganda in local time and Unix timestamp format.
		durMuhurtam - The duration of the muhurta in hours and minutes.
		tithiPaksha - The tithi and paksha  in local time and Unix timestamp format.
		nakshatra - The nakshatra in local time and Unix timestamp format.
		nakshatraPada - The pada of the nakshatra for the requested date in local time and Unix timestamp format.
		yoga - The time of Yogas  in local time and Unix timestamp format.
		karana - The time of Karanas in local time and Unix timestamp format.
		suryaNakshatra - The nakshatra  for the requested date in local time and Unix timestamp format.
		suryaPada - The pada of the nakshatra for the requested date in local time and Unix timestamp format.
		lunarRaasi - The lunarRaasi for the requested date in local time and Unix timestamp format.
		solarRaasi - The solarRaasi for the requested date in local time and Unix timestamp format.
		maasa -   The name of samvastsara in sanskrit
		samvatsara - The name of samvastsara in sanskrit .
		varjyam -  The varjyam for the requested date in local time and Unix timestamp format."
    
    """
    url = f"https://hindu-calender-panchang.p.rapidapi.com/"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    if alt:
        querystring['alt'] = alt
    if lng:
        querystring['lng'] = lng
    if lat:
        querystring['lat'] = lat
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hindu-calender-panchang.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

