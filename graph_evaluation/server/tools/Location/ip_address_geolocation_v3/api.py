import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def locateip(ip: str, format: str='"XML" or "JSON"', callback: str='callback=JavaScriptFunctionName', compact: str='compact=Y', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Geolocates an IP address. Please check our documentation at www.IPAddressLabs.com"
    ip: Contains the IP address to geolocate. Alternatively, the "ip" parameter may contain "local-ip" as a static value instead of the visitor's IP address. In such case, the IP address of the host from which our service is invoked will be geolocated. This is particularly useful when our service is invoked from a desktop application that needs to geolocate the IP address of the host from which it is being executed. It is also very useful when our service is invoked by a static javascript code that is executed in the visitor's browser (instead of by the web server in which the site resides).
        format: Specifies the data format for the response (XML or JSON). If this parameter is not present the default format is XML.
        callback: We also support JSON with Padding (JSONP), which allows callers to insert a dynamic script element and requires a user-specified function name. If you have specified the format as JSON, you can request that our service return the JSON data wrapped in a specified function name. To do so, use the "callback" parameter in the form "callback={your JavaScript function name}".
        compact: By default, the answers in XML and JSON come formated with indentations, spacings and line breaks. This facilitates its reading and understanding for humans. However, our service will be ultimately used by automated processes through parsing. To facilitate this process and abreviate the size of the response, it is possible to obviate the formating elements. This can be achieved using the "compact" parameter, in the form "compact=Y".
        
    """
    url = f"https://ipaddresslabs-ip-address-geolocation.p.rapidapi.com/locateip"
    querystring = {'ip': ip, }
    if format:
        querystring['format'] = format
    if callback:
        querystring['callback'] = callback
    if compact:
        querystring['compact'] = compact
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ipaddresslabs-ip-address-geolocation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

