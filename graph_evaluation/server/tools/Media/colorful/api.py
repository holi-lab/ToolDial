import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_random_named_color(quantity: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**quantity** _integer_ (optional) The number of colors to return. Max is 50. If no quantity is specified, returns only one color-object. Quantities above 50 will default to 50.
		
		If a quantity less than 1 is used, it will default to 0, and an empty array will be returned. If the quantity cannot be parsed to an integer, a 400 error will be returned.
		
		Returns an array of objects, or one object, containing information about random colors from the list of named color keywords that can be used in CSS according to the W3 standard. The color-object contains the color codes in RGB, CMYK, HEX, HSL, and HSV (HSB). In addition the property "safe" specifies whether the color is [web-safe](https://en.wikipedia.org/wiki/Web_colors#Web-safe_colors) or not."
    
    """
    url = f"https://colorful3.p.rapidapi.com/randomNamedColor/{quantity}"
    querystring = {}
    if quantity:
        querystring['quantity'] = quantity
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "colorful3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_random_color(quantity: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**quantity** _integer_ (optional) The number of colors to return. Max is 50. If no quantity is specified, returns only one color-object. Any quantities above 50 will default to 50.
		
		If a quantity less than 1 is used, it will default to 0, and an empty array will be returned. If the quantity cannot be parsed to an integer, a 400 error will be returned.
		
		Returns an array of objects, or one object, containing information about random colors from the sRGB color space. The color-object contains the color codes in RGB, CMYK, HEX, HSL, and HSV (HSB). In addition the property "safe" specifies whether the color is [web-safe](https://en.wikipedia.org/wiki/Web_colors#Web-safe_colors) or not. If the color has a name that can be used in CSS, the name(s) will also be included. The nearest color(s) will also be included."
    
    """
    url = f"https://colorful3.p.rapidapi.com/randomColor/{quantity}"
    querystring = {}
    if quantity:
        querystring['quantity'] = quantity
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "colorful3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

