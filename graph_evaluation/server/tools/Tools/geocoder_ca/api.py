import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def geocode(addresst: str=None, stno: str=None, city: str=None, prov: str=None, locate: str=None, strict: str=None, standard: str=None, decimal: str=None, postal: str=None, geoit: str=None, jsonp: str=None, callback: str=None, utm: str=None, is_id: str=None, strictmode: str=None, showpostal: str=None, topmatches: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    addresst: The name of the street address.	 A string no longer that 220 bytes.
        stno: Street Number	 A number.
        city: The City Name	 A string no longer that 220 bytes.
        prov: The Two letter Canadian province code or US state abbreviation.	 (AB,BC,MB,NB,NL,NS,NT,NU,ON,PE,QC,SK,YT)
        locate: Optionally you can specify a location. for example 330 metcalfe ottawa or metcalfe & wellington ottawa	 This is an optional parameter for your convenience. We will parse out an address or intersection from it.
        strict: Optional Parameter for enabling strict parsing of free form location input.
        standard: Return the properly formated "standartized" address in the response.	 This is an optional parameter for address standardization. There is only one valid value: 1
        decimal: An integer positive number.	 This is an optional parameter to limit the number of decimal places in the response. (note that a small number will reduce accuracy)
        postal: A six letter canadian postal code	 The postal code format should follow the following format ANANAN where N represents a number and A a letter.
        geoit: The output type desired.	 Only one allowed value: XML
        jsonp: Output in JSON format.	 Optionally you may request data in JSON format. Accepted value: 1
        callback: Callback string if Output is in JSON format.	 Optionally you may request data in JSON format. The callback can be any string value.
        utm: Output latitude/longitude pair in The Universal Transverse Mercator (UTM) geographic coordinate system.	 Optional. Accepted value: 1
        is_id: optionally you can include your own transaction id. this will be returned along with the response if provided.	 a number or string no longer that 15 bytes.
        strictmode: Optionally you can prevent geocoder from making guesses on your input - for example if you enter just a city name without the state or province, instead of geocoder determining the most likely city, it will let you chose from a list of suggestions.	 Allowed values are Integer 0 or 1.
        showpostal: Optionally - If you supply just a street address (or intersection), the showpostal parameter will instruct the algorithm to return the postal code of the location along with the latitude/longitude pair.	 Only one allowed value: 1
        topmatches: Optionally - If you supply a partial street address and wish to obtain a fixed number of the most likely suggestions, send a value through this parameter.	 This must be the maximum number of suggestions desired in the response. The topmatches parameter value must be a positive integer.
        
    """
    url = f"https://community-geocoderca.p.rapidapi.com/"
    querystring = {}
    if addresst:
        querystring['addresst'] = addresst
    if stno:
        querystring['stno'] = stno
    if city:
        querystring['city'] = city
    if prov:
        querystring['prov'] = prov
    if locate:
        querystring['locate'] = locate
    if strict:
        querystring['strict'] = strict
    if standard:
        querystring['standard'] = standard
    if decimal:
        querystring['decimal'] = decimal
    if postal:
        querystring['postal'] = postal
    if geoit:
        querystring['geoit'] = geoit
    if jsonp:
        querystring['jsonp'] = jsonp
    if callback:
        querystring['callback'] = callback
    if utm:
        querystring['utm'] = utm
    if is_id:
        querystring['id'] = is_id
    if strictmode:
        querystring['strictmode'] = strictmode
    if showpostal:
        querystring['showpostal'] = showpostal
    if topmatches:
        querystring['topmatches'] = topmatches
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-geocoderca.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

