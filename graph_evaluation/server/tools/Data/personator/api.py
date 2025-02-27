import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def personator_endpoint(act: str, last: str=None, lastline: str=None, mak: str=None, state: str='ca', comp: str=None, ip: str=None, a2: str=None, ctry: str=None, bmo: str=None, bday: str=None, byr: str=None, ss: str=None, first: str=None, format: str='json', a1: str='22382 avenida empresa', postal: str=None, phone: str=None, full: str=None, email: str=None, city: str='rsm', ff: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access to Personator Web Service"
    act: Actions (Check, Verify, Append, Move)
        last: Last Name
        lastline: Last Line (City, State, Postal)
        mak: Melissa Address Key
        state: State (State and City required OR postal)
        comp: Company Name
        ip: IP Address
        a2: Address Line 2
        ctry: Country
        bmo: Birth Month
        bday: Birth Day
        byr: Birth Year
        ss: Social Security Number
        first: First Name
        format: Format of Response
        a1: Address Line 1
        postal: Postal Code
        phone: Phone Number
        full: Full Name
        email: Email Address
        city: City
        ff: Free Form
        
    """
    url = f"https://personator2.p.rapidapi.com/v3/WEB/ContactVerify/doContactVerify"
    querystring = {'act': act, }
    if last:
        querystring['last'] = last
    if lastline:
        querystring['lastline'] = lastline
    if mak:
        querystring['mak'] = mak
    if state:
        querystring['state'] = state
    if comp:
        querystring['comp'] = comp
    if ip:
        querystring['ip'] = ip
    if a2:
        querystring['a2'] = a2
    if ctry:
        querystring['ctry'] = ctry
    if bmo:
        querystring['bmo'] = bmo
    if bday:
        querystring['bday'] = bday
    if byr:
        querystring['byr'] = byr
    if ss:
        querystring['ss'] = ss
    if first:
        querystring['first'] = first
    if format:
        querystring['format'] = format
    if a1:
        querystring['a1'] = a1
    if postal:
        querystring['postal'] = postal
    if phone:
        querystring['phone'] = phone
    if full:
        querystring['full'] = full
    if email:
        querystring['email'] = email
    if city:
        querystring['city'] = city
    if ff:
        querystring['ff'] = ff
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "personator2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

