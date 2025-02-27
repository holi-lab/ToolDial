import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fraudlabs_api(ip: str, license: str, region: str=None, country: str=None, postal: str=None, domain: str=None, phone: str=None, bin: str=None, binname: str=None, binphone: str=None, shipaddr: str=None, shipcity: str=None, shipregion: str=None, shippostal: str=None, shipcountry: str=None, queryid: str=None, city: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Credit Card Fraud Detection API"
    ip: IP address where order originated from.
        license: API license key.
        region: Region of billing address.
        country: Country code of billing address.
        postal: Postal/Zip code of billing address.
        domain: Email address domain name.
        phone: Customer phone number.
        bin: First 6 digits of credit card number to identify issuing bank.
        binname: Name of the bank which issued the credit card.
        binphone: Customer service phone number listed on back of credit card.
        shipaddr: Address of shipping address if different than billing address.
        shipcity: City of shipping address if different than billing address.
        shipregion: Region of shipping address if different than billing address.
        shippostal: Postal/Zip code of shipping address if different than billing address.
        shipcountry: Country of shipping address if different than billing address.
        queryid: Additional information to identify this transaction.
        city: City of billing address.
        
    """
    url = f"https://fraudlabs-fraudlabs-v1.p.rapidapi.com/fraudlabswebservice.asmx"
    querystring = {'IP': ip, 'LICENSE': license, }
    if region:
        querystring['REGION'] = region
    if country:
        querystring['COUNTRY'] = country
    if postal:
        querystring['POSTAL'] = postal
    if domain:
        querystring['DOMAIN'] = domain
    if phone:
        querystring['PHONE'] = phone
    if bin:
        querystring['BIN'] = bin
    if binname:
        querystring['BINNAME'] = binname
    if binphone:
        querystring['BINPHONE'] = binphone
    if shipaddr:
        querystring['SHIPADDR'] = shipaddr
    if shipcity:
        querystring['SHIPCITY'] = shipcity
    if shipregion:
        querystring['SHIPREGION'] = shipregion
    if shippostal:
        querystring['SHIPPOSTAL'] = shippostal
    if shipcountry:
        querystring['SHIPCOUNTRY'] = shipcountry
    if queryid:
        querystring['QUERYID'] = queryid
    if city:
        querystring['CITY'] = city
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fraudlabs-fraudlabs-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

