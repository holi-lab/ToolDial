import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lookup_address(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup or verify billing and shipping addresses"
    address: Street or mailing address (URL-encoded)
        
    """
    url = f"https://autohost-threat-intelligence.p.rapidapi.com/lookup/geocode/{address}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "autohost-threat-intelligence.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lookup_ip(ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup threat reports associated with an IP address"
    ip: IP Address
        
    """
    url = f"https://autohost-threat-intelligence.p.rapidapi.com/lookup/ip-threats/{ip}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "autohost-threat-intelligence.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lookup_geoip(ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find location details by IP address"
    ip: IP Address
        
    """
    url = f"https://autohost-threat-intelligence.p.rapidapi.com/lookup/geoip/{ip}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "autohost-threat-intelligence.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lookup_domain(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find company details by domain"
    domain: domain name (e.g. oracle.com)
        
    """
    url = f"https://autohost-threat-intelligence.p.rapidapi.com/lookup/domain/{domain}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "autohost-threat-intelligence.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lookup_whois(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Discover who owns a domain"
    domain: domain name (e.g. google.com)
        
    """
    url = f"https://autohost-threat-intelligence.p.rapidapi.com/lookup/whois/{domain}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "autohost-threat-intelligence.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lookup_breaches(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Discover how many times a user has been compromised"
    email: URL encoded email address
        
    """
    url = f"https://autohost-threat-intelligence.p.rapidapi.com/lookup/breaches/{email}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "autohost-threat-intelligence.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lookup_sexoffender(fname: str, state: str, lname: str, dob: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search sex offenders in the USA"
    fname: First name
        state: State (e.g. CA)
        lname: Last name
        dob: Date of birth (YYYY-MM-DD)
        
    """
    url = f"https://autohost-threat-intelligence.p.rapidapi.com/lookup/sexoffender"
    querystring = {'fname': fname, 'state': state, 'lname': lname, 'dob': dob, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "autohost-threat-intelligence.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lookup_ofac(name: str, dob: str, passport_number: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the OFAC Sanctions List by name, date of birth and aliases"
    name: Full name or alias
        dob: Date of birth (YYYY-MM-DD)
        passport_number: Passport number
        
    """
    url = f"https://autohost-threat-intelligence.p.rapidapi.com/lookup/ofac"
    querystring = {'name': name, 'dob': dob, }
    if passport_number:
        querystring['passport_number'] = passport_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "autohost-threat-intelligence.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lookup_phone(number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup line type and carrier"
    number: URL-encoded phone number in international format (e.g. +441234567890)
        
    """
    url = f"https://autohost-threat-intelligence.p.rapidapi.com/lookup/phone/{number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "autohost-threat-intelligence.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validate_phone(number: str, country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ensure that the phone number is valid"
    number: URL-encoded phone number in international format (e.g. +441234567890)
        country: ISO country code (two letters)
        
    """
    url = f"https://autohost-threat-intelligence.p.rapidapi.com/validate/phone/{number}"
    querystring = {}
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "autohost-threat-intelligence.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validate_email(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ensure that the email address is valid"
    email: URL encoded email address
        
    """
    url = f"https://autohost-threat-intelligence.p.rapidapi.com/validate/email/{email}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "autohost-threat-intelligence.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

