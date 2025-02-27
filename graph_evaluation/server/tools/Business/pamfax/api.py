import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def listversions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists the current Versions.  Result contains versions for the PamFax Gadget, Client etc and returns the version and update url"
    
    """
    url = f"https://pamfax-pamfax.p.rapidapi.com/Common/ListVersions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pamfax-pamfax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getnumberinfo(faxnumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get some information about a fax number.  Result contains zone, type, city, ... Validates and corrects the number too."
    faxnumber: The faxnumber to query (incl countrycode: +12139851886, min length: 8)
        
    """
    url = f"https://pamfax-pamfax.p.rapidapi.com/NumberInfo/GetNumberInfo"
    querystring = {'faxnumber': faxnumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pamfax-pamfax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listcountriesforzone(zone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all countries in the given zone  Result includes their translated names, countrycode and country-prefix."
    zone: Zone of the country which is wanted (1-7)
        
    """
    url = f"https://pamfax-pamfax.p.rapidapi.com/Common/ListCountriesForZone"
    querystring = {'zone': zone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pamfax-pamfax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listlanguages(min_percent_translated: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all available languages.  Result may be filtered to that only languages are returned that are at least translated $min_percent_translated %"
    min_percent_translated: the percentage value the languages have to be translated (default: 75)
        
    """
    url = f"https://pamfax-pamfax.p.rapidapi.com/Common/ListLanguages"
    querystring = {}
    if min_percent_translated:
        querystring['min_percent_translated'] = min_percent_translated
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pamfax-pamfax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listcountries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a preview page for a fax.  May be in progress, sent or from inbox."
    
    """
    url = f"https://pamfax-pamfax.p.rapidapi.com/Common/ListCountries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pamfax-pamfax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listcurrencies(code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of supported currencies.  Result contains convertion rates too. If $code is given will only return the specified currency's information."
    code: CurrencyCode
        
    """
    url = f"https://pamfax-pamfax.p.rapidapi.com/Common/ListCurrencies"
    querystring = {}
    if code:
        querystring['code'] = code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pamfax-pamfax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listsupportedfiletypes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the supported file types for documents that can be faxed."
    
    """
    url = f"https://pamfax-pamfax.p.rapidapi.com/Common/ListSupportedFileTypes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pamfax-pamfax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countfaxes(type: str, usertoken: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the number of faxes from users history with a specific state."
    type: Possible values: history, inbox, inbox_unread, outbox or unpaid
        
    """
    url = f"https://pamfax-pamfax.p.rapidapi.com/FaxHistory/CountFaxes"
    querystring = {'type': type, 'usertoken': usertoken, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pamfax-pamfax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listtimezones(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all supported timezones.  TimezonesList result list will contain attributes 'default' and (if a user is logged in) 'user_timezone' which contain that values. Additionally the corresponding list entries are marked with attributes 'is_default' and 'is_user_timezone'."
    
    """
    url = f"https://pamfax-pamfax.p.rapidapi.com/Common/ListTimezones"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pamfax-pamfax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpageprice(faxnumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate the expected price per page to a given fax number.  Use GetNumberInfo when you do not need pricing information, as calculating expected price takes longer then just looking up the info for a number."
    faxnumber: The faxnumber to query (incl countrycode: +12139851886, min length: 8). Login user first to get personalized prices.
        
    """
    url = f"https://pamfax-pamfax.p.rapidapi.com/NumberInfo/GetPagePrice"
    querystring = {'faxnumber': faxnumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pamfax-pamfax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcurrentcultureinfo(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the current culture info data."
    
    """
    url = f"https://pamfax-pamfax.p.rapidapi.com/Common/GetCurrentCultureInfo"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pamfax-pamfax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listzones(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns price and price_pro for a given zone"
    
    """
    url = f"https://pamfax-pamfax.p.rapidapi.com/Common/ListZones"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pamfax-pamfax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verifyuser(username: str, password: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Verifies a PamFax user via username/password"
    username: Username of the user or the md5 of user's username. That's what he has entered when he registered
        password: The password (or the md5 of the password) that the user entered in the registration process for the given username (case sensitive)
        
    """
    url = f"https://pamfax-pamfax.p.rapidapi.com/Session/VerifyUser"
    querystring = {'username': username, 'password': password, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pamfax-pamfax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcurrentsettings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the current settings for timezone and currency.  This is the format/timezone ALL return values of the API are in. These are taken from the user (if logged in, the api user's settings or the current ip address)"
    
    """
    url = f"https://pamfax-pamfax.p.rapidapi.com/Common/GetCurrentSettings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pamfax-pamfax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getgeoipinformation(ip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Geo information based on the given IP address (IPV4)"
    ip: the ip to get geo information off
        
    """
    url = f"https://pamfax-pamfax.p.rapidapi.com/Common/GetGeoIPInformation"
    querystring = {'ip': ip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pamfax-pamfax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def liststrings(ids: str=None, culture: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of strings translated into the given language."
    ids: array of String identifiers. You may also pass a comma separated list as $ids[0] (ids[0]=BTN_YES,BTN_NO).
        culture: culture identifier, defaults to users culture. Accepts full culture-codes like en-US, de-DE and just a language code like en, de, ...
        
    """
    url = f"https://pamfax-pamfax.p.rapidapi.com/Common/ListStrings"
    querystring = {}
    if ids:
        querystring['ids'] = ids
    if culture:
        querystring['culture'] = culture
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pamfax-pamfax.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

