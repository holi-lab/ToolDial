import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def orgcode_cards(x_xsrf_token: str, muid: str, msid: str, orgcode: str, locale: str, uuid: str='string', x_mjx_server: str='string', x_passthru_values: str='string', x_deviceinstall: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of cards for the given session. User authentication is required."
    x_xsrf_token: A value used to help prevent cross-site request forgery attacks.
        muid: User ID.
        msid: Session ID.
        orgcode: The organization associated with the request.
        locale: Language used.
        uuid: Optional Unique ID for the request. This value is passed through all layers of the system. If it is not specified, a value is generated.
        x_mjx_server: mBanking node identifier for load-balanced environments. This must be passed as a header, rather than a query parameter, so that the mBanking load balancer does not make use of the banking node for DSM request routing, but allows the DSM to pass it through when making requests to the mBanking server.
        x_passthru_values: Optional key value pairs to be passed as-is through to the integration layer. The format should be key1=value1;key2=value2;key3=value3.
        x_deviceinstall: The unique identifier assigned to this device during registration.
        
    """
    url = f"https://test2113.p.rapidapi.com/{orgcode}/cards"
    querystring = {'X-XSRF-TOKEN': x_xsrf_token, 'muid': muid, 'msid': msid, 'locale': locale, }
    if uuid:
        querystring['Uuid'] = uuid
    if x_mjx_server:
        querystring['X-MJX-Server'] = x_mjx_server
    if x_passthru_values:
        querystring['X-Passthru-Values'] = x_passthru_values
    if x_deviceinstall:
        querystring['X-DeviceInstall'] = x_deviceinstall
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "test2113.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def orgcode_cardart_bins(muid: str, locale: str, orgcode: str, msid: str, bins: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns card art URLs by BINs. Different URLs can be defined for each BIN/range of BINs. User authentication is not required."
    muid: User ID.
        locale: Language used.
        orgcode: The organization associated with the request.
        msid: Session ID.
        bins: One or more six-digit bin separated by commas.
        
    """
    url = f"https://test2113.p.rapidapi.com/{orgcode}/cardart/{bins}"
    querystring = {'muid': muid, 'locale': locale, 'msid': msid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "test2113.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def orgcode_cards_cardid(x_xsrf_token: str, cardid: str, muid: str, msid: str, orgcode: str, locale: str, uuid: str='string', x_deviceinstall: str='string', x_mjx_server: str='string', x_passthru_values: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns card information for a given card ID. User authentication is required."
    x_xsrf_token: A value used to help prevent cross-site request forgery attacks.
        cardid: The card ID.
        muid: User ID.
        msid: Session ID.
        orgcode: The organization associated with the request.
        locale: Language used.
        uuid: Optional Unique ID for the request. This value is passed through all layers of the system. If it is not specified, a value is generated.
        x_deviceinstall: The unique identifier assigned to this device during registration.
        x_mjx_server: mBanking node identifier for load-balanced environments. This must be passed as a header, rather than a query parameter, so that the mBanking load balancer does not make use of the banking node for DSM request routing, but allows the DSM to pass it through when making requests to the mBanking server.
        x_passthru_values: Optional key value pairs to be passed as-is through to the integration layer. The format should be key1=value1;key2=value2;key3=value3.
        
    """
    url = f"https://test2113.p.rapidapi.com/{orgcode}/cards/{cardid}"
    querystring = {'X-XSRF-TOKEN': x_xsrf_token, 'muid': muid, 'msid': msid, 'locale': locale, }
    if uuid:
        querystring['Uuid'] = uuid
    if x_deviceinstall:
        querystring['X-DeviceInstall'] = x_deviceinstall
    if x_mjx_server:
        querystring['X-MJX-Server'] = x_mjx_server
    if x_passthru_values:
        querystring['X-Passthru-Values'] = x_passthru_values
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "test2113.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def orgcode_accounts_accountid_cards(x_xsrf_token: str, locale: str, msid: str, orgcode: str, muid: str, accountid: str, x_mjx_server: str='string', x_deviceinstall: str='string', x_passthru_values: str='string', uuid: str='string', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of cards belonging to a specific account. User authentication is required."
    x_xsrf_token: A value used to help prevent cross-site request forgery attacks.
        locale: Language used.
        msid: Session ID.
        orgcode: The organization associated with the request.
        muid: User ID.
        accountid: The parent account ID.
        x_mjx_server: mBanking node identifier for load-balanced environments. This must be passed as a header, rather than a query parameter, so that the mBanking load balancer does not make use of the banking node for DSM request routing, but allows the DSM to pass it through when making requests to the mBanking server.
        x_deviceinstall: The unique identifier assigned to this device during registration.
        x_passthru_values: Optional key value pairs to be passed as-is through to the integration layer. The format should be key1=value1;key2=value2;key3=value3.
        uuid: Optional Unique ID for the request. This value is passed through all layers of the system. If it is not specified, a value is generated.
        
    """
    url = f"https://test2113.p.rapidapi.com/{orgcode}/accounts/{accountid}/cards"
    querystring = {'X-XSRF-TOKEN': x_xsrf_token, 'locale': locale, 'msid': msid, 'muid': muid, }
    if x_mjx_server:
        querystring['X-MJX-Server'] = x_mjx_server
    if x_deviceinstall:
        querystring['X-DeviceInstall'] = x_deviceinstall
    if x_passthru_values:
        querystring['X-Passthru-Values'] = x_passthru_values
    if uuid:
        querystring['Uuid'] = uuid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "test2113.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

