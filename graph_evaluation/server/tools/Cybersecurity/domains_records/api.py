import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def initiate_download_of_domain_records_a_on_the_requested_domain_with_different_output_file_formats(format: str, domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "initiate download of domain records (A) on the requested domain with different output file formats.
		formats: csv, json, maltego and subdomains.
		Usually,
		1. csv format can be import to xls or other table view;
		2. maltego format can be import to app. Maltego (www.maltego.com)."
    
    """
    url = f"https://domains-records.p.rapidapi.com/download/domain/pairs/{format}/{domain}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domains-records.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def count_records_a_by_ipv4_json_response(ipv4: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "count records (pairs hostname - IPv4) by ipv4 address"
    
    """
    url = f"https://domains-records.p.rapidapi.com/info/ip/{ipv4}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domains-records.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_domain_records_a_by_ipv4_json_response(ipv4: str, limit: str=None, weeks: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get domain records(A) by IPv4, by default, search depth 8 weeks, response limit 2000 records."
    
    """
    url = f"https://domains-records.p.rapidapi.com/find/ip/{ipv4}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if weeks:
        querystring['weeks'] = weeks
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domains-records.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def count_records_a_by_domain_json_response(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "count records (pairs hostname - IPv4) by domain"
    
    """
    url = f"https://domains-records.p.rapidapi.com/info/domain/pairs/{domain}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domains-records.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def initiate_download_of_domain_records_a_on_the_requested_ipv4_with_different_output_file_formats(ipv4: str, format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "initiate download of domain records (A) on the requested domain with different output file formats.
		formats: csv, json, maltego.
		Usually,
		1. csv format can be import to xls or other table view;
		2. maltego format can be import to app. Maltego (www.maltego.com)."
    
    """
    url = f"https://domains-records.p.rapidapi.com/download/network/{format}/32/{ipv4}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domains-records.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subdomains_from_domain_stream_text_response(domain: str, limit: int=2000, weeks: int=8, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stream subdomains, by default, search depth 8 weeks, response limit 2000 records."
    
    """
    url = f"https://domains-records.p.rapidapi.com/stream/subdomains/{domain}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if weeks:
        querystring['weeks'] = weeks
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domains-records.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_domain_records_pairs_hostname_and_ipv4_by_given_domain_stream_text_response(domain: str, limit: int=2000, weeks: int=8, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stream records about subdomains as text(csv).
		By default, search depth 8 weeks, response limit 2000 records."
    
    """
    url = f"https://domains-records.p.rapidapi.com/stream/domain/{domain}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if weeks:
        querystring['weeks'] = weeks
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domains-records.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_hostnames_from_ipv4_stream_text(address: str, limit: int=2000, weeks: int=8, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "stream  text(csv) of unique pairs from address and hostname about given ipv4 address.
		By default, search depth 8 weeks, response limit 2000 records."
    limit: result limit
        weeks: search for last weeks from current date
        
    """
    url = f"https://domains-records.p.rapidapi.com/stream/find/ip/{address}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if weeks:
        querystring['weeks'] = weeks
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domains-records.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_domain_records_pairs_hostname_and_ipv4_by_given_domain_json_response(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns records FQDN (type A) with IPv4 with history as json"
    
    """
    url = f"https://domains-records.p.rapidapi.com/find/domain/pairs/{domain}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domains-records.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

