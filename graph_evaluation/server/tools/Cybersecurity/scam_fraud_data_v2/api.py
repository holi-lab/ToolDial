import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getmentions(blockchain_address: str=None, name: str=None, source: str=None, complainant_location: str=None, description: str=None, acquired: str=None, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of `mentions` for agents and blockchain addresses which are claimed to be involved in scam and fraud activity, such as victim reports or media events.  Use the query parameters to filter responses to potential parties of interest."
    source: Source of the event
        complainant_location: The Country of mention or victim location
        description: The event context details
        acquired: The date the event was captured
        type: The category of the event
        
    """
    url = f"https://scam-fraud-data.p.rapidapi.com/events/mentions/"
    querystring = {}
    if blockchain_address:
        querystring['blockchain_address'] = blockchain_address
    if name:
        querystring['name'] = name
    if source:
        querystring['source'] = source
    if complainant_location:
        querystring['complainant_location'] = complainant_location
    if description:
        querystring['description'] = description
    if acquired:
        querystring['acquired'] = acquired
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scam-fraud-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbrokers(company_address: str=None, agent_name: str=None, company_name: str=None, agent_hash: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of `brokers` which have been identified in relation to mentioned scam and fraud events.  Use the following query parameters to filter events to parties of interest:"
    company_name: Company name associated with the agent
        agent_hash: Unique id created for the agent
        
    """
    url = f"https://scam-fraud-data.p.rapidapi.com/agents/brokers/"
    querystring = {}
    if company_address:
        querystring['company_address'] = company_address
    if agent_name:
        querystring['agent_name'] = agent_name
    if company_name:
        querystring['company_name'] = company_name
    if agent_hash:
        querystring['agent_hash'] = agent_hash
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scam-fraud-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

