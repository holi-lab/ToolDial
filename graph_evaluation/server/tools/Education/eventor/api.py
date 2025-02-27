import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def create_event(agent: str, eventinfostr1: str, phonenum: str, eventname: str, eventinfostr2: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create an event
		1)Finds agentId/eventId
		2) Insert record in table=events
			[agentId
			eventId
			phoneNum [12]
		              eventInfoStr1 (50)
			eventInfoStr2 (50)
		              timestamp]
		3) If agent or eventName not found, create them in Agent.eventName tables
		
		End point should work with either:
		/logEvent 
		or 
		/createEventInSql
		
		e.g.
		/createEventInSql?eventName=classes&agent=Sherwood%20Ford&phoneNum=6506900639&eventInfoStr1=Campaign1:CLOSED"
    
    """
    url = f"https://eventor.p.rapidapi.com/logEvent"
    querystring = {'agent': agent, 'eventInfoStr1': eventinfostr1, 'phoneNum': phonenum, 'eventName': eventname, }
    if eventinfostr2:
        querystring['eventInfoStr2'] = eventinfostr2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eventor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def delete_event(deletenumitems: int, date: str, eventname: str='classes', agent: str='Sherwood Ford', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Finds the events on date=#date
		Deletes the first #deleteNumItems
		
		For example, if there are 5 events and deleteNumItems=3, then delete first 3 and leave 2
		
		
		/deleteEvent?agent=Sherwood%20Ford&eventName=classes&date=2020-02-17&deleteNumItems=3"
    
    """
    url = f"https://eventor.p.rapidapi.com/deleteEvent"
    querystring = {'deleteNumItems': deletenumitems, 'date': date, }
    if eventname:
        querystring['eventName'] = eventname
    if agent:
        querystring['agent'] = agent
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eventor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_event_count(dateto: str, datefrom: str, eventname: str='classes', agent: str='Sherwood Ford', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Count(*) where events.date >=dateFrom and event.date<=dateTo
		
		E.g.
		/getEventCount?agent=Sherewood&eventName=classes&dateFrom=2020-02-17&dateTo=2020-02-27"
    
    """
    url = f"https://eventor.p.rapidapi.com/getEventCount"
    querystring = {'dateTo': dateto, 'dateFrom': datefrom, }
    if eventname:
        querystring['eventName'] = eventname
    if agent:
        querystring['agent'] = agent
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eventor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_event_data(eventname: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "1)Finds agentId/eventId
		2)Searches for all matches
		
		/getEventData
		
		e.g.
		/getEventData?date=2020-02-17&eventName=scheduleCompleteAutoServ"
    
    """
    url = f"https://eventor.p.rapidapi.com/getEventData"
    querystring = {'eventName': eventname, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eventor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

