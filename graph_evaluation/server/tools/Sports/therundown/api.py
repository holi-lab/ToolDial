import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def events_games(date: str, sportid: str, affiliate_ids: str='1,2,3', offset: str='0', include: str='scores', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /sports/{sport-id}/events/{date} endpoint to request events for a particular sport on a particular date. The current odds and markets will be returned when available. To get historical odds for each market, use the individual `moneyline`, `spread`, and `total` endpoints in the `Lines` endpoints group.
		
		The date range defaults to UTC unless an offset query parameter is specified, which is the offset from UTC in minutes. For example, if the request is meant to be made from CDT, the offset should be offset=300 (5 hours).  
		
		Specifying optional include values may be used to get lines for all markets (instead of just full-game by default) in addition to scores or the team names from specific sportsbooks. 
		
		To request multiple, simply add multiple values and duplicate the include= parameter in the request like so: `?include=all_periods&include=scores`.  
		
		When `include=all_periods` is used, the key for the lines changes from lines to line_periods.  
		
		An optional offset in minutes from UTC may be sent in the request to group events by date with an offset. 
		
		For example, if you are in CDT and want to see events grouped by date in CDT, then specify ?offset=300.  
		
		Any value of 0.0001 represents the value NotPublished. This means that the sportsbook currently has not published a price or wager for this event, or that the line was removed."
    date: The date (in ISOO8601 format) of the events to return for the sport. Defaults to UTC unless an offset is specified in minutes (see query params)
        offset: The offset in minutes from the provided date. The default offset is UTC.
        include: (Optional) Specifies whether the response should include all available markets (halves/quarters/periods) instead of just full-game by default. 
        
    """
    url = f"https://therundown-therundown-v1.p.rapidapi.com/sports/{sportid}/events/{date}"
    querystring = {}
    if affiliate_ids:
        querystring['affiliate_ids'] = affiliate_ids
    if offset:
        querystring['offset'] = offset
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "therundown-therundown-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def conferences(sportid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get conferences for a sport"
    
    """
    url = f"https://therundown-therundown-v1.p.rapidapi.com/sports/{sportid}/conferences"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "therundown-therundown-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def divisions(sportid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get divisions for a sport"
    
    """
    url = f"https://therundown-therundown-v1.p.rapidapi.com/sports/{sportid}/divisions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "therundown-therundown-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams(sportid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Teams provides a list of teams that are included in the `normalized_teams` attribute of the events responses."
    
    """
    url = f"https://therundown-therundown-v1.p.rapidapi.com/sports/{sportid}/teams"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "therundown-therundown-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def openers(sportid: int, date: str, include: str='scores&include=all_periods', offset: str='300', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides the opening lines offered for the various sportsbooks (affiliates). 
		
		The key of the associate array in the lines attribute maps to the affiliate_id of the affiliate. For example, the key of "1" maps to the affiliate of "5Dimes".
		
		 Values of 0.0001 represent a pulled line, or a line that was removed from the sportsbook. 
		
		Add the below includes in the Optional parameters to include things like scores, team names for each sportsbook, or all available periods (halves/quarters/periods) for the events."
    date: The date in yyyy-mm-dd format (eg: 2018-12-16). Defaults to UTC.
        include: (Optional) Specifies whether the response should include scores, status, and other game information.
        offset: The offset in minutes from UTC by which to group the events for the dates. For example, if the user is in CST, then offset is 360.
        
    """
    url = f"https://therundown-therundown-v1.p.rapidapi.com/sports/{sportid}/openers/{date}"
    querystring = {}
    if include:
        querystring['include'] = include
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "therundown-therundown-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def totals(include: str='all_periods', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the current and historical totals for the provided `line_id`.
		
		Each market from each sportsbook for each game has  a unique `line_id`. This `line_id` can be used to get historical pricing and values for each market.
		
		The date_updated value corresponds to when the price or value for this market changed."
    include: Specifies whether the response should include all available periods (halves/quarters/periods) in the response for events
        
    """
    url = f"https://therundown-therundown-v1.p.rapidapi.com/lines/{line-id}/total"
    querystring = {}
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "therundown-therundown-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def moneyline(lineid: int, include: str='all_periods', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the current and historical moneyline for the provided `line_id`. 
		
		Each market from each sportsbook for each game has  a unique `line_id`. This `line_id` can be used to get historical pricing and values for each market.
		
		The date_updated value corresponds to when the price or value for this market changed."
    include: Specifies whether the response should include all available periods (halves/quarters/periods) in the response for events
        
    """
    url = f"https://therundown-therundown-v1.p.rapidapi.com/lines/{lineid}/moneyline"
    querystring = {}
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "therundown-therundown-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def spread(lineid: int, include: str='all_periods', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the current and historical spread for the provided `line_id`.
		
		Each market from each sportsbook for each game has  a unique `line_id`. This `line_id` can be used to get historical pricing and values for each market.
		
		The `date_updated` value corresponds to when the price or value for this  market changed."
    include: Specifies whether the response should include all available periods (halves/quarters/periods) in the response for events
        
    """
    url = f"https://therundown-therundown-v1.p.rapidapi.com/lines/{lineid}/spread"
    querystring = {}
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "therundown-therundown-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dates_with_odds(format: str='date', offset: int=300, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the dates of the sport's events that contain odds that are on or after the current date. 
		
		To get all the events regardless of whether or not they have odds use the events by sport and date endpoint.
		
		An optional offset in minutes from UTC may be sent in the request to get dates offset from UTC. For example, if you want the results in CDT, then specify ?offset=300."
    format: Optional format endpoint that will return the dates as unix epoch (format=epoch) or ISO8601 (default)
        
    """
    url = f"https://therundown-therundown-v1.p.rapidapi.com/sports/{sport-id}/dates"
    querystring = {}
    if format:
        querystring['format'] = format
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "therundown-therundown-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def affiliates_sportsbooks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get affiliates (aka sportsbooks).
		
		The `affiliate_id` value of each sportsbook is used as the key in the `lines` or `line_periods` events response. For example, `5Dimes` has an `affiliate_id` of `1`, so their lines are available with the key of `1` in the `events` endpoints responses."
    
    """
    url = f"https://therundown-therundown-v1.p.rapidapi.com/affiliates"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "therundown-therundown-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event(eventid: str, include: str='scores', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /events/{event-id} endpoint can be used to request information for a single event.  The currently available odds from each sportsbook will be returned with the game information.
		
		Specifying optional include values may be used to get lines for all markets (instead of just full-game by default) in addition to scores or the team names from specific sportsbooks. 
		
		To request multiple, simply add multiple values and duplicate the include= parameter in the request like so: `&include=allperiods&include=scores`.  When `include=allperiods` is used, the key for the lines changes from lines to line_periods. 
		
		Any value of `0.0001` represents the value NotPublished. This means that the sportsbook currently has not published a price or odds for this wager, or that the line was removed."
    include: When `include=all_periods` is used, the key for the lines changes from lines to line_periods.

When `include=scores` is used, game status and scores will be returned.

To request multiple, simply add multiple values and duplicate the include= parameter in the request like so: ?include=all_periods&include=scores. 
        
    """
    url = f"https://therundown-therundown-v1.p.rapidapi.com/events/{eventid}"
    querystring = {}
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "therundown-therundown-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schedules(sportid: int, is_from: str=None, limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the schedule for a sportID.
		
		Use the `from` parameter to change the starting date in the format of `yyyy-mm-dd` (ex: `2020-09-20`), which defaults to today. 
		
		Use a `limit` parameter to set the number of events returned in the response. The max is `500` and the default is `50`.
		
		The schedules are ordered by `date_event` ascending, so send the largest or last `date_event` value current response to get the next available page."
    limit: Max value is 500. Default is 50
        
    """
    url = f"https://therundown-therundown-v1.p.rapidapi.com/sports/{sportid}/schedule"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "therundown-therundown-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def delta_changed_events(last_id: str, include: str='all_periods', sport_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The `/delta` endpoint returns only events that have changed since the last request, specified by the delta last id value contained in any /events request.    The delta last id should be passed to the `/delta` request as the `last id` parameter. A sport id may be optionally used to filter the delta response.    Specifying optional include values may be used to get lines for all markets (instead of just full-game by default) in addition to scores or the team names from specific sportsbooks. To request multiple, simply add multiple values and duplicate the include= parameter in the request like so: &include=all periods&include=scores.    When include=all periods is used, the key for the lines changes from lines to line periods.    A 400 will be returned if too many events have been updated since the last request. If this happens, you should use any of the events endpoints (events by sport by date) to retrieve a full snapshot of the events before using the delta endpoint again with the updated last id.    Any value of 0.0001 represents the value NotPublished. This means that the sportsbook currently has not published a price or odds for this wager, or that the line was removed."
    last_id: The delta_last_id included in the delta, events by sport, or events by sport by date request
        include: (Optional) Specifies whether the response should include all available markets (halves/quarters/periods) instead of just full-game by default. 
        include: (Optional) Specifies whether the response should include scores, status, and other game information.
        sport_id: (Optional) The sport_id by which to filter the delta request.
        
    """
    url = f"https://therundown-therundown-v1.p.rapidapi.com/delta"
    querystring = {'last_id': last_id, }
    if include:
        querystring['include'] = include
    if include:
        querystring['include'] = include
    if sport_id:
        querystring['sport_id'] = sport_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "therundown-therundown-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def closing(sportid: int, date: str, include: str='scores&include=all_periods', offset: str='300', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides the closing lines offered for the various sportsbooks (affiliates). 
		
		The key of the associate array in the lines attribute maps to the affiliate_id of the affiliate. For example, the key of "1" maps to the affiliate of "5Dimes".
		
		 Values of 0.0001 represent a pulled line, or a line that was removed from the sportsbook. 
		
		Add the below includes in the Optional parameters to include things like scores, team names for each sportsbook, or all available periods (halves/quarters/periods) for the events."
    date: The date in yyyy-mm-dd format (eg: 2018-12-16). Defaults to UTC.
        include: (Optional) Specifies whether the response should include scores, status, and other game information.
        offset: The offset in minutes from UTC by which to group the events for the dates. For example, if the user is in CST, then offset is 360.
        
    """
    url = f"https://therundown-therundown-v1.p.rapidapi.com/sports/{sportid}/closing/{date}"
    querystring = {}
    if include:
        querystring['include'] = include
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "therundown-therundown-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sports(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get available sports. Sports may or may not have events depending on season and what is available on sportsbooks."
    
    """
    url = f"https://therundown-therundown-v1.p.rapidapi.com/sports"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "therundown-therundown-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

