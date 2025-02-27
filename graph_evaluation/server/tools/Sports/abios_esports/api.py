import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getmatchespostgameservertimelinespositionsbyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Postgame Server Positions Timeline for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/postgame/server/timelines/positions"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatcheslivecvtimelineskillsbyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Computer Vision Kills Timeline for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/live/cv/timelines/kills"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchespostgameserversummaryroundsbyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Postgame Server Round Summary for a specific match.
		This is only applicable for games that are played in rounds, such as
		Counter-Strike: Global Offensive
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change. However, we do peridiocally update our
		server data parsers to extract new data points and resolve inconsistencies.
		In those cases we will make those improvements available for all matches in
		our database, which means that historical Postgame Server data might
		see improvement even years down the line.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/postgame/server/summary/rounds"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchesliveapisummarybyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Live API Summary for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available and will reflect
		the latest state of any corresponding [Push API Channel](/docs/en/push-api/introduction/available-channels).
		
		For example the latest update on the `dota_live_api_states` channel for
		match `x` will be available as `/v3/matches/x/live/api/summary`.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/live/api/summary"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchespostgameservertimelineshealingbyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Postgame Server Healing Timeline for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/postgame/server/timelines/healing"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmappoolsbygameid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available [Map Pools](/docs/en/knowledge-base/resources/auxiliary#maps-and-map-pools)
		available for the given [Game](/docs/en/knowledge-base/resources/game).
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/games/{is_id}/mappools"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchespostgameapitimelineskillsbyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Postgame API Kills Timeline for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/postgame/api/timelines/kills"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrolesbygameid(is_id: int, take: int=50, filter: str=None, id_space: str='kambi', order: str=None, skip: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available `Role`s within a
		[Game](/docs/en/knowledge-base/resources/game).
		"
    id: Resource id
        take: Amount of results to take
        filter: How to filter the response.
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        order: How to order the results
        skip: Amount of results to skip
        
    """
    url = f"https://abios-esports.p.rapidapi.com/games/{is_id}/roles"
    querystring = {}
    if take:
        querystring['take'] = take
    if filter:
        querystring['filter'] = filter
    if id_space:
        querystring['id_space'] = id_space
    if order:
        querystring['order'] = order
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getseries(order: str=None, skip: int=0, take: int=50, filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available [Series](/docs/en/knowledge-base/resources/series).
		
		The [Team Series](#operation/Teams-getSeriesByTeamId), [Lineup Series](#operation/Lineups-getSeriesByLineupId)
		and [Player Series](#operation/Players-getSeriesByPlayerId) can be used to retrieve information
		about series where a specific
		[Team](/docs/en/knowledge-base/resource/competitors#teams),
		[Lineup](/docs/en/knowledge-base/resources/competitors#line-ups) or
		[Player](/docs/en/knowledge-base/resources/competitors#players) has
		participated, respectively.
		
		## Useful Filters
		
		* `filter=id<={1,2,3,4,5,6,7,10}` will fetch series with id 1 through 7 and 10 in a single request
		* `filter=tournament.id=1` will fetch all series part of tournament with id `1`
		* `filter=lifecycle=upcoming` will fetch all series that are yet to be played
		* `filter=lifecycle=upcoming&order=start-asc` will fetch all upcoming series sorted by their start date in ascending order
		* `filter=lifecycle=live` will fetch all series that are currently considered live
		* `filter=tier=1` will fetch all series that are considered tier 1
		
		## Filtering on Participants and finding Head to Head data
		
		In order to retrieve information about series where one or several
		particular rosters has participated you can use the following filters.
		
		* `filter=participants.roster.id^{x}` will fetch all series in which roster with id `x` has participated
		* `filter=participants.roster.id={x,y}` will fetch all series where the only participants are rosters with id `x` and `y`, i.e. the head-to-head series between those two rosters
		* `filter=participants.roster.id>={x,y}` will fetch all series where at least rosters with id `x` and `y` have participated
		* `filter=participants.roster.is^{x,y},participants.roster.id^{a,b}` will fetch all series where roster `(x or y) and (a or b)` participated. Useful if e.g one team has rosters `x` and `y` whereas another team has rosters `a` and `b`. The result would be the teams head-to-head series
		
		Read more about head-to-head filtering [here](/docs/en/atlas/usage/head-to-head).
		"
    order: How to order the results
        skip: Amount of results to skip
        take: Amount of results to take
        filter: How to filter the response.
        
    """
    url = f"https://abios-esports.p.rapidapi.com/series"
    querystring = {}
    if order:
        querystring['order'] = order
    if skip:
        querystring['skip'] = skip
    if take:
        querystring['take'] = take
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchespostgameservertimelinesobjectivesbyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Postgame Server Objectives Timeline for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/postgame/server/timelines/objectives"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchbyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the details of a single [Match](/docs/en/knowledge-base/resources/series#match).
		
		The [Multiple Matches](#operation/Matches-getMatches) operation can be used to
		retrieve multiple matches by id at once using the filter
		`filter=id<={1,2,5,6}`.
		
		Retrieving all matches part of a particular series can be done using the
		[Multiple Matches](#operation/Matches-getMatches) operation and the filter
		`filter=series.id=x`, where `x` is the series id.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimagebyid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the details of a single image.
		"
    id: Resource id
        
    """
    url = f"https://abios-esports.p.rapidapi.com/images/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getassetbyid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the details of a single [Game Asset](/docs/en/knowledge-base/resources/auxiliary#game-assets).
		
		The [Multiple Assets](#operation/Assets-getAssets) operation can be used to
		retrieve multiple assets by id at once using the filter
		`filter=id<{1,2,5,6}`.
		
		## Caching
		
		Game Assets rarely change so they can be cached for hours or even days.
		"
    id: Resource id
        
    """
    url = f"https://abios-esports.p.rapidapi.com/assets/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchespostgameservertimelinesdamagebyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Postgame Server Damage Timeline for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/postgame/server/timelines/damage"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getseriesbytournamentid(is_id: int, order: str=None, skip: int=0, filter: str=None, id_space: str='kambi', take: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all [Series](/docs/en/knowledge-base/resources/series) that
		have been, or are scheduled to be, played in the given
		[Tournament](/docs/en/knowledge-base/resources/tournament)
		
		This is an alias for adding the filter `filter=tournament.id=x` (where `x` is the
		given tournament id) to the already supplied filter and doing a request for
		[Multiple Series](#operation/Series-getSeries).
		"
    id: Resource id
        order: How to order the results
        skip: Amount of results to skip
        filter: How to filter the response.
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        take: Amount of results to take
        
    """
    url = f"https://abios-esports.p.rapidapi.com/tournaments/{is_id}/series"
    querystring = {}
    if order:
        querystring['order'] = order
    if skip:
        querystring['skip'] = skip
    if filter:
        querystring['filter'] = filter
    if id_space:
        querystring['id_space'] = id_space
    if take:
        querystring['take'] = take
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrostersbyplayerid(is_id: int, skip: int=0, filter: str=None, id_space: str='kambi', take: int=50, order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all [Rosters](/docs/en/knowledge-base/resources/competitors#rosters)
		that the given [Player](/docs/en/knowledge-base/resources/competitors#players)
		has been part of.
		
		This is a shorthand for doing a request for [Player Lineups](#operation/Players-getLineupsByPlayerId),
		collecting all the lineup ids, and doing a request for [Multiple Rosters](#operation/Rosters-getRosters)
		with the filter `filter=line_up.id^{x,y,z}`, where `x,y,z` is the list of collected roster ids.
		"
    id: Resource id
        skip: Amount of results to skip
        filter: How to filter the response.
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        take: Amount of results to take
        order: How to order the results
        
    """
    url = f"https://abios-esports.p.rapidapi.com/players/{is_id}/rosters"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if filter:
        querystring['filter'] = filter
    if id_space:
        querystring['id_space'] = id_space
    if take:
        querystring['take'] = take
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchespostgameservertimelinesbombsbyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Postgame Server Bombs Timeline for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/postgame/server/timelines/bombs"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getorganisations(order: str=None, take: int=50, skip: int=0, filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available [Organisations](/docs/en/knowledge-base/resources/competitors#organisations).
		
		## Useful Filters
		
		* `filter=id<={1,2,3,4,5,6,7,10}` will fetch organisations with id 1 through 7 and 10 in a single request
		* `filter=teams.id^{x}` will fetch all organisation that team with id `x` is part of
		"
    order: How to order the results
        take: Amount of results to take
        skip: Amount of results to skip
        filter: How to filter the response.
        
    """
    url = f"https://abios-esports.p.rapidapi.com/organisations"
    querystring = {}
    if order:
        querystring['order'] = order
    if take:
        querystring['take'] = take
    if skip:
        querystring['skip'] = skip
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getgames(filter: str=None, order: str=None, skip: int=0, take: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available [Games](/docs/en/knowledge-base/resources/game).
		"
    filter: How to filter the response.
        order: How to order the results
        skip: Amount of results to skip
        take: Amount of results to take
        
    """
    url = f"https://abios-esports.p.rapidapi.com/games"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if order:
        querystring['order'] = order
    if skip:
        querystring['skip'] = skip
    if take:
        querystring['take'] = take
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmanualstatsbymatchid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available stats that have been manually entered by our editors for players that are participating in a specific match.
		
		Stats are match scoped statistics for various events tied to a player in a roster. Type of stats can vary depending on the game. 
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/manual/stats"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsubstagesbytournamentid(is_id: int, skip: int=0, filter: str=None, order: str=None, take: int=50, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all
		[Substages](/docs/en/knowledge-base/resources/stages) of the
		given [Tournament](/docs/en/knowledge-base/resources/tournament).
		
		This is an alias for adding the filter `filter=tournament.id=x` (where `x` is the
		given tournament id) to the already supplied filter and doing a request for
		[Multiple Substages](#operation/Substages-getSubstages).
		
		Note that [Tournament Stages](#operation/Tournaments-getStagesByTournamentId) returns
		the entire substage resource as well.
		"
    id: Resource id
        skip: Amount of results to skip
        filter: How to filter the response.
        order: How to order the results
        take: Amount of results to take
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/tournaments/{is_id}/substages"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if filter:
        querystring['filter'] = filter
    if order:
        querystring['order'] = order
    if take:
        querystring['take'] = take
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getplayers(skip: int=0, order: str=None, take: int=50, filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available [Players](/docs/en/knowledge-base/resources/competitors#players).
		
		## Useful Filters
		
		* `filter=id<={1,2,3,4,5,6,7,10}` will fetch players with id 1 through 7 and 10 in a single request
		* `filter=nick_name=device` will fetch the players that has "device" as their nickname
		* `filter=nick_name~=de*` will fetch all players with a nickname that starts with "de" and is followed by any number of characters
		* `filter=nick_name~=*dev*` will fetch all players with a nickname that contains the string "dev"
		* `filter=nick_name~=d__` will fetch all players with a nickname that starts with "d" and is followed by exactly two letters
		
		The `also_known_as` identifier will be treated as a scalar so that pattern
		matching can be done against it. This means that regular set-operations
		will not work against it. I.e. something like `filter=also_known_as^{de}`
		is not supported.
		
		* `filter=also_known_as~=de*` will fetch all players with at least one AKA that starts with "de" and is followed by any number of characters
		* `filter=also_known_as~=*dev*` will fetch all players with at least one AKA that contains the string "dev"
		"
    skip: Amount of results to skip
        order: How to order the results
        take: Amount of results to take
        filter: How to filter the response.
        
    """
    url = f"https://abios-esports.p.rapidapi.com/players"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if order:
        querystring['order'] = order
    if take:
        querystring['take'] = take
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstandingrosterbyid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the details of a single [Standing Roster](/docs/en/knowledge-base/resources/competitors#standing-rosters).
		
		The [Team Standing Rosters](#operation/Teams-getStandingRostersByTeamId) and
		[Player Standing Rosters](#operation/Players-getStandingRostersByPlayerId)
		operations will retrieve the standing rosters which a particular team or
		player, respectively, has been a part of.
		"
    id: Resource id
        
    """
    url = f"https://abios-esports.p.rapidapi.com/standingrosters/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getincidents(filter: str=None, take: int=50, skip: int=0, order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    filter: How to filter the response.
        take: Amount of results to take
        skip: Amount of results to skip
        order: How to order the results
        
    """
    url = f"https://abios-esports.p.rapidapi.com/incidents"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if take:
        querystring['take'] = take
    if skip:
        querystring['skip'] = skip
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrostersbyseriesid(is_id: int, order: str=None, filter: str=None, take: int=50, skip: int=0, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all
		[Rosters](/docs/en/knowledge-base/resources/competitors#rosters) that are
		participating in the given [Series](/docs/en/knowledge-base/resources/series).
		
		This is shorthand for collecting all the participating rosters in the
		series, adding the filter `filter=id<={x,y,z}` and doing a request for
		[Multiple Rosters](#operation/Rosters-getRosters).
		"
    id: Resource id
        order: How to order the results
        filter: How to filter the response.
        take: Amount of results to take
        skip: Amount of results to skip
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/series/{is_id}/rosters"
    querystring = {}
    if order:
        querystring['order'] = order
    if filter:
        querystring['filter'] = filter
    if take:
        querystring['take'] = take
    if skip:
        querystring['skip'] = skip
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchespostgameserversummarybyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Postgame Server Summary for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change. However, we do peridiocally update our
		server data parsers to extract new data points and resolve inconsistencies.
		In those cases we will make those improvements available for all matches in
		our database, which means that historical Postgame Server data might
		see improvement even years down the line.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/postgame/server/summary"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsubstagebyid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve information about a single [Substage](/docs/en/knowledge-base/resources/stages).
		
		The [Multiple Substages](#operation/Substages-getSubstages) operation can be used to
		retrieve multiple substages by id at once using the filter
		`filter=id<={1,2,5,6}`.
		
		Retrieving all substages part of a particular tournament can be done using the
		[Multiple Substages](#operation/Substages-getSubstages) operation and the filter
		`filter=tournament.id=x`, where `x` is the tournament id.
		"
    id: Resource id
        
    """
    url = f"https://abios-esports.p.rapidapi.com/substages/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmaps(filter: str=None, skip: int=0, take: int=50, order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available [Maps](/docs/en/knowledge-base/resources/auxiliary#map-and-map-pools).
		
		To see all maps currently considered competitive (if applicable) see [Game Map Pools](#operation/Games-getMapPoolsByGameId).
		
		## Useful Filters
		
		* `filter=id<={1,2,3,4,5,6,7,10}` will fetch maps with id 1 through 7 and 10 in a single request
		* `filter=game.id=5` will fetch all maps in Counter-Strike: Global Offensive
		
		## Caching
		
		Maps rarely change so they can be cached for hours or even days.
		"
    filter: How to filter the response.
        skip: Amount of results to skip
        take: Amount of results to take
        order: How to order the results
        
    """
    url = f"https://abios-esports.p.rapidapi.com/maps"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if skip:
        querystring['skip'] = skip
    if take:
        querystring['take'] = take
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstandingrostersbyplayerid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all [Standing Rosters](/docs/en/knowledge-base/resources/competitors#standing-rosters)
		that the given [Player](/docs/en/knowledge-base/resources/competitors#players) has
		been part of.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/players/{is_id}/standingrosters"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstagesbytournamentid(is_id: int, order: str=None, take: int=50, id_space: str='kambi', filter: str=None, skip: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all
		[Stages](/docs/en/knowledge-base/resources/stages) of the
		given [Tournament](/docs/en/knowledge-base/resources/tournament).
		
		This is an alias for adding the filter `filter=tournament.id=x` (where `x` is the
		given tournament id) to the already supplied filter and doing a request for
		[Multiple Stages](#operation/Stages-getStages).
		"
    id: Resource id
        order: How to order the results
        take: Amount of results to take
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        filter: How to filter the response.
        skip: Amount of results to skip
        
    """
    url = f"https://abios-esports.p.rapidapi.com/tournaments/{is_id}/stages"
    querystring = {}
    if order:
        querystring['order'] = order
    if take:
        querystring['take'] = take
    if id_space:
        querystring['id_space'] = id_space
    if filter:
        querystring['filter'] = filter
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchespostgameapitimelineslevelsbyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Postgame API Levels Timeline for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/postgame/api/timelines/levels"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrosterbyid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the details of a single [Roster](/docs/en/knowledge-base/resources/competitors#rosters).
		
		The [Multiple Rosters](#operation/Rosters-getRosters) operation can be used to
		retrieve multiple rosters by id at once using the filter
		`filter=id<{1,2,5,6}`.
		"
    id: Resource id
        
    """
    url = f"https://abios-esports.p.rapidapi.com/rosters/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getteambyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the details of a single [Team](/docs/en/knowledge-base/resources/competitors#teams).
		
		The [Multiple Teams](#operation/Teams-getTeams) operation can be used to
		retrieve multiple teams by id at once using the filter
		`filter=id<{1,2,5,6}`.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/teams/{is_id}"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getseriesbyrosterid(is_id: int, take: int=50, order: str=None, filter: str=None, skip: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all [Series](/docs/en/knowledge-base/resources/series) that the given
		[Roster](/docs/en/knowledge-base/resources/competitors#rosters) has
		participated in.
		
		This is an alias for adding the filter `filter=participants.roster.id^{x}` (where `x` is the
		given roster id) to the already supplied filter and doing a request for
		[Multiple Series](#operation/Series-getSeries).
		"
    id: Resource id
        take: Amount of results to take
        order: How to order the results
        filter: How to filter the response.
        skip: Amount of results to skip
        
    """
    url = f"https://abios-esports.p.rapidapi.com/rosters/{is_id}/series"
    querystring = {}
    if take:
        querystring['take'] = take
    if order:
        querystring['order'] = order
    if filter:
        querystring['filter'] = filter
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getseriesbyteamid(is_id: int, filter: str=None, skip: int=0, take: int=50, id_space: str='kambi', order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all
		[Series](/docs/en/knowledge-base/resources/series) that the
		given [Team](/docs/en/knowledge-base/resources/competitors#teams) has
		participated in.
		
		This is a shorthand for doing a request for [Team Rosters](#operation/Teams-getRostersByTeamId),
		collecting all the roster ids, and doing a request for [Multiple Series](#operation/Series-getSeries)
		with the filter `filter=participants.roster.id^{x,y,z}`, where `x,y,z` is the
		list of collected roster ids.
		"
    id: Resource id
        filter: How to filter the response.
        skip: Amount of results to skip
        take: Amount of results to take
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        order: How to order the results
        
    """
    url = f"https://abios-esports.p.rapidapi.com/teams/{is_id}/series"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if skip:
        querystring['skip'] = skip
    if take:
        querystring['take'] = take
    if id_space:
        querystring['id_space'] = id_space
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbirthdatebyplayerid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available information about a players birthdate. 
		Can return just Year, Year and Month, or Year, Month, Day.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/players/{is_id}/birthdate"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettournamentbyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the details of a single [Tournament](/docs/en/knowledge-base/resources/tournament).
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/tournaments/{is_id}"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchespostgameapitimelinesexperiencebyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Postgame API Experience Timeline for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/postgame/api/timelines/experience"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getassets(filter: str=None, skip: int=0, take: int=50, order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available [Game Assets](/docs/en/knowledge-base/resources/auxiliary#game-assets).
		
		Game Assets are divided into one `category` each and in some cases further
		divided into a `subcategory`.
		
		## Useful Filters
		
		* `filter=id<={1,2,3,4,5,6,7,10}` will fetch assets with id 1 through 7 and 10 in a single request
		* `filter=game.id=5` will fetch all assets in Counter-Strike: Global Offensive
		* `filter=category=item` will fetch all assets categorized as item
		* `filter=game.id=1,category=item` will fetch all assets categorized as item in Dota 2
		
		Many Play-By-Play data structures only include the Game Asset identifier.
		Using the filter in the first example all of them can be retrieved in a
		single request.
		
		## Caching
		
		Game Assets rarely change so they can be cached for hours or even days.
		"
    filter: How to filter the response.
        skip: Amount of results to skip
        take: Amount of results to take
        order: How to order the results
        
    """
    url = f"https://abios-esports.p.rapidapi.com/assets"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if skip:
        querystring['skip'] = skip
    if take:
        querystring['take'] = take
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcoveragebymatchid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint describes what Play-By-Play data you can expect to find about
		a particular match. You can read more about this
		[here](/docs/en/atlas/usage/coverage).
		"
    id: Resource id
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/coverage"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrostersbyteamid(is_id: int, order: str=None, filter: str=None, id_space: str='kambi', skip: int=0, take: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all
		[Rosters](/docs/en/knowledge-base/resources/competitors#rosters) that the
		given [Team](/docs/en/knowledge-base/resources/competitors#teams) has
		been part of.
		
		This is an alias for adding the filter `filter=team.id=x` (where `x` is the
		given team id) to the already supplied filter and doing a request for
		[Multiple Rosters](#operation/Rosters-getRosters).
		"
    id: Resource id
        order: How to order the results
        filter: How to filter the response.
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        skip: Amount of results to skip
        take: Amount of results to take
        
    """
    url = f"https://abios-esports.p.rapidapi.com/teams/{is_id}/rosters"
    querystring = {}
    if order:
        querystring['order'] = order
    if filter:
        querystring['filter'] = filter
    if id_space:
        querystring['id_space'] = id_space
    if skip:
        querystring['skip'] = skip
    if take:
        querystring['take'] = take
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcountry(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the details of a single [Country](/docs/en/knowledge-base/resources/auxiliary#region).
		
		The [Multiple Countries](#operation/Countries-getCountries) operation can
		be used to retrieve multiple countries by id at once using the filter
		`filter=id<{1,2,5,6}`.
		
		Retrieving all countries within a specific region can be done using the
		[Region Countries](#operation/Countries-getCountriesByRegionId) operation.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/countries/{is_id}"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchespostgameservertimelineseconomybyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Postgame Server Economy Timeline for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/postgame/server/timelines/economy"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdraftbyseriesid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available details of a series draft. 
		A draft is when the participants take turns to pick or ban something related to the game. 
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/series/{is_id}/draft"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrolehistorybyplayerid(is_id: int, order: str=None, filter: str=None, id_space: str='kambi', take: int=50, skip: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve details about which role a
		[Player](/docs/en/knowledge-base/resources/competitors#players) has been known
		to primarily play during a certain period.
		"
    id: Resource id
        order: How to order the results
        filter: How to filter the response.
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        take: Amount of results to take
        skip: Amount of results to skip
        
    """
    url = f"https://abios-esports.p.rapidapi.com/players/{is_id}/rolehistory"
    querystring = {}
    if order:
        querystring['order'] = order
    if filter:
        querystring['filter'] = filter
    if id_space:
        querystring['id_space'] = id_space
    if take:
        querystring['take'] = take
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcasters(order: str=None, take: int=50, filter: str=None, skip: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available [Casters](/docs/en/knowledge-base/resources/auxiliary#casters).
		
		The username listed in the stream object is what the caster is known as on
		that platform. In other words, it is what is needed in order to embed the
		stream.
		
		## Useful Filters
		
		* `filter=id<={1,2,3,4,5,6,7,10}` will fetch casters with id 1 through 7 and 10 in a single request
		* `filter=platform.id=1` will fetch all casters on platform 1, which happen to be `Twitch`
		* `filter=stream.online=1` will fetch all casters that are currently broadcasting
		"
    order: How to order the results
        take: Amount of results to take
        filter: How to filter the response.
        skip: Amount of results to skip
        
    """
    url = f"https://abios-esports.p.rapidapi.com/casters"
    querystring = {}
    if order:
        querystring['order'] = order
    if take:
        querystring['take'] = take
    if filter:
        querystring['filter'] = filter
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatcherealtimesapitimelinesnetworthleadbyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the current state of the Realtime API Net Worth Lead timeline.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available and will reflect
		the latest state of the entire history of
		[net-worth-lead-updated](/docs/api-reference/dota-realtime#tag/dota_realtime_api_events-Team-Net-Worth-Lead-Updated)
		events, with the exception that some match metadata is not included.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/realtime/api/timelines/networthlead"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsocialmediaplatforms(take: int=50, order: str=None, skip: int=0, filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all social media platforms that are supported for players and
		teams.
		"
    take: Amount of results to take
        order: How to order the results
        skip: Amount of results to skip
        filter: How to filter the response.
        
    """
    url = f"https://abios-esports.p.rapidapi.com/socialmedia/platforms"
    querystring = {}
    if take:
        querystring['take'] = take
    if order:
        querystring['order'] = order
    if skip:
        querystring['skip'] = skip
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getscoringbyseriesid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the scoring system for a series played in a
		[battle_royale](/docs/en/knowledge-base/resources/game) game.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/series/{is_id}/scoring"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcasterbyid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the details of a single [Caster](/docs/en/knowledge-base/resources/auxiliary#casters).
		
		The [Multiple Casters](#operation/Casters-getCasters) operation can be used to
		retrieve multiple casters by id at once using the filter
		`filter=id<{1,2,5,6}`.
		
		The username listed in the stream object is what the caster is known as on
		that platform. In other words, it is what is needed in order to embed the
		stream.
		"
    id: Resource id
        
    """
    url = f"https://abios-esports.p.rapidapi.com/casters/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getseriesbylineupid(is_id: int, skip: int=0, take: int=50, order: str=None, filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all [Series](/docs/en/knowledge-base/resources/series) that
		the given [Lineup](/docs/en/knowledge-base/resources/competitors#line-ups) has
		participated in.
		
		This is a shorthand for doing a request for [Lineup Rosters](#operation/Lineups-getRostersByLineupId),
		collecting all the roster ids, and doing a request for [Multiple Series](#operation/Series-getSeries)
		with the filter `filter=participants.roster.id^{x,y,z}`, where `x,y,z` is the
		list of collected roster ids.
		"
    id: Resource id
        skip: Amount of results to skip
        take: Amount of results to take
        order: How to order the results
        filter: How to filter the response.
        
    """
    url = f"https://abios-esports.p.rapidapi.com/lineups/{is_id}/series"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if take:
        querystring['take'] = take
    if order:
        querystring['order'] = order
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsubstagesbystageid(is_id: int, skip: int=0, order: str=None, take: int=50, filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all
		[Substages](/docs/en/knowledge-base/resources/stages) of the
		given [Stage](/docs/en/knowledge-base/resources/stages).
		
		This is an alias for adding the filter `filter=stage.id=x` (where `x` is the
		given stage id) to the already supplied filter and doing a request for
		[Multiple Substages](#operation/Substages-getSubstages).
		
		Note that [Multiple Stages](#operation/Stages-getStages) already return the entire
		substage resource.
		"
    id: Resource id
        skip: Amount of results to skip
        order: How to order the results
        take: Amount of results to take
        filter: How to filter the response.
        
    """
    url = f"https://abios-esports.p.rapidapi.com/stages/{is_id}/substages"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if order:
        querystring['order'] = order
    if take:
        querystring['take'] = take
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrostersbylineupid(is_id: int, order: str=None, take: int=50, skip: int=0, filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all [Rosters](/docs/en/knowledge-base/resources/competitors#rosters)
		that the given
		[Lineup](/docs/en/knowledge-base/resources/competitors#line-ups) has been part
		of.
		
		This is an alias for adding the filter `filter=line_up.id=x` (where `x` is the
		given lineup id) to the already supplied filter and doing a request for
		[Multiple Rosters](#operation/Rosters-getRosters).
		"
    id: Resource id
        order: How to order the results
        take: Amount of results to take
        skip: Amount of results to skip
        filter: How to filter the response.
        
    """
    url = f"https://abios-esports.p.rapidapi.com/lineups/{is_id}/rosters"
    querystring = {}
    if order:
        querystring['order'] = order
    if take:
        querystring['take'] = take
    if skip:
        querystring['skip'] = skip
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrosters(skip: int=0, filter: str=None, order: str=None, take: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available [Rosters](/docs/en/knowledge-base/resources/competitors#rosters).
		
		## Useful Filters
		
		* `filter=id<={1,2,3,4,5,6,7,10}` will fetch rosters with id 1 through 7 and 10 in a single request
		* `filter=team.id=1` will fetch the rosters that team with id `1` is been part of
		"
    skip: Amount of results to skip
        filter: How to filter the response.
        order: How to order the results
        take: Amount of results to take
        
    """
    url = f"https://abios-esports.p.rapidapi.com/rosters"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if filter:
        querystring['filter'] = filter
    if order:
        querystring['order'] = order
    if take:
        querystring['take'] = take
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchespostgameservertimelineslevelsbyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Postgame Server Levels Timeline for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/postgame/server/timelines/levels"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getplayerbyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the details of a single [Player](/docs/en/knowledge-base/resources/competitors#players).
		
		The [Multiple Players](#operation/Players-getPlayers) operation can be used to
		retrieve multiple players by id at once using the filter
		`filter=id<{1,2,5,6}`.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/players/{is_id}"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getregionbycountryid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the details of which region a country is in.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/countries/{is_id}/region"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getregions(order: str=None, filter: str=None, take: int=50, skip: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available [Regions](/docs/en/knowledge-base/resources/auxiliary#regions).
		
		## Caching
		
		Regions rarely change so they can be cached for hours or even days.
		"
    order: How to order the results
        filter: How to filter the response.
        take: Amount of results to take
        skip: Amount of results to skip
        
    """
    url = f"https://abios-esports.p.rapidapi.com/regions"
    querystring = {}
    if order:
        querystring['order'] = order
    if filter:
        querystring['filter'] = filter
    if take:
        querystring['take'] = take
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstages(take: int=50, filter: str=None, order: str=None, skip: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available [Stages](/docs/en/knowledge-base/resources/stages).
		
		Typically this operation would be used to retrieve multiple stages at once
		using the `filter=id<={1,2,5,6}` or the `filter=tournament.id=x` filters,
		rather than iterating all available ones.
		
		## Useful Filters
		
		* `filter=id<={1,2,3,4,5,6,7,10}` will fetch stages with id 1 through 7 and 10 in a single request
		* `filter=tournament.id=1` will fetch all stages that are part of tournament `1`
		"
    take: Amount of results to take
        filter: How to filter the response.
        order: How to order the results
        skip: Amount of results to skip
        
    """
    url = f"https://abios-esports.p.rapidapi.com/stages"
    querystring = {}
    if take:
        querystring['take'] = take
    if filter:
        querystring['filter'] = filter
    if order:
        querystring['order'] = order
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchespostgameservertimelinesexperiencebyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Postgame Server Experience Timeline for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/postgame/server/timelines/experience"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getracesbygameid(is_id: int, skip: int=0, id_space: str='kambi', take: int=50, filter: str=None, order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available playable `Race`s within a
		[Game](/docs/en/knowledge-base/resources/game).
		"
    id: Resource id
        skip: Amount of results to skip
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        take: Amount of results to take
        filter: How to filter the response.
        order: How to order the results
        
    """
    url = f"https://abios-esports.p.rapidapi.com/games/{is_id}/races"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if id_space:
        querystring['id_space'] = id_space
    if take:
        querystring['take'] = take
    if filter:
        querystring['filter'] = filter
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsubstages(filter: str=None, take: int=50, skip: int=0, order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available [Substages](/docs/en/knowledge-base/resources/stages).
		
		Typically this operation would be used to retrieve multiple substages at
		once using the `filter=id<={1,2,5,6}` or the `filter=tournament.id=x`
		filters, rather than iterating all available ones.
		
		## Useful Filters
		
		* `filter=id<={1,2,3,4,5,6,7,10}` will fetch substages with id 1 through 7 and 10 in a single request
		* `filter=tournament.id=1` will fetch all substages that are part of tournament `1`
		* `filter=tournament.id=1,phase=final` will fetch all substages that represents the "final" phase that are part of tournament `1`
		"
    filter: How to filter the response.
        take: Amount of results to take
        skip: Amount of results to skip
        order: How to order the results
        
    """
    url = f"https://abios-esports.p.rapidapi.com/substages"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if take:
        querystring['take'] = take
    if skip:
        querystring['skip'] = skip
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchespostgameapitimelinesobjectivesbyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Postgame API Objectives Timeline for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/postgame/api/timelines/objectives"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getteamspostgameserversummarybyid(is_id: int, to: str=None, distribution: str='avg', id_space: str='kambi', is_from: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Postgame Server Summary for a specific team.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change. However, we do peridiocally update our
		server data parsers to extract new data points and resolve inconsistencies.
		In those cases we will make those improvements available for all teams in
		our database, which means that historical Postgame Server data might
		see improvement even years down the line.
		"
    id: Resource id
        to: End of the time interval
        distribution: 
The distribution to apply to response.

max: returns the maximum value for each datapoint

min: returns the minimum value for each datapoint

avg: returns the average value for each datapoint

Px: returns the value representing the requested percentile. where x is an integer value, x >= 0, x <= 100. e.g. P50

        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        is_from: Start of the time interval
        
    """
    url = f"https://abios-esports.p.rapidapi.com/teams/{is_id}/postgame/server/summary"
    querystring = {}
    if to:
        querystring['to'] = to
    if distribution:
        querystring['distribution'] = distribution
    if id_space:
        querystring['id_space'] = id_space
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcastersbyseriesid(is_id: int, filter: str=None, skip: int=0, id_space: str='kambi', order: str=None, take: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all
		[Casters](/docs/en/knowledge-base/resources/auxiliary#casters) that are
		broadcasting the given [Series](/docs/en/knowledge-base/resources/series).
		
		This returns a list ordered by id with the exception that the first entry will be the primary caster, if any.
		"
    id: Resource id
        filter: How to filter the response.
        skip: Amount of results to skip
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        order: How to order the results
        take: Amount of results to take
        
    """
    url = f"https://abios-esports.p.rapidapi.com/series/{is_id}/casters"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if skip:
        querystring['skip'] = skip
    if id_space:
        querystring['id_space'] = id_space
    if order:
        querystring['order'] = order
    if take:
        querystring['take'] = take
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatches(take: int=50, filter: str=None, id_space: str='kambi', skip: int=0, order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available [Matches](/docs/en/knowledge-base/resources/series#match).
		
		Using the [Multiple Series](#operation/Series-getSeries) as an entrypoint to
		iterate events is, in most cases, preferable. Typically, this operation
		would be used to retrieve multiple specific matches at once. This can be
		done using the filter `filter=id<={1,2,5,6}`.
		
		## Useful Filters
		
		* `filter=id<={1,2,3,4,5,6,7,10}` will fetch matches with id 1 through 7 and 10 in a single request
		* `filter=lifecycle=upcoming` will fetch all matches that are yet to be played
		* `filter=lifecycle=live` will fetch all matches that are currently considered live
		* `filter=series.id=x` will fetch all matches that are part of series with id `x`
		
		## Filtering on Participants and finding Head to Head data
		
		In order to retrieve information about matches where one or several
		particular rosters has participated you can use the following filters.
		
		* `filter=participants.roster.id^{x}` will fetch all matches in which roster with id `x` has participated
		* `filter=participants.roster.id={x,y}` will fetch all matches where the only participants are rosters with id `x` and `y`, i.e. the head-to-head matches between those two rosters
		* `filter=participants.roster.id>={x,y}` will fetch all matches where at least rosters with id `x` and `y` have participated
		* `filter=participants.roster.is^{x,y},participants.roster.id^{a,b}` will fetch all series where roster `(x or y) and (a or b)` participated. Useful if e.g one team has rosters `x` and `y` whereas another team has rosters `a` and `b`. The result would be the teams head-to-head series
		
		Read more about about head-to-head filtering [here](/docs/en/atlas/usage/head-to-head).
		"
    take: Amount of results to take
        filter: How to filter the response.
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        skip: Amount of results to skip
        order: How to order the results
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches"
    querystring = {}
    if take:
        querystring['take'] = take
    if filter:
        querystring['filter'] = filter
    if id_space:
        querystring['id_space'] = id_space
    if skip:
        querystring['skip'] = skip
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getseriesbyplayerid(is_id: int, order: str=None, id_space: str='kambi', filter: str=None, take: int=50, skip: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all [Series](/docs/en/knowledge-base/resources/series) that the given
		[Player](/docs/en/knowledge-base/resources/competitors#players) has
		participated in.
		
		This is a shorthand for doing a request for [Player Lineups](#operation/Players-getLineupsByPlayerId),
		collecting all the lineup ids, doing a request for [Multiple Rosters](#operation/Rosters-getRosters)
		with the filter `filter=line_up.id^{x,y,z}`, where `x,y,z` is the list of collected
		roster ids, and lastly collecting all the roster ids and doing a request for
		[Multiple Series](#operation/Series-getSeries) with the filter
		`filter=participants.roster.id^{a,b,c}`, where `a,b,c` is the list of collected
		roster ids.
		"
    id: Resource id
        order: How to order the results
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        filter: How to filter the response.
        take: Amount of results to take
        skip: Amount of results to skip
        
    """
    url = f"https://abios-esports.p.rapidapi.com/players/{is_id}/series"
    querystring = {}
    if order:
        querystring['order'] = order
    if id_space:
        querystring['id_space'] = id_space
    if filter:
        querystring['filter'] = filter
    if take:
        querystring['take'] = take
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchespostgameservertimelineskillsbyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Postgame Server Kills Timeline for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/postgame/server/timelines/kills"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmapsbygameid(is_id: int, order: str=None, filter: str=None, id_space: str='kambi', take: int=50, skip: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available [Maps](/docs/en/knowledge-base/resources/auxiliary#maps-and-map-pools)
		for the [Game](/docs/en/knowledge-base/resources/game).
		
		This is an alias for adding the filter `filter=game.id=x` (where `x` is the
		given game id) to the already supplied filter and doing a request for
		[Multiple Maps](#operation/Maps-getMaps).
		"
    id: Resource id
        order: How to order the results
        filter: How to filter the response.
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        take: Amount of results to take
        skip: Amount of results to skip
        
    """
    url = f"https://abios-esports.p.rapidapi.com/games/{is_id}/maps"
    querystring = {}
    if order:
        querystring['order'] = order
    if filter:
        querystring['filter'] = filter
    if id_space:
        querystring['id_space'] = id_space
    if take:
        querystring['take'] = take
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatcherealtimesapisummarybyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the current state of the Realtime API Summary for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available and will reflect
		the latest state of any corresponding [Push API Channel](/docs/en/push-api/introduction/available-channels).
		
		For example the latest update on the `dota_realtime_api_states` channel for
		match `x` will be available as `/v3/matches/x/realtime/api/summary`.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/realtime/api/summary"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrostersbysubstageid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all
		[Rosters](/docs/en/knowledge-base/resources/competitors#rosters) that *are
		expected to compete* in the given
		[Substage](/docs/en/knowledge-base/resources/stages)
		"
    id: Resource id
        
    """
    url = f"https://abios-esports.p.rapidapi.com/substages/{is_id}/rosters"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchesbyseriesid(is_id: int, take: int=50, skip: int=0, order: str=None, filter: str=None, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all
		[Matches](/docs/en/knowledge-base/resources/competitors#rosters) that are
		part of the given [Series](/docs/en/knowledge-base/resources/series).
		
		This is an alias for adding the filter `filter=series.id=x` (where `x` is the
		given series id) to the already supplied filter and doing a request for
		[Multiple Matches](#operation/Matches-getMatches).
		"
    id: Resource id
        take: Amount of results to take
        skip: Amount of results to skip
        order: How to order the results
        filter: How to filter the response.
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/series/{is_id}/matches"
    querystring = {}
    if take:
        querystring['take'] = take
    if skip:
        querystring['skip'] = skip
    if order:
        querystring['order'] = order
    if filter:
        querystring['filter'] = filter
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstagebyid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the details of a single [Stage](/docs/en/knowledge-base/resources/stages).
		
		The [Multiple Stages](#operation/Stages-getStages) operation can be used to
		retrieve multiple stages by id at once using the filter
		`filter=id<={1,2,5,6}`.
		
		Retrieving all stages part of a particular tournament can be done using the
		[Multiple Stages](#operation/Stages-getStages) operation and the filter
		`filter=tournament.id=x`, where `x` is the tournament id.
		"
    id: Resource id
        
    """
    url = f"https://abios-esports.p.rapidapi.com/stages/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatcheslivecvsummarybyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the current state of the Live ComputerVision Summary for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available and will reflect
		the latest state of any corresponding [Push API Channel](/docs/en/push-api/introduction/available-channels).
		
		For example the latest update on the `csgo_live_cv_states` channel for
		match `x` will be available as `/v3/matches/x/live/cv/summary`.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/live/cv/summary"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcountries(filter: str=None, order: str=None, skip: int=0, take: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available [Countries](/docs/en/knowledge-base/resources/auxiliary#regions).
		
		All countries in a region can be retrieved using the [Region Countries](#operation/Countries-getCountriesByRegionId)
		operation.
		
		## Useful Filters
		
		* `filter=id<={1,2,3,4,5,6,7,10}` will fetch countries with id 1 through 7 and 10 in a single request
		"
    filter: How to filter the response.
        order: How to order the results
        skip: Amount of results to skip
        take: Amount of results to take
        
    """
    url = f"https://abios-esports.p.rapidapi.com/countries"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if order:
        querystring['order'] = order
    if skip:
        querystring['skip'] = skip
    if take:
        querystring['take'] = take
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstreamsbycasterid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    id: Resource id
        
    """
    url = f"https://abios-esports.p.rapidapi.com/casters/{is_id}/streams"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchespostgameapisummarybyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Postgame API Summary for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/postgame/api/summary"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlineupsbyplayerid(is_id: int, skip: int=0, filter: str=None, order: str=None, take: int=50, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all [Lineups](/docs/en/knowledge-base/resources/competitors#line-ups)
		that the given [Player](/docs/en/knowledge-base/resources/competitors#players)
		has been a part of.
		
		This is an alias for adding the filter `filter=players.id^{x}` (where `x`
		is the given player id) to the already supplied filter and doing a request
		for [Multiple Lineups](#operation/Lineups-getLineups).
		"
    id: Resource id
        skip: Amount of results to skip
        filter: How to filter the response.
        order: How to order the results
        take: Amount of results to take
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/players/{is_id}/lineups"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if filter:
        querystring['filter'] = filter
    if order:
        querystring['order'] = order
    if take:
        querystring['take'] = take
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstandingrostersbyteamid(is_id: int, take: int=50, skip: int=0, filter: str=None, order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all
		[Standing Rosters](/docs/en/knowledge-base/resources/competitors#standing-rosters) that the
		given [Team](/docs/en/knowledge-base/resources/competitors#teams) has
		been part of.
		"
    id: Resource id
        take: Amount of results to take
        skip: Amount of results to skip
        filter: How to filter the response.
        order: How to order the results
        
    """
    url = f"https://abios-esports.p.rapidapi.com/teams/{is_id}/standingrosters"
    querystring = {}
    if take:
        querystring['take'] = take
    if skip:
        querystring['skip'] = skip
    if filter:
        querystring['filter'] = filter
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getseriesbyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the details of a single [Series](/docs/en/knowledge-base/resources/series).
		
		The [Multiple Series](#operation/Series-getSeries) operation can be used to
		retrieve multiple series by id at once using the filter
		`filter=id<={1,2,5,6}`.
		
		Retrieving all series part of a particular tournament can be done using the
		[Multiple Series](#operation/Series-getSeries) operation and the filter
		`filter=tournament.id=x`, where `x` is the tournmanet id.
		
		The [Team Series](#operation/Teams-getSeriesByTeamId), [Lineup Series](#operation/Lineups-getSeriesByLineupId)
		and [Player Series](#operation/Players-getSeriesByPlayerId) can be used to retrieve information
		about series where a specific
		[Team](/docs/en/knowledge-base/resource/competitors#teams),
		[Lineup](/docs/en/knowledge-base/resources/competitors#line-ups) or
		[Player](/docs/en/knowledge-base/resources/competitors#players) has
		participated, respectively.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/series/{is_id}"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getteams(skip: int=0, take: int=50, order: str=None, filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available [Teams](/docs/en/knowledge-base/resources/competitors#teams).
		
		## Useful Filters
		
		* `filter=id<={1,2,3,4,5,6,7,10}` will fetch players with id 1 through 7 and 10 in a single request
		* `filter=name=astralis` will fetch the teams named "astralis"
		* `filter=nick_name~=mou*` will fetch all teams with a name that starts with "mou" and is followed by any number of characters
		* `filter=nick_name~=*mou*` will fetch all teams with a name that contains the string "mou"
		* `filter=nick_name~=d__` will fetch all teams with a name that starts with "d" and is followed by exactly two letters
		"
    skip: Amount of results to skip
        take: Amount of results to take
        order: How to order the results
        filter: How to filter the response.
        
    """
    url = f"https://abios-esports.p.rapidapi.com/teams"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if take:
        querystring['take'] = take
    if order:
        querystring['order'] = order
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmatchespostgameapitimelineseconomybyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Postgame API Economy Timeline for a specific match.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change.
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/postgame/api/timelines/economy"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlineups(order: str=None, filter: str=None, skip: int=0, take: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available [Lineups](/docs/en/knowledge-base/resources/competitors#line-ups).
		
		## Useful Filters
		
		* `filter=id<={1,2,3,4,5,6,7,10}` will fetch lineups with id 1 through 7 and 10 in a single request
		* `filter=players.id={1,2,3,4,5}` will fetch the lineup that makes up players `1` through `5`
		* `filter=players.id^{1}` will fetch all lineups that player `1` is part of
		* `filter=players.id>={1,2}` will fetch all lineups that both player `1` and `2` are part of, including the one that consists of only player `1` and `2`
		"
    order: How to order the results
        filter: How to filter the response.
        skip: Amount of results to skip
        take: Amount of results to take
        
    """
    url = f"https://abios-esports.p.rapidapi.com/lineups"
    querystring = {}
    if order:
        querystring['order'] = order
    if filter:
        querystring['filter'] = filter
    if skip:
        querystring['skip'] = skip
    if take:
        querystring['take'] = take
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmapbyid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the details of a single [Map](/docs/en/knowledge-base/resources/auxiliary#map-and-map-pools).
		
		## Caching
		
		Maps rarely change so they can be cached for hours or even days.
		"
    id: Resource id
        
    """
    url = f"https://abios-esports.p.rapidapi.com/maps/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getteamspostgameapisummarybyid(is_id: int, is_from: str=None, to: str=None, id_space: str='kambi', distribution: str='avg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the Postgame API Summary for a specific team.
		
		You can read more about Coverage [here](/docs/en/atlas/usage/coverage).
		
		This endpoint is updated as soon as the data is available. Once in place
		this data is unlikely to change.
		"
    id: Resource id
        is_from: Start of the time interval
        to: End of the time interval
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        distribution: 
The distribution to apply to response.

max: returns the maximum value for each datapoint

min: returns the minimum value for each datapoint

avg: returns the average value for each datapoint

Px: returns the value representing the requested percentile. where x is an integer value, x >= 0, x <= 100. e.g. P50

        
    """
    url = f"https://abios-esports.p.rapidapi.com/teams/{is_id}/postgame/api/summary"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    if id_space:
        querystring['id_space'] = id_space
    if distribution:
        querystring['distribution'] = distribution
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getorganisationbyid(is_id: int, filter: str=None, skip: int=0, order: str=None, take: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the details of a single [Organisation](/docs/en/knowledge-base/resources/competitors#organisations).
		"
    id: Resource id
        filter: How to filter the response.
        skip: Amount of results to skip
        order: How to order the results
        take: Amount of results to take
        
    """
    url = f"https://abios-esports.p.rapidapi.com/organisations/{is_id}"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if skip:
        querystring['skip'] = skip
    if order:
        querystring['order'] = order
    if take:
        querystring['take'] = take
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlineupbyid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the details of a single [Lineup](/docs/en/knowledge-base/resources/competitors#line-ups).
		
		The [Multiple Lineups](#operation/Lineups-getLineups) operation can be used
		to retrieve multiple lineups by id at once using the filter
		`filter=id<{1,2,5,6}`.
		"
    id: Resource id
        
    """
    url = f"https://abios-esports.p.rapidapi.com/lineups/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcountriesbyregionid(is_id: int, skip: int=0, take: int=50, filter: str=None, order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all Countries that are part of a given Region.
		
		Read more about regions and countries [here](/docs/en/knowledge-base/resources/auxiliary#regions)
		"
    id: Resource id
        skip: Amount of results to skip
        take: Amount of results to take
        filter: How to filter the response.
        order: How to order the results
        
    """
    url = f"https://abios-esports.p.rapidapi.com/regions/{is_id}/countries"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if take:
        querystring['take'] = take
    if filter:
        querystring['filter'] = filter
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getparticipantsbytournamentid(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all the participants of a [Tournament](/docs/en/knowledge-base/resources/tournament).
		"
    id: Resource id
        
    """
    url = f"https://abios-esports.p.rapidapi.com/tournaments/{is_id}/participants"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getseriesbysubstageid(is_id: int, take: int=50, skip: int=0, filter: str=None, order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all [Series](/docs/en/knowledge-base/resources/series) that
		have been, or are schedule to be, played in the given
		[Substage](/docs/en/knowledge-base/resources/stages)
		
		This is an alias for adding the filter `filter=substage.id=x` (where `x` is the
		given substage id) to the already supplied filter and doing a request for
		[Multiple Series](#operation/Series-getSeries).
		"
    id: Resource id
        take: Amount of results to take
        skip: Amount of results to skip
        filter: How to filter the response.
        order: How to order the results
        
    """
    url = f"https://abios-esports.p.rapidapi.com/substages/{is_id}/series"
    querystring = {}
    if take:
        querystring['take'] = take
    if skip:
        querystring['skip'] = skip
    if filter:
        querystring['filter'] = filter
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getgamebyid(is_id: int, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the details of a single [Game](/docs/en/knowledge-base/resources/game).
		"
    id: Resource id
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/games/{is_id}"
    querystring = {}
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrostersbymatchid(is_id: int, skip: int=0, order: str=None, filter: str=None, take: int=50, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all [Rosters](/docs/en/knowledge-base/resources/competitors#rosters)
		that are participating in the given
		[Match](/docs/en/knowledge-base/resources/series#match).
		
		This is shorthand for collecting all the participating rosters in the
		match, adding the filter `filter=id<={x,y,z}` and doing a request for
		[Multiple Rosters](#operation/Rosters-getRosters).
		"
    id: Resource id
        skip: Amount of results to skip
        order: How to order the results
        filter: How to filter the response.
        take: Amount of results to take
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/matches/{is_id}/rosters"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if order:
        querystring['order'] = order
    if filter:
        querystring['filter'] = filter
    if take:
        querystring['take'] = take
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettournaments(filter: str=None, skip: int=0, take: int=50, order: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all available [Tournaments](/docs/en/knowledge-base/resources/tournaments).
		
		## Useful Filters
		
		* `filter=id<={1,2,3,4,5,6,7,10}` will fetch tournaments with id 1 through 7 and 10 in a single request
		* `filter=tier=1` will fetch all tournaments that are considered tier 1
		* `filter=start>=2020-10-31T00:00:00Z` will fetch all tournaments that start at "2020-10-31T00:00:00" or later
		"
    filter: How to filter the response.
        skip: Amount of results to skip
        take: Amount of results to take
        order: How to order the results
        
    """
    url = f"https://abios-esports.p.rapidapi.com/tournaments"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if skip:
        querystring['skip'] = skip
    if take:
        querystring['take'] = take
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcastersbytournamentid(is_id: int, filter: str=None, order: str=None, skip: int=0, take: int=50, id_space: str='kambi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve
		[Casters](/docs/en/knowledge-base/resources/auxiliary#casters) that are
		expected to broadcast [Series](/docs/en/knowledge-base/resources/series) in the given
		[Tournament](/docs/en/knowledge-base/resources/tournament).
		
		This is shorthand for collecting all the tournament caster ids, adding the
		filter `filter=id<={x,y,z}` and doing a request for [Multiple Casters](#operation/Casters-getCasters).
		"
    id: Resource id
        filter: How to filter the response.
        order: How to order the results
        skip: Amount of results to skip
        take: Amount of results to take
        id_space: Which id space the given identifier is part of. This parameter is only relevant if you are querying the API using a third-party identifier such as one obtained from an odds or media provider. By default any identifier will be considered part of the Abios id space. You can read more about this [here](/docs/en/atlas/usage/external-ids).
        
    """
    url = f"https://abios-esports.p.rapidapi.com/tournaments/{is_id}/casters"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    if order:
        querystring['order'] = order
    if skip:
        querystring['skip'] = skip
    if take:
        querystring['take'] = take
    if id_space:
        querystring['id_space'] = id_space
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "abios-esports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

