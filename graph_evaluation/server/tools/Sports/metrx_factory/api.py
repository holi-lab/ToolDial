import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of countries for which teams and metrics are available.
		There are no parameters to provide.
		
		For free subscriptions the following limits apply:
		
		`Max. requests per hour: 20`
		`Min. time between requests: 1 minute`
		`Max. countries returned: 5`"
    
    """
    url = f"https://metrx-factory.p.rapidapi.com/v1/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metrx-factory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches(teamid: str='J13iU6cGNJKdcgL-tDSCig', teamids: str=None, maxstart: str=None, venue: str=None, teamtype: str=None, competitionstageid: str=None, minstart: str='2022-04-15T12:00', estimatebill: bool=None, timezone: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of matches for given parameters.
		
		Both historical and upcoming matches are supported. If no parameters are provided then all upcoming matches are returned. The time span covering next matches ranges between 8 hours for *Basic* and 60 hours for paid subscriptions.
		
		For free subscriptions the following limits apply:
		
		`Max. requests per hour: 20`
		`Min. time between requests: 30 seconds`
		`Max. matches returned: 5`"
    teamid: The identifier of a team competing in a match. Use the `Get Teams` operation to find valid identifiers.
        teamids: A comma-separated list of two team identifiers. For head-to-head matches only.
        maxstart: The upper bound of the time span in which matches may start (inclusive).
**Pattern**: `yyyy-MM-dd['T'HH:mm]`
        venue: The venue where the team referenced by the `teamId `parameter competes in a match (case-insensitive).
**Allowed**: `H` Home, `A` Away
        teamtype: The type of teams that compete in a match (case-insensitive).
**Allowed**: `C` Club team, `N` National team
        competitionstageid: The identifier of the competition stage to which a match is related. Use the `Get Competitions` operation to find valid identifiers.
        minstart: The lower bound of the time span in which matches may start (inclusive).
**Pattern**: `yyyy-MM-dd['T'HH:mm]`
        estimatebill: Flag whether a cost estimate should be returned for the above parameters instead of the result.
**Default**: false
        timezone: The time zone used to interprete the `minStart` and `maxStart` parameters and to format date values in the API response.
**Default**: UTC
        
    """
    url = f"https://metrx-factory.p.rapidapi.com/v1/matches"
    querystring = {}
    if teamid:
        querystring['teamId'] = teamid
    if teamids:
        querystring['teamIds'] = teamids
    if maxstart:
        querystring['maxStart'] = maxstart
    if venue:
        querystring['venue'] = venue
    if teamtype:
        querystring['teamType'] = teamtype
    if competitionstageid:
        querystring['competitionStageId'] = competitionstageid
    if minstart:
        querystring['minStart'] = minstart
    if estimatebill:
        querystring['estimateBill'] = estimatebill
    if timezone:
        querystring['timeZone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metrx-factory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions(estimatebill: bool=None, namelike: str='champ', year: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of competitions that match the given parameters.
		Competitions are divided into stages which differ in playing mode (league vs. knock-out). If no parameters are provided then all competitions hold in this year are returned.
		
		For free subscriptions the following limits apply:
		
		`Max. requests per hour: 20`
		`Min. time between requests: 30 seconds`
		`Max. competitions returned: 5`"
    estimatebill: Flag whether a cost estimate should be returned for the above parameters instead of the result.
**Default**: false
        namelike: The full or partial name of a competition (case-insensitive).
**Min. length**: 4 chars

Refer to [API reference](https://docs.metrxfactory.io/competitions.htm) for a list of available competitions.
        year: The year a competition is hold. If a competition ends in another year that it was started then any included year matches.
E.g. 2020 matches competitions from 2019-20, 2020 and 2020-21.
        
    """
    url = f"https://metrx-factory.p.rapidapi.com/v1/competitions"
    querystring = {}
    if estimatebill:
        querystring['estimateBill'] = estimatebill
    if namelike:
        querystring['nameLike'] = namelike
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metrx-factory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams(estimatebill: bool=None, competitionstageid: str=None, countryid: str=None, namelike: str='liverp', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of teams that match the given parameters.
		At least one of the request parameters is required.
		
		For free subscriptions the following limits apply:
		
		`Max. requests per hour: 20`
		`Min. time between requests: 30 seconds`
		`Max. teams returned: 5`"
    estimatebill: Flag whether a cost estimate should be returned for the above parameters instead of the result.
**Default**: false
        competitionstageid: The identifier of the competition stage in which teams compete. Use the `Get Competitions` operation to find valid identifiers.
        countryid: The identifier of the country which the teams come from. Use the `Get Countries` operation to find valid identifiers.
        namelike: The full or partial name of a team (case-insensitive).
**Min. length**: 4 chars
        
    """
    url = f"https://metrx-factory.p.rapidapi.com/v1/teams"
    querystring = {}
    if estimatebill:
        querystring['estimateBill'] = estimatebill
    if competitionstageid:
        querystring['competitionStageId'] = competitionstageid
    if countryid:
        querystring['countryId'] = countryid
    if namelike:
        querystring['nameLike'] = namelike
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metrx-factory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_odds_history(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the trend of average market odds for a single match.
		
		Still in beta phase this endpoint will be soon available for the *Mega* subscription. Note that testing with the *Basic* subscription will result in a 403 HTTP error (Forbidden)."
    
    """
    url = f"https://metrx-factory.p.rapidapi.com/v1/market/match-odds-history"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metrx-factory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competition_performance_indices(format: str=None, maxrank: int=None, minrank: int=None, teamtype: str=None, domestic: bool=None, timezone: str=None, estimatebill: bool=None, date: str='2022-05-30T12:00', competitionids: str=None, competitionid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a snapshot of global competition performance indices sorted by rank in ascending order.
		Separate index lists are maintained for club/national team and domestic/international competitions.
		
		If no parameters are provided then the latest club team competitions ranking is returned in absolute index format. For details about index calculation and factors refer to the documentation.
		
		For free subscriptions the following limits apply:
		
		`Max. requests per hour: 20`
		`Min. time between requests: 1 minute`
		`Max. performances returned: 5`"
    format: The format applied to index values (case-insensitive).
**Allowed**: `ABS` Absolute, `REL` Relative.
**Default**: ABS
BASIC subscriptions support only `ABS`.
        maxrank: The maximum rank of performances according to index value (inclusive).
        minrank: The minimum rank of performances according to index value (inclusive). Minimum value for BASIC subscriptions is 31.
        teamtype: The type of teams for which performances should be included (case-insensitive).
**Allowed**: `C` Club team, `N` National team
**Default**: C
        domestic: Status whether included performances should include domestic or international competitions.
        timezone: The time zone used to interpret the `date` parameter and to format date values in the API response.
**Default**: UTC
        estimatebill: Flag whether a cost estimate should be returned for the above parameters instead of the result.
**Default**: false
        date: The date for which performance indices should be evaluated.
**Pattern**: `yyyy-MM-dd['T'HH:mm]`
**Default**: Now
        competitionids: Alternative to the `competitionId` parameter. A comma-separated list of competition identifiers for which performances should be included. Use the `Get Competitions` operation to find valid identifiers.
        competitionid: The identifier of a competition for which the performance should be included. Use the `Get Competitions` operation to find valid identifiers.
        
    """
    url = f"https://metrx-factory.p.rapidapi.com/v1/performance-index/competitions"
    querystring = {}
    if format:
        querystring['format'] = format
    if maxrank:
        querystring['maxRank'] = maxrank
    if minrank:
        querystring['minRank'] = minrank
    if teamtype:
        querystring['teamType'] = teamtype
    if domestic:
        querystring['domestic'] = domestic
    if timezone:
        querystring['timeZone'] = timezone
    if estimatebill:
        querystring['estimateBill'] = estimatebill
    if date:
        querystring['date'] = date
    if competitionids:
        querystring['competitionIds'] = competitionids
    if competitionid:
        querystring['competitionId'] = competitionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metrx-factory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_performance_indices(projections: str='I,LIC,LMD', format: str=None, minrank: int=None, teamid: str=None, teamtype: str=None, countryid: str=None, competitionstageid: str=None, maxrank: int=None, timezone: str=None, date: str='2022-05-30T12:00', projection: str=None, estimatebill: bool=None, teamids: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a snapshot of global team performance indices sorted by rank in ascending order.
		Separate index lists are maintained for club and national teams. The indication of projections controls which data should be collected and has direct impact on the payload and billed credits.
		
		If no parameters are provided then the latest club team ranking is returned with only the main index projection in absolute index format. For details about index calculation and factors refer to the documentation.
		
		For free subscriptions the following limits apply:
		
		`Max. requests per hour: 20`
		`Min. time between requests: 1 minute`
		`Max. performances returned: 5`"
    projections: Alternative to the `projection` parameter. A comma-separated list of projections applied when building the result (case-insensitive).
        format: The format applied to index values (case-insensitive).
**Allowed**: `ABS` Absolute, `REL` Relative.
**Default**: ABS
BASIC subscriptions support only `ABS`.
        minrank: The minimum rank of performances according to main index value (inclusive). Minimum value for BASIC subscriptions is 51.
        teamid: The identifier of a team for which the performance should be included. Use the `Get Teams` operation to find valid identifiers.
        teamtype:  The type of teams for which performances should be included (case-insensitive).
**Allowed**: `C` Club team, `N` National team
**Default**: C
        countryid: The identifier of a country whose teams are included in the result. Use the `Get Countries` operation to find valid identifiers.
        competitionstageid: The identifier of a competition stage whose competing teams are included in the result. Use the `Get Competitions` operation to find valid identifiers.
        maxrank: The maximum rank of performances according to main index value (inclusive).
        timezone: The time zone used to interpret the `date` parameter and to format date values in the API response.
**Default**: UTC
        date: The date for which performance indices should be evaluated.
**Pattern**: `yyyy-MM-dd['T'HH:mm]`
**Default**: Now
        projection: A single projection applied when building the result (case-insensitive).
**Allowed**: `I` Main index, `VI` Venue index, `IT` Index trend, `LIC` Last index change, `LMD` Last match details
**Default**: I

Refer to [API reference](https://docs.metrxfactory.io/#team-pfm-idx-proj-enum) for further details.
        estimatebill: Flag whether a cost estimate should be returned for the above parameters instead of the result.
**Default**: false
        teamids: An alternative to the `teamId` parameter. A comma-separated list of team identifiers for which performances should be included. Use the `Get Teams` operation to find valid identifiers.
        
    """
    url = f"https://metrx-factory.p.rapidapi.com/v1/performance-index/teams"
    querystring = {}
    if projections:
        querystring['projections'] = projections
    if format:
        querystring['format'] = format
    if minrank:
        querystring['minRank'] = minrank
    if teamid:
        querystring['teamId'] = teamid
    if teamtype:
        querystring['teamType'] = teamtype
    if countryid:
        querystring['countryId'] = countryid
    if competitionstageid:
        querystring['competitionStageId'] = competitionstageid
    if maxrank:
        querystring['maxRank'] = maxrank
    if timezone:
        querystring['timeZone'] = timezone
    if date:
        querystring['date'] = date
    if projection:
        querystring['projection'] = projection
    if estimatebill:
        querystring['estimateBill'] = estimatebill
    if teamids:
        querystring['teamIds'] = teamids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metrx-factory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_metrics(maxstart: str=None, teamid: str=None, teamtype: str=None, configurations: str='OL:5,OF:PB', acceptnometric: bool=None, projection: str=None, minstart: str=None, timezone: str=None, competitionstageid: str=None, estimatebill: bool=None, configuration: str=None, projections: str='MD,TI,CI,XG,XH,XP', matchid: str='RpnikTcEMruYxLgFA3irZg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the metrics of a single match, a specific team or matches within a time span.
		
		This is the main endpoint for retrieving match related figures such as performance ratios, expected and actual scores, statistically fair and market odds. The indication of projections controls which data should be collected and has direct impact on the payload and billed credits.
		
		If no parameters are provided then the performance index of competing teams *(TI)*, expected goals *(XG)* and expected moneyline odds *(XM)* along with match details *(MD)* are returned for upcoming matches. The time span covering next matches ranges between 8 hours for *Basic* and 60 hours for paid subscriptions.
		
		For free subscriptions the following limits apply:
		
		`Max. requests per hour: 20`
		`Min. time between requests: 1 minute`
		`Max. matches returned: 3`"
    maxstart: The upper bound of the time span in which matches may start (inclusive).
**Pattern**: `yyyy-MM-dd['T'HH:mm]`
        teamid: The identifier of a team for which metrics should be evaluated. Use the `Get Teams` operation to find valid identifiers.
        teamtype: The type of teams for which the metrics should be evaluated (case-insensitive).
**Allowed**: `C` Club team, `N` National team
**Default**: C
        configurations: Alternative to the `configuration` parameter. A comma-separated list of key/value pair configurations applied to selected projections (case-insensitive).
        acceptnometric: Flag whether the response should include matches for which metrics are not available for the requested projections.
**Default**: false
        projection: A single projection applied when building the result (case-insensitive).
**Allowed**: `MD` Match details, `TI` Team index, `TIV` Team venue index, `TIT` Team index trend, `CI` Competition index, `XG` Expected goals, `XS` Expected shots, `XC` Expected corners, `CG` Actual goals, `CS` Actual shots, `CC` Actual corners, `SP` Score probabilities, `XM` Expected moneyline, `XH` Expected handicaps, `XP` Expected points, `CM` Actual moneyline, `CH` Actual handicaps, `CP` Actual points

Refer to [API reference](https://docs.metrxfactory.io/#metrics-proj-enum) for further details.
        minstart: The lower bound of the time span in which matches may start (inclusive).
**Pattern**: `yyyy-MM-dd['T'HH:mm]`
        timezone: The time zone used to interprete the `minStart` and `maxStart` parameters and to format date values in the API response.
**Default**: UTC
        competitionstageid: The identifier of a competition stage for which metrics should be evaluated. Use the `Get Competitions` operation to find valid identifiers.
        estimatebill: Flag whether a cost estimate should be returned for the above parameters instead of the result.
**Default**: false
        configuration: A single key/value pair configuration applied to selected projections (case-insensitive).
**Pattern**: `{key}:{value}`
**Allowed keys**: `PIF` Performance index format, `XSQ` Expected scores quality, `SPM` Minimum scores probability, `OF` Odds format, `OL` Odds lines, `XOM` Expected odds margin. 

Refer to [API reference](https://docs.metrxfactory.io/#metric-attr-enum) for further details.
        projections: Alternative to the `projection` parameter. A comma-separated list of projections applied when building the result (case-insensitive).
**Default**: MD,TI,XG,CG,XM
        matchid: The identifier of a match for which the metrics should be evaluated. Use the `Get Matches` operation to find valid identifiers.
        
    """
    url = f"https://metrx-factory.p.rapidapi.com/v1/match-metrics"
    querystring = {}
    if maxstart:
        querystring['maxStart'] = maxstart
    if teamid:
        querystring['teamId'] = teamid
    if teamtype:
        querystring['teamType'] = teamtype
    if configurations:
        querystring['configurations'] = configurations
    if acceptnometric:
        querystring['acceptNoMetric'] = acceptnometric
    if projection:
        querystring['projection'] = projection
    if minstart:
        querystring['minStart'] = minstart
    if timezone:
        querystring['timeZone'] = timezone
    if competitionstageid:
        querystring['competitionStageId'] = competitionstageid
    if estimatebill:
        querystring['estimateBill'] = estimatebill
    if configuration:
        querystring['configuration'] = configuration
    if projections:
        querystring['projections'] = projections
    if matchid:
        querystring['matchId'] = matchid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metrx-factory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_match_metrics(metric: str, projection: str=None, order: str=None, filter: str=None, projections: str=None, start: str=None, configurations: str=None, competitionstageid: str=None, configuration: str=None, maxcount: int=None, estimatebill: bool=None, filters: str=None, teamid: str='k4IoQd5BNiubRLwmHKuIow', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the metrics of matches with top values regarding a certain metric.
		
		This is the premium endpoint for goal-oriented search of match related figures and is available only for the *Mega* subscription. It allows custom logic for match selection and supports functions used to join metrics. Finding the teams with highest conversion rates or market lines with the highest discrepancies to statistical odds are simple tasks by means of this operation.
		
		Furthermore, it may save you credits in comparison to the *Match Metrics* operation as only a filtered subset will be fetched. By default, the Top-10 upcoming matches are returned. Historical top searches are supported for specific teams and competition stages as well.
		
		Note that testing this endpoint with the *Basic* subscription will result in a 403 HTTP error (Forbidden). Instead, refer to the [docs](https://docs.metrxfactory.io/#get-/match-metrics/top) to learn about the features offered by the endpoint."
    metric: An individual metric or expression that describes the top criteria used to sort matches. Expressions support functions, metrics and decimal numbers for custom calculation.

Refer to [API reference](https://docs.metrxfactory.io/#metric-enum) for further details.
        projection: A single projection applied when building the result (case-insensitive).
**Allowed**: `MD` Match details, `TI` Team index, `TIV` Team venue index, `TIT` Team index trend, `CI` Competition index, `XG` Expected goals, `XS` Expected shots, `XC` Expected corners, `CG` Actual goals, `CS` Actual shots, `CC` Actual corners, `SP` Score probabilities, `XM` Expected moneyline, `XH` Expected handicaps, `XP` Expected points, `CM` Actual moneyline, `CH` Actual handicaps, `CP` Actual points

Refer to [API reference](https://docs.metrxfactory.io/#metrics-proj-enum) for further details.
        order: The order used to sort matches by relevance (case-insensitive).
Allowed: `ASC` Ascending, `DESC` Descending
**Default**: DESC
        filter: A single metric/operator/metric triple building a predicate that must be true in order to evalute a match.

Refer to [API reference](https://docs.metrxfactory.io/#get-/match-metrics/top) for further details.
        projections: Alternative to the `projection` parameter. A comma-separated list of projections applied when building the result (case-insensitive).
        start: The time when matches to evaluate start(ed).
**Allowed**: `P` Past, `U` Upcoming
**Default**: U
        configurations: Alternative to the `configuration` parameter. A comma-separated list of key/value pair configurations applied to selected projections (case-insensitive).
        competitionstageid: The identifier of a competition stage for which metrics should be evaluated. Use the `Get Competitions` operation to find valid identifiers.
        configuration: A single key/value pair configuration applied to selected projections (case-insensitive).
Pattern: `{key}:{value}`
Allowed keys: `PIF` Performance index format, `XSQ` Expected scores quality, `SPM` Minimum scores probability, `OF` Odds format, `OL` Odds lines, `XOM` Expected odds margin. 

Refer to [API reference](https://docs.metrxfactory.io/#metric-attr-enum) for further details.
        maxcount: The maximum number of match metrics returned.
**Default**: 10
        estimatebill: Flag whether a cost estimate should be returned for the above parameters instead of the result.
**Default**: false
        filters: Alternative to the `filter` parameter. A comma-separated list of metric/operator/metric triples building predicates that must be true in order to evalute a match.
        teamid: The identifier of a team for which metrics should be evaluated. Use the `Get Teams` operation to find valid identifiers.
        
    """
    url = f"https://metrx-factory.p.rapidapi.com/v1/match-metrics/top"
    querystring = {'metric': metric, }
    if projection:
        querystring['projection'] = projection
    if order:
        querystring['order'] = order
    if filter:
        querystring['filter'] = filter
    if projections:
        querystring['projections'] = projections
    if start:
        querystring['start'] = start
    if configurations:
        querystring['configurations'] = configurations
    if competitionstageid:
        querystring['competitionStageId'] = competitionstageid
    if configuration:
        querystring['configuration'] = configuration
    if maxcount:
        querystring['maxCount'] = maxcount
    if estimatebill:
        querystring['estimateBill'] = estimatebill
    if filters:
        querystring['filters'] = filters
    if teamid:
        querystring['teamId'] = teamid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metrx-factory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_performance_index_history(teamid: str, estimatebill: bool=None, projection: str=None, mindate: str='2022-01-15T12:00', maxdate: str=None, timezone: str=None, projections: str='I,LIC,LMD', format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the history of a team's performance index. Each datapoint represents an index change based on a match performance. The indication of projections controls which data should be collected and has direct impact on the payload and billed credits.
		
		There is one obligatory parameter which identifies the team of the trend. If no more parameters are provided then the trend for the last year's performance (*Basic* subscriptions: one month) is returned with only the main index projection in absolute index format. For details about index calculation and factors refer to the documentation.
		
		For free subscriptions the following limits apply:
		
		`Max. requests per hour: 20`
		`Min. time between requests: 1 minute`
		`Max. time span returned: 1 month`"
    teamid: The identifier of the team for which the performances should be collected. Use the `Get Teams` operation to find valid identifiers.
        estimatebill: Flag whether a cost estimate should be returned for the above parameters instead of the result.
**Default**: false
        projection: A single projection applied when building the result (case-insensitive).
**Allowed**: `I` Main index, `VI` Venue index, `IT` Index trend, `LIC` Last index change, `LMD` Last match details
**Default**: I

Refer to [API reference](https://docs.metrxfactory.io/#team-pfm-idx-proj-enum) for further details.
        mindate: The lower bound of the history's time span (inclusive).
**Pattern**: `yyyy-MM-dd['T'HH:mm]`
        maxdate: The upper bound of the history's time span (inclusive).
**Pattern**: `yyyy-MM-dd['T'HH:mm]`
        timezone: The time zone used to interpret the `minDate` and `maxDate` parameters.
**Default**: UTC
        projections: Alternative to the `projection` parameter. A comma-separated list of projections applied when building the result (case-insensitive).
        format: The format applied to index values (case-insensitive).
**Allowed**: `ABS` Absolute, `REL` Relative.
**Default**: ABS
BASIC subscriptions support only `ABS`.
        
    """
    url = f"https://metrx-factory.p.rapidapi.com/v1/performance-index/team-history"
    querystring = {'teamId': teamid, }
    if estimatebill:
        querystring['estimateBill'] = estimatebill
    if projection:
        querystring['projection'] = projection
    if mindate:
        querystring['minDate'] = mindate
    if maxdate:
        querystring['maxDate'] = maxdate
    if timezone:
        querystring['timeZone'] = timezone
    if projections:
        querystring['projections'] = projections
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metrx-factory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

