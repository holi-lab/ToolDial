import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def team_statistics(team: str, type_of_statistics: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Team statistics"
    type_of_statistics: **Enter one from available types of statistics:**
exact number of goals in the match, 
result first half and the match,
goals over, 
goals under, 
home vs away full time result, 
full time result
        
    """
    url = f"https://football-dolphin.p.rapidapi.com/teamstatistics/{type_of_statistics}/{team}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-dolphin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_season_statistics(type_of_statistics: str, season: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Football season statistics"
    type_of_statistics: **Enter one from available types of statistics:**
all scores,
exact number of goals in the match,
goals over,
goals under,
home vs away full time result,
home vs away result first half and the match
        season: **Enter one season from all available seasons:**
1995/96, 1996/97, 1997/98, 1999/00, 2000/01, 2001/02, 2002/03, 2003/04, 2004/05, 2005/06, 2006/07, 2007/08, 2008/09, 2009/10, 2010/11, 2011/12, 2012/13, 2013/14, 2014/15, 2015/16, 2016/17, 2017/18, 2018/19, 2019/20, 2020/21, 2021/22
        
    """
    url = f"https://football-dolphin.p.rapidapi.com/footballseasonstatistics/{type_of_statistics}/{season}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-dolphin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def head_to_head_statistics(first_team: str, second_team: str, type_of_statistics: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Head to head statistics"
    first_team: **Enter first team from all available teams:** Arsenal, Aston Villa, Barnsley, Birmingham, Blackburn, Blackpool, Bolton, Bournemouth, Bradford, Brighton, Burnley, Cardiff, Charlton, Chelsea, Coventry, Crystal Palace, Derby, Everton, Fulham, Huddersfield, Hull, Ipswich, Leeds, Leicester, Liverpool, Man City, Man United, Middlesbrough, Newcastle, Norwich, Nott'm Forest, Portsmouth, QPR, Reading, Sheffield United, Sheffield Weds, Southampton, Stoke, Sunderland, Swansea, Tottenham, Watford, West Brom, West Ham, Wigan, Wimbledon, Wolves
        second_team: **Enter second team from all available teams:** Arsenal, Aston Villa, Barnsley, Birmingham, Blackburn, Blackpool, Bolton, Bournemouth, Bradford, Brighton, Burnley, Cardiff, Charlton, Chelsea, Coventry, Crystal Palace, Derby, Everton, Fulham, Huddersfield, Hull, Ipswich, Leeds, Leicester, Liverpool, Man City, Man United, Middlesbrough, Newcastle, Norwich, Nott'm Forest, Portsmouth, QPR, Reading, Sheffield United, Sheffield Weds, Southampton, Stoke, Sunderland, Swansea, Tottenham, Watford, West Brom, West Ham, Wigan, Wimbledon, Wolves
        type_of_statistics: **Enter one from available types of statistics:** 
full time result, 
home vs away full time result, 
result first half and the match,
exact number of goals in the match, 
goals over, 
goals under
        
    """
    url = f"https://football-dolphin.p.rapidapi.com/headtoheadstatistics/"
    querystring = {'first_team': first_team, 'second_team': second_team, 'type_of_statistics': type_of_statistics, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-dolphin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

