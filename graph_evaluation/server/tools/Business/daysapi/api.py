import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def calendar_arithmetic(seconds: int=0, months: int=0, hours: int=0, days: int=8, date: str='2021-09-08 21:07:09', minutes: int=0, years: int=0, tz: str='UTC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will take in a date number of years, months, days,
		hours, minutes, and seconds as query parameters and return the date
		with the addtion or subtraction produced from entered query
		parameters.
		
		This endpoint can do addition and subtraction. To do subtraction just add
		`-` befor the integer like so `-8` and the endpoint will subtract based
		on the query.
		
		**Note**: Please enter properly formatted dates and optionally times.
		This endpoint will try and figure out what is entered but will output
		incorrect dates and times if date format isn't well formatted."
    seconds: number of seconds.
        months: Number of months
        hours: Number of hours.
        days: Number of days.
        date: Date for arithmetic calculation.
        minutes: Number of minutes.
        years: Number of years.
        tz: Time Zone. Timezone must be in [IANA](https://bit.ly/3h8wd73) format.
        
    """
    url = f"https://daysapi.p.rapidapi.com/calendar/arithmetic"
    querystring = {}
    if seconds:
        querystring['seconds'] = seconds
    if months:
        querystring['months'] = months
    if hours:
        querystring['hours'] = hours
    if days:
        querystring['days'] = days
    if date:
        querystring['date'] = date
    if minutes:
        querystring['minutes'] = minutes
    if years:
        querystring['years'] = years
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daysapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def is_weekday(date: str='2021-09-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint will return True or False of date entered is a weekday.
		
		If date is a Saturday or Sunday, then `false` will be returned."
    date: Date to check for weekday.
        
    """
    url = f"https://daysapi.p.rapidapi.com/calendar/is_weekday"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daysapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def business_delta(second_date: str='2021-09-16', first_date: str='2021-09-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given two dates. This endpoint will output the number of business
		days between them.
		
		Dates can be entered in any order. Please enter readable dates.
		Doesn't have to be ISO or RFC formatted dates."
    second_date: Second date of dates between.
        first_date: First date of dates between.
        
    """
    url = f"https://daysapi.p.rapidapi.com/business/delta"
    querystring = {}
    if second_date:
        querystring['second_date'] = second_date
    if first_date:
        querystring['first_date'] = first_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daysapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def business_days(date: str='2021-09-08', days: int=8, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate working days from given date with given number of days.
		
		The timezone is set to US/Eastern due to US banks operate only in
		that timezone."
    date: Enter date to add or subtract business days from.You can enter any readable date. Doesn't have to be ISO or RFC formatted.
        days: Number of business days. Default is 8 business days.
        
    """
    url = f"https://daysapi.p.rapidapi.com/business/days"
    querystring = {}
    if date:
        querystring['date'] = date
    if days:
        querystring['days'] = days
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daysapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def is_weekend(date: str='2021-09-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint will return True if date falls on the weekend, Saturday or
		Sunday."
    date: Checks if date given is a weekend.
        
    """
    url = f"https://daysapi.p.rapidapi.com/calendar/is_weekend"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daysapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def difference_calendar(date_one: str='2021-09-08', date_two: str='2021-11-08', tz_2: str='UTC', tz_1: str='UTC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes in two dates and calculates the difference for
		you with the queries you enter."
    date_one: First date to get difference
        date_two: Second date to get calendar difference
        tz_2: Please entered prefered timzone. Use `IANA` format.
        tz_1: Please entered prefered timzone. Use `IANA` format.
        
    """
    url = f"https://daysapi.p.rapidapi.com/calendar/difference"
    querystring = {}
    if date_one:
        querystring['date_one'] = date_one
    if date_two:
        querystring['date_two'] = date_two
    if tz_2:
        querystring['tz_2'] = tz_2
    if tz_1:
        querystring['tz_1'] = tz_1
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "daysapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

