import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def year_fractions(start_date: str, end_date: str, dcc_types: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Computes and returns the year fraction for a period time in the specified day count convention."
    start_date: The start date of the period time in YYYY-MM-DD format
        end_date: The end date of the period time in YYYY-MM-DD format
        dcc_types: The day count convention
        
    """
    url = f"https://date-calculator2.p.rapidapi.com/datetime/dcc/year_fractions"
    querystring = {'start_date': start_date, 'end_date': end_date, 'dcc_types': dcc_types, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "date-calculator2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_zone_converter(datetime: str, from_tzname: str='UTC', to_tzname: str='UTC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts time from time zone to another taking into account Daylight Saving Time (DST) and accepts present, past, or future dates."
    datetime: The date time in  [ISO 8601 format](https://www.w3.org/TR/NOTE-datetime)
        from_tzname: The time zone name
        to_tzname: The time zone name
        
    """
    url = f"https://date-calculator2.p.rapidapi.com/datetime/timezone/conversion"
    querystring = {'datetime': datetime, }
    if from_tzname:
        querystring['from_tzname'] = from_tzname
    if to_tzname:
        querystring['to_tzname'] = to_tzname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "date-calculator2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datedif(end_date: str, start_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Computes the number of  years, months, days, hours, minutes, seconds or microseconds between two dates (end_date - start_date)."
    end_date: The start date w/ or w/o the time part in  [ISO 8601 format](https://www.w3.org/TR/NOTE-datetime)
        start_date: The start date w/ or w/o the time part in  [ISO 8601 format](https://www.w3.org/TR/NOTE-datetime)
        
    """
    url = f"https://date-calculator2.p.rapidapi.com/datetime/datedif"
    querystring = {'end_date': end_date, 'start_date': start_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "date-calculator2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eomonth(months: int, start_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Computes and returns a date on the last day of the month in the future or past. This resource behave exactly as Excel EOMONTH function."
    months: The number of months before or after start_date. A positive value for months yields a future date; a negative value yields a past date.
        start_date: The start date w/ or w/o the time part in  [ISO 8601 format](https://www.w3.org/TR/NOTE-datetime)
        
    """
    url = f"https://date-calculator2.p.rapidapi.com/datetime/eomonth"
    querystring = {'months': months, 'start_date': start_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "date-calculator2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def now(tzname: str='UTC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Computes and returns local time zone taking into account Daylight Saving Time (DST)."
    tzname: The time zone name
        
    """
    url = f"https://date-calculator2.p.rapidapi.com/datetime/timezone/now"
    querystring = {}
    if tzname:
        querystring['tzname'] = tzname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "date-calculator2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rdates(start_date: str, rrules: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Computes and returns recurring dates based on very flexible recurrence rules as defined and specified in the [iCalendar RFC 5545](https://tools.ietf.org/html/rfc5545 "ICalendar"). It's possible to specify more than one recurrence rule."
    start_date: The start date w/ or w/o the time part in  [ISO 8601 format](https://www.w3.org/TR/NOTE-datetime)
        rrules: The recurrence rules list as string seperated by space; Each rule should be prefixed by **RRULE:**
        
    """
    url = f"https://date-calculator2.p.rapidapi.com/datetime/rdates"
    querystring = {'start_date': start_date, 'rrules': rrules, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "date-calculator2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sdate(start_date: str, weekday: str='MO', hours: int=14, microseconds: int=0, seconds: int=0, minutes: int=0, days: int=0, months: int=-1, weeks: int=1, years: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Shifts  a date or a datetime in the future or past."
    start_date: The start date w/ or w/o the time part in  [ISO 8601 format](https://www.w3.org/TR/NOTE-datetime)
        weekday: Allows to shift a date to the specified next or previous Nth weekday. It should be one of the weekday instances (SU, MO, TU, WE, TH, FR, SA). These instances may receive a parameter N, specifying the Nth weekday in parentheses, which could be positive or negative (like MO(+1) or MO(-2)). Not specifying it is the same as specifying +1. If the calculated date is already Monday, using MO(1) or MO(-1) won't change the day.
        hours: The number of hours before(negative value) or after(positive value) start_date.
        microseconds: The number of microseconds before(negative value) or after(positive value) start_date.
        seconds: The number of seconds before(negative value) or after(positive value) start_date.
        minutes: The number of minutes before(negative value) or after(positive value) start_date.
        days: The number of days before(negative value) or after(positive value) start_date.
        months: The number of months before(negative value) or after(positive value) start_date.
        weeks: The number of weeks before(negative value) or after(positive value) start_date.
        years: The number of years before(negative value) or after(positive value) start_date.
        
    """
    url = f"https://date-calculator2.p.rapidapi.com/datetime/sdate"
    querystring = {'start_date': start_date, }
    if weekday:
        querystring['weekday'] = weekday
    if hours:
        querystring['hours'] = hours
    if microseconds:
        querystring['microseconds'] = microseconds
    if seconds:
        querystring['seconds'] = seconds
    if minutes:
        querystring['minutes'] = minutes
    if days:
        querystring['days'] = days
    if months:
        querystring['months'] = months
    if weeks:
        querystring['weeks'] = weeks
    if years:
        querystring['years'] = years
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "date-calculator2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def edate(months: int, start_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Computes and returns a date on the same day of the month in the future or past. This resource behave exactly as Excel EDATE function."
    months: The number of months before or after start_date. A positive value for months yields a future date; a negative value yields a past date.
        start_date: The start date w/ or w/o the time part in  [ISO 8601 format](https://www.w3.org/TR/NOTE-datetime)
        
    """
    url = f"https://date-calculator2.p.rapidapi.com/datetime/edate"
    querystring = {'months': months, 'start_date': start_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "date-calculator2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

