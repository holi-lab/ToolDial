import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reverse(x_api_key: str='eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImp0aSI6ImM5MTU3ZGM5NTlkNGIyNGZiYTE2NWVhZDMxMTllYWQwMWIyNDNjYmNjNGY3ZDg2MTYzZGZjMmE4OGJiYmM4MmE1YmFkOWYyZjRkMjJmYTIwIn0.eyJhdWQiOiI3NTMwIiwianRpIjoiYzkxNTdkYzk1OWQ0YjI0ZmJhMTY1ZWFkMzExOWVhZDAxYjI0M2NiY2M0ZjdkODYxNjNkZmMyYTg4YmJiYzgyYTViYWQ5ZjJmNGQyMmZhMjAiLCJpYXQiOjE1NzkwMDYzODcsIm5iZiI6MTU3OTAwNjM4NywiZXhwIjoxNTgxNTExOTg3LCJzdWIiOiIiLCJzY29wZXMiOlsiYmFzaWMiXX0.k9U7-8rT-qVxPKXOABRBdN99W6ejTelUyYrPwpnSeQ72S8LoFffFWBueiwwSyxpbAj4Fb-FjHomaORA_nb0_HaIPCD7Bz6ZgacE5cSXP-10z7rgDFSu-J3q7tNXD6vGcY6s18hMfBh4l1xrPnjmtwhAhN_e5ILTxwJbRBtBVXmQ6OZKmf1gJmF0rF-OQ1kCt8CmnvRgQLGtOanbQbM0mCoCbfh81zBcJQ7l6s-PLhTFuXhepm6cyBmpGOAu1YLeUTX4yXVUOaUtJpGE6rNR1WeGGCm91eXv_nEd_N5VAq90Mv235DSvrx_I37VrNb3G9xn1SO1STk3O6j0AEitjCWQ', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "reverse geocode"
    
    """
    url = f"https://reverse-geocode.p.rapidapi.com/reverse?lat=35.732474329636865&lon=51.42287135124207"
    querystring = {}
    if x_api_key:
        querystring['x-api-key'] = x_api_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reverse-geocode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

