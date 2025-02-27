import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def performance_monitor_metrics(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "- Retrieve real-time CPU usage in percentage (%).
		- Monitor memory utilization and get the percentage of memory in use.
		- Track network latency to assess network performance.
		- Access system bottlenecks information to identify performance constraints.
		- Easy integration into your application or system using the RESTful API.
		- Accelerate troubleshooting and performance optimization of TI systems."
    
    """
    url = f"https://perf-monitor-metrics.p.rapidapi.com/api/system/performance"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "perf-monitor-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

