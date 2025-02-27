import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def merge_two_pst_file(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If you want to split a large PST file into smaller smaller PST files. So you can use Mailsdaddy PST Split Software. It is very easy to use this software. Which splits large PST files in just a few clicks. Also, this tool provides some more options whereby date, email address, and folder-wise PST merge software supports ANSI and Unicode PST files and supports all MS Outlook versions as well.
		visit for more information:-https://www.mailsdaddy.com/pst-merge-and-join/
		Manual Method:-https://www.mailsdaddy.com/blogs/manual-trick-combine-multiple-pst-files-single-pst-file/"
    
    """
    url = f"https://merge-two-pst-file.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "merge-two-pst-file.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

