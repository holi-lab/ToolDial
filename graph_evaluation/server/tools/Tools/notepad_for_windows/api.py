import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def notepad_for_windows(url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Notepad is a simple text editor that is included with the Windows operating system. It allows users to create, open and save text files. The files can be saved in plain text format with the .txt file extension. Notepad can be used for a wide range of tasks, including creating documents, writing scripts, taking notes, and editing configuration files.
		
		Some of the key features of Notepad include:
		
		Simplicity: Notepad has a minimalistic interface, making it easy to use for users of all experience levels.
		Plain text format: Notepad saves files in plain text format, which means that the files are not formatted with any specific font, color, or other formatting options, And the files are often used for writing code.
		Line numbering: Notepad allows to number the lines of the text.
		Word wrap: Notepad allows to wrap the text to the next line if it exceeds the width of the Notepad window.
		Search and Replace: Notepad allows to search for specific text within a document and replace it with new text.
		File Management: Notepad provides basic file management functionality, such as the"
    
    """
    url = f"https://notepad-for-windows.p.rapidapi.com/"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "notepad-for-windows.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

