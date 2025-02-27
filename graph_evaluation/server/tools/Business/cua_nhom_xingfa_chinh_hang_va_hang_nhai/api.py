import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cua_nhom_xingfa(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cửa nhôm Xingfa nhập khẩu chính hãng bao gồm cửa nhôm Xingfa xếp trượt, mở quay. Các mẫu cửa nhôm Xingfa gồm 4 cánh, 3 cánh, 2 cánh, 1 cánh. Màu cửa nhôm kính Xingfa gồm: vân gỗ, xám, trắng. Báo giá cửa nhôm Xingfa nhập khẩu chính hãng xem tại Thủ Đô Group: https://xingfagroup.vn/bao-gia-cua-nhom-xingfa_p374.aspx"
    
    """
    url = f"https://cua-nhom-xingfa-chinh-hang-va-hang-nhai.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cua-nhom-xingfa-chinh-hang-va-hang-nhai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

