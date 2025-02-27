import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lucky88wiki_website_ch_nh_ch_c_a_nh_c_i_lucky88(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lucky88.wiki - Lucky88 ⚡️ Link vào Lucky88 chính thức, cá độ bóng đá online, cá cược thể thao trực tuyến Game cá cược đa dạng từ Bóng Đá, Đá gà, Casino, Xổ số, Nổ hủ, Slot...cùng với nhiều Khuyến mãi hấp dẫn Đảm bảo An Toàn - Uy Tín - Hợp Pháp Số 1
		 #lucky88 #lucky88fan #lucky88dotfan #lucky88chamfan
		#đăng ký lucky88 #nạp tiền lucky88 #rút tiền lucky88 #tải ứng dụng lucky88  /m/033_l8  /m/033_l8 #Casino #Lucky88wiki
		
		Địa chỉ : 420 Trần Khát Chân, Thanh Nhàn, Hoàn Kiếm, Hà Nội, Việt Nam
		Hotline 0942355551
		Gmail: lucky88wiki@gmail.com
		Website https://lucky88.wiki/ là địa chỉ chính thức của Lucky88
		https://lucky88.wiki/dang-ky-lucky88/ 
		https://lucky88.wiki/nap-tien-lucky88/ 
		https://lucky88.wiki/rut-tien-lucky88/ 
		https://lucky88.wiki/tai-ung-dung-lucky88/ 
		https://sites.google.com/view/lucky88wiki 
		https://en.wikipedia.org/wiki/Online_casino 
		https://lucky88wiki.blogspot.com/2022/04/lucky88wiki.html
		https://gallery.autodesk.com/users/H7DJCV3JJ5VR3P7P?relProf=1
		https://sketchfab.com/lucky88wiki
		https://dev.to/lucky88wiki
		https://visual.ly/users/lucky88wiki/portfolio
		https://kuula.co/profile/lucky88wi
		https://lucky88wiki.cgsociety.org/profile
		https://cults3d.com/en/users/lucky88wiki
		https://www.trepup.com/lucky88wiki-555656555491025
		https://app.bountysource.com/people/100379-lucky88wiki
		https://www.goodreads.com/user/show/149856811-lucky88wiki
		https://www.metal-archives.com/users/lucky88wiki
		https://gifyu.com/lucky88wiki
		https://os.mbed.com/users/lucky88wiki/
		http://photozou.jp/user/top/3312140
		https://gfycat.com/@lucky88wiki
		https://piqs.de/user/lucky88wiki
		https://git.project-hobbit.eu/-/snippets/5895
		https://notionpress.com/author/499787
		https://www.longisland.com/profile/lucky88wiki
		https://www.noteflight.com/profile/101f0a22974c"
    
    """
    url = f"https://lucky88wiki.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lucky88wiki.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

