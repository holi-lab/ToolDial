import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tdtctop(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "https://tdtc.top/   Trang tải game chính thức của nhà cái TDTC. Chuyên cung cấp những sản phẩm game đổi thưởng, cá cược online hàng đầu ngày nay. Cụ thể như: phá băng, bầu cua, avengers,...khuyến mãi mới nhất từ TDTC club. Tải game nhận khuyến mãi hấp dẫn
		#nhà_cái_TDTC #Game_bầu_cua # tdtctop #Bắn_cá #Xổ_số #đăng_ký_TDTC #đăng_nhập_TDTC #link_vào_TDTC
		Email: tdtctop@gmail.com
		Hotline 0359.26.8181
		Địa chỉ 364 Đ. Ng. Gia Tự, Đức Giang, Long Biên, Hà Nội, Việt Nam
		Xem thêm 
		https://sites.google.com/view/tdtctop 
		https://tdtc.top/huong-dan-tdtc/ 
		https://tdtc.top/kinh-nghiem-tdtc/ 
		https://tdtc.top/news/ 
		https://tdtctop.blogspot.com/ 
		https://d.cosx.org/u/tdtctop
		http://tdtctop.bravesites.com/
		http://micro.blog/tdtctop
		https://community.dynamics.com/members/tdtctop
		https://www.fontshop.com/people/tdtc-top
		https://www.lifeofpix.com/photographers/tdtctop/
		https://teletype.in/@tdtctop
		https://www.intensedebate.com/people/tdtctop1
		https://community.virginmedia.com/t5/user/viewprofilepage/user-id/849467
		https://seedandspark.com/user/tdtc-top
		https://godotengine.org/qa/user/tdtctop
		https://cults3d.com/en/users/tdtctop
		https://coub.com/tdtctop53118
		https://www.emoneyspace.com/tdtctop
		https://www.giantbomb.com/profile/tdtctop/
		http://tdtctop.jigsy.com/
		https://www.wpgmaps.com/forums/users/tdtctop/
		https://gifyu.com/tdtctop"
    
    """
    url = f"https://tdtctop.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tdtctop.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

