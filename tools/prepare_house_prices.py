# -*- coding: utf-8 -*-
"""校验房价CSV缓存；只有显式--download时获取两份教材镜像文件。"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/house_prices'
FILES={
    'kaggle_house_pred_train.csv':'585e9cc93e70b39160e7921475f9bcd7d31219ce',
    'kaggle_house_pred_test.csv':'fa19780a7b011d9b009e8bff8e99922a8ee2eb90',
}


def prepare(download=False):
    DATA.mkdir(parents=True,exist_ok=True);records=[]
    for name,expected in FILES.items():
        path=DATA/name
        if path.is_file():
            if hashlib.sha1(path.read_bytes()).hexdigest()!=expected:
                raise ValueError('已有缓存哈希不符，保留原文件，请检查：'+str(path))
        elif not download:
            raise FileNotFoundError('缺少本地比赛CSV。请使用--download或准备已校验数据：'+str(path))
        else:
            url='https://d2l-data.s3-accelerate.amazonaws.com/'+name
            partial=DATA/(name+'.part')
            if partial.exists():raise FileExistsError('保留已有临时文件：'+str(partial))
            try:
                with urlopen(url,timeout=30) as response,partial.open('xb') as handle:
                    total=0
                    while True:
                        block=response.read(65536)
                        if not block:break
                        total+=len(block)
                        if total>2_000_000:raise ValueError('文件超出预期大小')
                        handle.write(block)
                if hashlib.sha1(partial.read_bytes()).hexdigest()!=expected:
                    raise ValueError('下载内容与教材SHA-1不一致')
                partial.rename(path)
            finally:
                if partial.is_file():partial.unlink()
        raw=path.read_bytes()
        records.append({'filename':name,'bytes':len(raw),'sha1':hashlib.sha1(raw).hexdigest(),
                        'sha256':hashlib.sha256(raw).hexdigest(),'url':'https://d2l-data.s3-accelerate.amazonaws.com/'+name})
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true',help='显式获取两份CSV；不安装依赖')
    args=parser.parse_args();print(json.dumps(prepare(args.download),ensure_ascii=False,indent=2))
