# -*- coding: utf-8 -*-
"""校验本机已有 Fashion-MNIST，在项目 data 内创建四个只读用途的硬链接。

用法：python tools/prepare_fashion_mnist.py --source <已有 data 目录>
不下载、不安装；不改写源文件。目标已有时只校验，不覆盖。
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from torchvision.datasets import FashionMNIST

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data/FashionMNIST/raw"


def digest_file(path, opener=open, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(source):
    source_raw = Path(source).resolve() / "FashionMNIST/raw"
    checks = []
    for filename, expected in FashionMNIST.resources:
        archive = source_raw / filename
        raw = source_raw / filename[:-3]
        if digest_file(archive, algorithm="md5") != expected:
            raise ValueError("归档 MD5 不匹配: " + filename)
        raw_hash = digest_file(raw)
        if raw_hash != digest_file(archive, opener=gzip.open):
            raise ValueError("解压文件与归档内容不匹配: " + filename)
        checks.append({"filename": raw.name, "archive_md5": expected,
                       "raw_sha256": raw_hash, "raw_bytes": raw.stat().st_size})
    TARGET.mkdir(parents=True, exist_ok=True)
    for check in checks:
        source_path = source_raw / check["filename"]
        target_path = TARGET / check["filename"]
        if not target_path.exists():
            os.link(source_path, target_path)
        if digest_file(target_path) != check["raw_sha256"]:
            raise ValueError("项目内已有目标内容不同，不覆盖: " + target_path.name)
        check["same_file"] = os.path.samefile(source_path, target_path)
        check["hard_link_count"] = target_path.stat().st_nlink
    metadata = {"source_root": str(Path(source).resolve()),
                "strategy": "hard links to verified existing raw files; no data download",
                "files": checks, "logical_raw_bytes": sum(c["raw_bytes"] for c in checks)}
    with (ROOT / "data/fashion_mnist_source.local.json").open(
            "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    public = {k: v for k, v in metadata.items() if k != "source_root"}
    public["dataset_origin"] = "https://github.com/zalandoresearch/fashion-mnist"
    with (ROOT / "reports/ch03/3_5_data_integrity.json").open(
            "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(public, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(public, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    prepare(parser.parse_args().source)
