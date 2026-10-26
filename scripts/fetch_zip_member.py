#!/usr/bin/env python3
"""Fetch one member of a remote ZIP using HTTP byte ranges (including ZIP64)."""
from __future__ import annotations

import argparse
from pathlib import Path
import struct
from urllib.request import Request, urlopen
import zlib


def request_range(url: str, start: int, end: int) -> bytes:
    request = Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "SignVision/1.0"})
    with urlopen(request, timeout=90) as response:
        return response.read()


def content_length(url: str) -> int:
    request = Request(url, method="HEAD", headers={"User-Agent": "SignVision/1.0"})
    with urlopen(request, timeout=90) as response:
        return int(response.headers["Content-Length"])


def zip64_values(extra: bytes, need_uncompressed: bool, need_compressed: bool, need_offset: bool):
    cursor = 0
    while cursor + 4 <= len(extra):
        kind, size = struct.unpack_from("<HH", extra, cursor)
        payload = extra[cursor + 4: cursor + 4 + size]
        if kind == 0x0001:
            pos = 0; values = {}
            if need_uncompressed: values["uncompressed"] = struct.unpack_from("<Q", payload, pos)[0]; pos += 8
            if need_compressed: values["compressed"] = struct.unpack_from("<Q", payload, pos)[0]; pos += 8
            if need_offset: values["offset"] = struct.unpack_from("<Q", payload, pos)[0]
            return values
        cursor += 4 + size
    return {}


def fetch_member(url: str, member: str) -> bytes:
    total = content_length(url)
    tail_start = max(0, total - 4 * 1024 * 1024)
    tail = request_range(url, tail_start, total - 1)
    eocd_at = tail.rfind(b"PK\x05\x06")
    if eocd_at < 0: raise RuntimeError("ZIP EOCD не найден")
    eocd = tail[eocd_at:eocd_at + 22]
    _, _, _, _, _, central_size32, central_offset32, _ = struct.unpack("<4s4H2LH", eocd)
    central_size, central_offset = central_size32, central_offset32
    if central_size32 == 0xFFFFFFFF or central_offset32 == 0xFFFFFFFF:
        locator_at = tail.rfind(b"PK\x06\x07", 0, eocd_at)
        if locator_at < 0: raise RuntimeError("ZIP64 locator не найден")
        zip64_offset = struct.unpack_from("<Q", tail, locator_at + 8)[0]
        zip64 = request_range(url, zip64_offset, zip64_offset + 55)
        if not zip64.startswith(b"PK\x06\x06"): raise RuntimeError("ZIP64 EOCD повреждён")
        central_size = struct.unpack_from("<Q", zip64, 40)[0]
        central_offset = struct.unpack_from("<Q", zip64, 48)[0]
    central = request_range(url, central_offset, central_offset + central_size - 1)
    cursor = 0; target = None
    while cursor + 46 <= len(central) and central[cursor:cursor+4] == b"PK\x01\x02":
        method = struct.unpack_from("<H", central, cursor + 10)[0]
        compressed32 = struct.unpack_from("<L", central, cursor + 20)[0]
        uncompressed32 = struct.unpack_from("<L", central, cursor + 24)[0]
        name_len, extra_len, comment_len = struct.unpack_from("<HHH", central, cursor + 28)
        offset32 = struct.unpack_from("<L", central, cursor + 42)[0]
        name_start = cursor + 46
        name = central[name_start:name_start+name_len].decode("utf-8")
        extra = central[name_start+name_len:name_start+name_len+extra_len]
        values = zip64_values(extra, uncompressed32 == 0xFFFFFFFF, compressed32 == 0xFFFFFFFF, offset32 == 0xFFFFFFFF)
        compressed = values.get("compressed", compressed32)
        uncompressed = values.get("uncompressed", uncompressed32)
        offset = values.get("offset", offset32)
        if name == member or name.endswith("/" + member):
            target = (method, compressed, uncompressed, offset); break
        cursor += 46 + name_len + extra_len + comment_len
    if not target: raise FileNotFoundError(f"{member} отсутствует в архиве")
    method, compressed, uncompressed, offset = target
    header = request_range(url, offset, offset + 29)
    if not header.startswith(b"PK\x03\x04"): raise RuntimeError("Локальный ZIP-заголовок повреждён")
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    data_offset = offset + 30 + name_len + extra_len
    packed = request_range(url, data_offset, data_offset + compressed - 1)
    if method == 0: data = packed
    elif method == 8: data = zlib.decompress(packed, -15)
    else: raise RuntimeError(f"Неподдерживаемый ZIP compression method {method}")
    if len(data) != uncompressed: raise RuntimeError("Размер распакованного файла не совпадает")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url"); parser.add_argument("member"); parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = fetch_member(args.url, args.member)
    args.output.write_bytes(data)
    print(f"Сохранено: {args.output} ({len(data):,} байт)")


if __name__ == "__main__": main()

