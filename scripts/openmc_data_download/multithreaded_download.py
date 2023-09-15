import asyncio
import concurrent.futures
import functools
import hashlib
import os
import signal
import socket
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

import requests
from rich.progress import Progress


async def get_size(url):
    response = requests.head(url)
    if int(response.headers["Content-Length"]) == 0:
        with urlopen(url) as response:
            return int(response.length)
    return int(response.headers["Content-Length"])


def download_range(
    url,
    start,
    end,
    output,
    progress,
    task_prog,
    timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
    cafile=None,
    capath=None,
    cadefault=False,
    context=None,
):
    page = Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"}
    )

    if output.is_file() and output.stat().st_size == end + 1 - start:
        progress.update(task_prog, advance=1)
        return

    with urlopen(
        page,
        timeout=timeout,
        cafile=cafile,
        capath=capath,
        cadefault=cadefault,
        context=context,
    ) as response, open(output, "wb") as f:
        while chunk := response.read(10000):
            f.write(chunk)
    progress.update(task_prog, advance=1)


async def download(run, url, output, chunk_size=10000000, **kwargs):
    file_size = await get_size(url)

    if output.is_file() and output.stat().st_size == file_size:
        return

    chunks = range(0, file_size, chunk_size)

    with Progress() as progress:
        task_prog = progress.add_task(
            f"Downloading {output.parts[-1]}", total=len(chunks)
        )

        tasks = [
            run(
                download_range,
                url,
                start,
                start + chunk_size - 1,
                Path(f"{output}.part{i}"),
                progress,
                task_prog,
                *kwargs.values(),
            )
            for i, start in enumerate(chunks)
        ]

        await asyncio.wait(tasks)

        with open(output, "wb") as o:
            for i in range(len(chunks)):
                chunk_path = Path(f"{output}.part{i}")
                with open(chunk_path, "rb") as s:
                    o.write(s.read())
        for i in range(len(chunks)):
            chunk_path = Path(f"{output}.part{i}").unlink()


def downloader(
    url: str,
    checksum: Optional[int] = None,
    as_browser: bool = False,
    output_path: Optional[os.PathLike] = None,
    output_filename: Optional[os.PathLike] = None,
    *,
    max_workers: int = 5,
    timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
    cafile=None,
    capath=None,
    cadefault=False,
    context=None,
):
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    loop = asyncio.new_event_loop()
    run = functools.partial(loop.run_in_executor, executor)

    asyncio.set_event_loop(loop)

    if output_filename is None:
        output_filename = Path(url.rsplit("/", 1)[-1])

    if output_path is None:
        local_path = Path(output_filename)
    else:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        local_path = Path(output_path) / output_filename

    try:
        loop.run_until_complete(
            download(
                run,
                url,
                local_path,
                timeout=timeout,
                cafile=cafile,
                capath=capath,
                cadefault=cadefault,
                context=context,
            )
        )
    finally:
        loop.close()

    if (
        checksum is not None
        and hashlib.md5(open(local_path, "rb").read()).hexdigest() != checksum
    ):
        raise OSError(
            f"MD5 checksum for {local_path} does not match."
            "Please ensure you checksum is correct."
        )
