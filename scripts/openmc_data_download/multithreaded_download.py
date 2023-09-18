import asyncio
import concurrent.futures
import functools
import hashlib
import os
import signal
from pathlib import Path
from typing import Callable, Optional

import requests
from rich.progress import Progress


async def get_size(url: str, timeout: int = 10) -> int:
    """Get size of file"""
    response = requests.head(  # noqa: ASYNC100
        url, allow_redirects=True, timeout=timeout
    )
    return int(response.headers["Content-Length"])


def download_range(
    url: str,
    start: int,
    end: int,
    output: Path,
    progress: Callable[[], None],
    timeout: int = 10,
):
    """Download worker"""
    if not (output.is_file() and output.stat().st_size == end + 1 - start):
        headers = {"User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"}
        response = requests.get(url, headers=headers, timeout=timeout)
        with open(output, "wb") as f:
            for part in response.iter_content(1024):
                f.write(part)
    progress()


async def _download(
    run: Callable,
    url: str,
    output: Path,
    *,
    chunk_size: int = 10000000,
    timeout: int = 10,
):
    """Downloader event loop"""
    if not url.startswith(("http:", "https:")):
        raise ValueError("Must be an http or https url")

    file_size = await get_size(url)

    if output.is_file() and output.stat().st_size == file_size:
        return

    chunks = range(0, file_size, chunk_size)
    with Progress() as progress:
        updater = functools.partial(
            progress.update,
            progress.add_task(f"Downloading {output.parts[-1]}", total=len(chunks)),
            advance=1,
        )
        tasks = [
            run(
                download_range,
                url,
                start,
                start + chunk_size - 1,
                Path(f"{output}.part{i}"),
                updater,
                timeout,
            )
            for i, start in enumerate(chunks)
        ]

        await asyncio.wait(tasks)

    with open(output, "wb") as o:  # noqa: ASYNC101
        for i in range(len(chunks)):
            chunk_path = Path(f"{output}.part{i}")
            with open(chunk_path, "rb") as s:  # noqa: ASYNC101
                o.write(s.read())

    # Write all before deleting
    for i in range(len(chunks)):
        chunk_path = Path(f"{output}.part{i}").unlink()


def downloader(
    url: str,
    checksum: Optional[int] = None,
    as_browser: bool = False,  # noqa: ARG001
    output_path: Optional[os.PathLike] = None,
    output_filename: Optional[os.PathLike] = None,
    *,
    max_workers: int = 5,
    timeout: int = 10,
):
    """Asynchronous multithreaded downloader"""
    if output_filename is None:
        output_filename = Path(url.rsplit("/", 1)[-1])

    if output_path is None:
        local_path = Path(output_filename)
    else:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        local_path = Path(output_path, output_filename)

    # Reenable Keyboard Interrupt
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    loop = asyncio.new_event_loop()
    run = functools.partial(loop.run_in_executor, executor)
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(_download(run, url, local_path, timeout=timeout))
    finally:
        loop.close()

    if (
        checksum is not None
        and hashlib.md5(open(local_path, "rb").read()).hexdigest()  # noqa: S324, SIM115
        != checksum
    ):
        raise OSError(
            f"MD5 checksum for {local_path} does not match."
            "Please ensure you checksum is correct."
        )
