import argparse
import fnmatch
import functools
import json
import sys
import tarfile
import zipfile
from pathlib import Path
from shutil import rmtree
from typing import Iterable
from unittest.mock import patch
from xml.etree import ElementTree

from multithreaded_download import downloader
from rich.progress import track

from bluemira.base.look_and_feel import (
    bluemira_print,
    bluemira_print_flush,
    bluemira_warn,
)


def state_download_size(download_size, uncompressed_size, units):
    bluemira_warn(
        f"This script will download up to {download_size} {units} "
        "of data. Extracting and processing the data may require as much "
        f"as {uncompressed_size} {units} of additional free disk space."
    )


def extractor(compressed_files, extraction_dir, del_compressed_file):
    Path.mkdir(extraction_dir, parents=True, exist_ok=True)

    if not isinstance(compressed_files, Iterable):
        compressed_files = [compressed_files]

    for f in compressed_files:
        suffix = "".join(f.suffixes[-2:])
        if suffix.endswith(".zip"):
            with zipfile.ZipFile(f, "r") as zip_handler:
                bluemira_print("Getting file list")
                file_list = zip_handler.infolist()
                for m in track(
                    filter_members(f.parts[-1], {z.filename: z for z in file_list}),
                    description="Extracting",
                ):
                    file = Path(extraction_dir, m.filename)
                    if not file.is_file() or file.stat().st_size != m.file_size:
                        zip_handler.extract(m, path=extraction_dir)

        elif suffix in (".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".xz"):
            with tarfile.open(f, "r") as tgz:
                bluemira_print("Getting file list")
                file_list = tgz.getmembers()
                for m in track(
                    filter_members(
                        f.parts[-1], {m.get_info()["name"]: m for m in file_list}
                    ),
                    description="Extracting",
                ):
                    if isinstance(m, str):
                        tgz.extract(m, path=extraction_dir)
                    else:
                        file = Path(extraction_dir, m.get_info()["name"])
                        if (
                            not file.is_file()
                            or file.stat().st_size != m.get_info()["size"]
                        ):
                            tgz.extract(m, path=extraction_dir)
        else:
            raise ValueError(
                f"File type not currently supported by extraction function {f!s}"
            )

    if del_compressed_file:
        rmtree(compressed_files, ignore_errors=True)


def filter_members(filename, members):
    import openmc_data.convert.convert_tendl as tendl
    from openmc_data import all_release_details

    with open(
        Path(Path(__file__).parent, "openmc_data_download/nuclear_data_isotopes.json")
    ) as fh:
        isotope_data = json.load(fh)
    filtered_members = []
    mem_keys = members.keys()
    if (
        filename
        in all_release_details["tendl"][tendl.args.release]["neutron"][
            "compressed_files"
        ]
    ):
        for m in isotope_data["tendl"]["neutron"]:
            if file := fnmatch.filter(mem_keys, f"*/{m}"):
                filtered_members.append(members[file[0]])
            else:
                bluemira_warn(f"Cant find {m} in {filename}")
    elif (
        filename == all_release_details["endf"]["b7.1"]["neutron"]["compressed_files"][0]
    ):
        for m in isotope_data["endf"]["neutron"]:
            if file := fnmatch.filter(mem_keys, f"*/{m[:-3]}_{m[-3:]}*.ace"):
                filtered_members.append(members[file[0]])
            else:
                bluemira_warn(f"Cant find {m} in {filename}")
    elif (
        filename == all_release_details["endf"]["b7.1"]["neutron"]["compressed_files"][1]
    ):
        filtered_members = ["bebeo.acer", "obeo.acer"]

    elif filename in all_release_details["endf"]["b7.1"]["photon"]["compressed_files"]:
        for m in isotope_data["endf"]["photon"]:
            if file := fnmatch.filter(mem_keys, f"*{m}*.endf"):
                filtered_members.append(members[file[0]])
            else:
                bluemira_warn(f"Cant find {m} in {filename}")
    return filtered_members


def convertion_progress(*string, **kwargs):
    bluemira_print_flush("".join([str(s) for s in string]))


def parse_args(*args):
    parser = argparse.ArgumentParser("Bluemira Neutronics data downloader")

    parser.add_argument("-l", "--location", default=Path.cwd() / "bluemira_openmc_data")
    parser.add_argument("--download_threads", type=int, default=5)
    p, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0], *unknown]
    return p


def main(*args):
    p = parse_args(*args)

    import openmc_data.convert.convert_endf as endf
    import openmc_data.convert.convert_tendl as tendl

    libs = (tendl, endf)
    lib_names = tuple(lib.__name__.split("_")[-1] for lib in libs)
    root_folder = p.location
    download = functools.partial(downloader, max_workers=p.download_threads)
    for name, lib in zip(lib_names, libs):
        lib.state_download_size = state_download_size
        lib.download = download
        lib.extract = extractor
        lib.args.destination = root_folder / name

    for name, lib in zip(lib_names, libs):
        bluemira_print(f"Downloading {name} cross section data")
        with patch(
            "builtins.print",
            new=convertion_progress,
        ):
            lib.main()
        print()

    bluemira_print("Removing uneeded files")
    thermal_prefix = endf.args.destination / "neutron"
    for i in ("c_O_in_BeO.h5", "c_Be_in_BeO.h5"):
        Path(thermal_prefix / i).unlink()

    bluemira_print("Combining cross section xml files")
    xml_files = [root_folder / name / "cross_sections.xml" for name in lib_names]
    xml_handle = [ElementTree.parse(xml) for xml in xml_files]
    for name, xml in zip(lib_names, xml_handle):
        data = xml.getroot()
        remove_list = []
        for elem in data.iter():
            if elem.tag == "library":
                if elem.attrib["materials"] in ("c_O_in_BeO", "c_Be_in_BeO"):
                    remove_list.append(elem)
                    continue
                elem.attrib["path"] = str(Path(name, elem.attrib["path"]))
        for elem in remove_list:
            data.remove(elem)

    data = xml_handle[0].getroot()
    for xml in xml_handle[1:]:
        data.extend(list(xml.getroot().iter())[1:])

    xml_handle[0].write(root_folder / "cross_sections.xml")


main()
