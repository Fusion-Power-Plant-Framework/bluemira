"""Neutronics data downloader main script"""

import argparse
import fnmatch
import functools
import json
import sys
import tarfile
import zipfile
from collections.abc import Callable, Iterable
from os import chdir
from pathlib import Path
from shutil import rmtree
from types import ModuleType
from unittest.mock import patch
from xml.etree import cElementTree  # noqa: S405

from multithreaded_download import downloader
from rich.progress import track

from bluemira.base.look_and_feel import (
    bluemira_print,
    bluemira_print_flush,
    bluemira_warn,
)


def state_download_size(download_size: int, uncompressed_size: int, units: str):
    """Patch warning to logging"""
    bluemira_warn(
        f"This script will download up to {download_size} {units} "
        "of data. Extracting and processing the data may require as much "
        f"as {uncompressed_size:.2f} {units} of additional free disk space."
    )


def extractor(
    compressed_files: list[Path],
    extraction_dir: Path,
    del_compressed_file: bool,  # noqa: FBT001
):
    """Customisable extractor

    Raises
    ------
    ValueError
        Unsupported file type
    """
    Path.mkdir(extraction_dir, parents=True, exist_ok=True)

    if not isinstance(compressed_files, Iterable):
        compressed_files = [compressed_files]

    for f in compressed_files:
        suffix = "".join(f.suffixes[-2:])
        if suffix.endswith(".zip"):
            with zipfile.ZipFile(f, "r") as zip_handler:
                bluemira_print("Getting file list")
                file_list = {z.filename: z for z in zip_handler.infolist()}
                for m in track(
                    filter_members(f.parts[-1], file_list), description="Extracting"
                ):
                    file = Path(extraction_dir, m.filename)
                    if not file.is_file() or file.stat().st_size != m.file_size:
                        zip_handler.extract(m, path=extraction_dir)

        elif suffix in {".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".xz"}:
            with tarfile.open(f, "r") as tgz:
                bluemira_print("Getting file list")
                file_list = {m.get_info()["name"]: m for m in tgz.getmembers()}
                for m in track(
                    filter_members(f.parts[-1], file_list), description="Extracting"
                ):
                    m_inf = m.get_info()
                    file = Path(extraction_dir, m_inf["name"])
                    if not file.is_file() or file.stat().st_size != m_inf["size"]:
                        tgz.extract(m, path=extraction_dir)
        else:
            raise ValueError(
                f"File type not currently supported by extraction function {f!s}"
            )

    if del_compressed_file:
        rmtree(compressed_files, ignore_errors=True)


def _filter_members(
    file: str,
    isotope_file: str,
    filename: str,
    members: dict[str, tarfile.TarInfo | zipfile.ZipInfo],
) -> list[tarfile.TarInfo] | list[zipfile.ZipInfo]:
    """Filter archive contents to only extract wanted files

    Returns
    -------
    :
        A filtered list of filepaths based on user selection
        for a given data library

    Raises
    ------
    ValueError
        If archive filename not known
    """
    import openmc_data.convert.convert_tendl as tendl  # noqa: PLC0415
    from openmc_data import all_release_details as ard  # noqa: PLC0415

    with open(Path(Path(file).parent, isotope_file)) as fh:
        isotope_data = json.load(fh)
    if (
        filename
        in ard["tendl"][tendl.args.release]["neutron"]["ace"]["compressed_files"]
    ):
        return _filter(
            filename, members, isotope_data["tendl"]["neutron"], lambda m: f"*/{m}"
        )
    if filename == ard["endf"]["b7.1"]["neutron"]["ace"]["compressed_files"][0]:
        return _filter(
            filename,
            members,
            isotope_data["endf"]["neutron"],
            lambda m: f"*/{m[:-3]}_{m[-3:]}*.ace",
        )
    if filename == ard["endf"]["b7.1"]["neutron"]["ace"]["compressed_files"][1]:
        return _filter(filename, members, ["bebeo", "obeo"], lambda m: f"{m}.acer")
    if filename in ard["endf"]["b7.1"]["photon"]["endf"]["compressed_files"]:
        return _filter(
            filename, members, isotope_data["endf"]["photon"], lambda m: f"*{m}*.endf"
        )
    raise ValueError("Unknown archive")


def _filter(
    filename: str,
    members: dict[str, tarfile.TarInfo | zipfile.ZipInfo],
    datakeys: list[str],
    filt: Callable,
) -> list[tarfile.TarInfo] | list[zipfile.ZipInfo]:
    """Filter archive members

    Returns
    -------
    :
        A filtered list of filepaths based on user selection
    """
    filtered_members = []
    mem_keys = members.keys()
    if datakeys == "*":
        return list(members.values())
    for m in datakeys:
        if file := fnmatch.filter(mem_keys, filt(m)):
            filtered_members.append(members[file[0]])
        else:
            bluemira_warn(f"Cant find {m} in {filename}")
    return filtered_members


def _convertion_progress(*string, **_kwargs):
    """Patch print function for logging"""
    bluemira_print_flush("".join([str(s) for s in string]))


def combine_xml(
    lib_names: tuple[str, ...], thermal_files: tuple[str, ...], thermal_prefix: Path
):
    """Combine cross_section.xml files from all libraries.

    The cross_section.xml file is parsed in a way that, if there are duplicate isotopes,
    the first appearance of this isotope will be used when extracting its cross-sections.
    This behaviour is enforced by :meth:`openmc.data.library.DataLibrary.get_by_material`
    where the first xml record with matching 'materials' string is returned.
    """
    bluemira_print("Removing uneeded files")
    for i in thermal_files:
        Path(thermal_prefix / f"{i}.h5").unlink()
    for file in ["bebeo", "obeo"]:  # acer files aren't needed.
        Path("endf-b7.1-ace", f"{file}.acer").unlink()

    bluemira_print("Combining cross section xml files")
    xml_handle = [
        cElementTree.parse(Path(name, "cross_sections.xml"))  # noqa: S313
        for name in lib_names
    ]
    for name, xml in zip(lib_names, xml_handle, strict=False):
        data = xml.getroot()
        remove_list = []
        for elem in data.iter():
            if elem.tag == "library":
                if elem.attrib["materials"] in thermal_files:
                    remove_list.append(elem)
                else:
                    elem.attrib["path"] = str(Path(name, elem.attrib["path"]))
        for elem in remove_list:
            data.remove(elem)

    data = xml_handle[0].getroot()
    for xml in xml_handle[1:]:
        data.extend(list(xml.getroot().iter())[1:])

    xml_handle[0].write("cross_sections.xml")


def download_data(
    download: Callable, libs: tuple[ModuleType, ...], lib_names: tuple[str, ...]
):
    """Download neutronics data"""
    for name, lib in zip(lib_names, libs, strict=False):
        bluemira_print(f"Downloading {name} cross section data")
        lib.download = download
        lib.extract = extractor
        lib.args.destination = Path(name)
        with patch("builtins.print", new=_convertion_progress):
            lib.main()
        print()


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments

    Returns
    -------
    :
        Parsed argument namespace
    """
    parser = argparse.ArgumentParser("Bluemira Neutronics data downloader")

    parser.add_argument("-l", "--location", default=Path.cwd() / "bluemira_openmc_data")
    parser.add_argument("--download-threads", type=int, default=5)
    parser.add_argument("--isotope-file", type=str, default="nuclear_data_isotopes.json")
    p, unknown = parser.parse_known_args(args)
    p.location = Path(p.location)
    sys.argv = [sys.argv[0], *unknown]
    return p


class ChgDir:
    """Change directory context manager"""

    def __init__(self, path: Path):
        self.path = path
        self.origin = Path().absolute()

    def __enter__(self):
        """Change directory"""
        chdir(self.path)

    def __exit__(self, typ, exc, tb):
        """Revert directory change"""
        chdir(self.origin)


def main(p):
    """Main function"""
    root_folder = p.location
    root_folder.mkdir(parents=True, exist_ok=True)

    download = functools.partial(downloader, max_workers=p.download_threads)

    # Imported after parsing arguments because argparse is called on import here...
    from openmc_data import utils  # noqa: PLC0415

    utils.state_download_size = state_download_size

    import openmc_data.convert.convert_endf as endf  # noqa: PLC0415
    import openmc_data.convert.convert_tendl as tendl  # noqa: PLC0415

    with open(Path(Path(__file__).resolve().parent, p.isotope_file)) as fh:
        isotope_data = json.load(fh)

    libs = tuple(
        a for a in (tendl, endf) if a.__name__.rsplit("_", 1)[-1] in isotope_data
    )
    lib_names = tuple(lib.__name__.split("_")[-1] for lib in libs)

    with ChgDir(root_folder):
        download_data(download, libs, lib_names)

        # convert_endf crashes if you dont have these available...
        thermal_files = ("c_O_in_BeO", "c_Be_in_BeO")
        thermal_prefix = endf.args.destination / "neutron"

        combine_xml(lib_names, thermal_files, thermal_prefix)


if __name__ == "__main__":
    p = parse_args(None)
    filter_members = functools.partial(
        _filter_members, str(Path(__file__).resolve()), p.isotope_file
    )
    main(p)
