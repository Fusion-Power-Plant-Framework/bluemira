import pathlib

from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.materials.cache import MaterialCache

test_materials_data_path = get_BP_path("materials/test_data", subfolder="tests")
test_materials_data_path = pathlib.Path(test_materials_data_path)
test_materials_path = test_materials_data_path / "materials.json"
test_mixtures_path = test_materials_data_path / "mixtures.json"
TEST_MATERIALS_CACHE: MaterialCache = MaterialCache()
TEST_MATERIALS_CACHE.load_from_file(test_materials_path.absolute())
TEST_MATERIALS_CACHE.load_from_file(test_mixtures_path.absolute())
