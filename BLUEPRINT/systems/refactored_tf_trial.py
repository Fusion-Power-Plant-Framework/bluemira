from abc import abstractclassmethod
import numpy as np


class CorrectorFactory:
    def __new__(cls, shape_type, input_loops, **kwargs) -> "TFCoilsCorrector":
        if shape_type in ["TP", "CP"]:
            return ResistiveCoilsCorrector(input_loops)
        elif shape_type == "S":
            return SplineCorrector(input_loops)
        else:
            ValueError(f"Unknown shape_type {shape_type}")


class TFCoilsCorrector(ToroidalFieldCoils):
    def __init__(self, input_loops):
        self._loops = input_loops

    def say_hi(self):
        pass

    def make_array(self):
        pass

    def _generate_xz_loops(self):
        super()._generate_xz_loops()
        # Do the correction
        ...
        return self._loops


class PictureFrameCorrector(TFCoilsCorrector):
    def say_hi(self):
        print(f"Hi, I'm {self.__class__.__name__}")

    def make_array(self):
        pass

    def _generate_xz_loops(self):
        super()._generate_xz_loops()
        ...
        return self._loops


class ResistiveCoilsCorrector(PictureFrameCorrector):

    """
    write stuff here
    """

    pass


class SplineCorrector(TFCoilsCorrector):
    def say_hi(self):
        print(f"Hello, I'm {self.__class__.__name__}")

    def make_array(self):
        pass


if __name__ == "__main__":
    shape_type = "S"
    thing = CorrectorFactory(shape_type)
    thing.say_hi()
