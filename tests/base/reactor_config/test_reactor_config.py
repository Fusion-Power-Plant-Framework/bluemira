# import pytest


from bluemira.base.reactor_config import ReactorConfig


class TestReactorConfigClass:
    """
    Tests for the Reactor Config class functionality.
    """

    def test_loading(self):
        reactor_config = ReactorConfig()
        reactor_config.global_params
        # with pytest.raises(ComponentError):
        #     ...
