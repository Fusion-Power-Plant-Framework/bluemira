import pytest

import tests.bluemira.test_component_integration as bm_test_reactor
import tests.BLUEPRINT.test_reactor as bp_test_reactor  # noqa :N813

# =============================================================================
# Smoke test reactor fixtures
# =============================================================================


@pytest.fixture(scope="session")
def reactor():
    reactor = bp_test_reactor.SmokeTestSingleNullReactor(
        bp_test_reactor.config,
        bp_test_reactor.build_config,
        bp_test_reactor.build_tweaks,
    )
    reactor.build()
    return reactor


@pytest.fixture(scope="session")
def BLUEPRINT_integration_reactor():
    reactor = bm_test_reactor.BluemiraReactor(
        bm_test_reactor.config,
        bm_test_reactor.build_config,
        bm_test_reactor.build_tweaks,
    )
    reactor.build()
    return reactor
