import matplotlib.pyplot as plt

from tests.BLUEPRINT.test_reactor import (
    SmokeTestSingleNullReactor,
    build_config,
    build_tweaks,
    config,
)

reactor = SmokeTestSingleNullReactor(config, build_config, build_tweaks)
reactor.build()

root_path = reactor.file_manager.reference_data_dirs["root"]
reactor_name = reactor.params["Name"]

# Save the systems
reactor.ATEC.save(f"{root_path}/{reactor_name}_ATEC.pkl")
reactor.BB.save(f"{root_path}/{reactor_name}_BB.pkl")
reactor.CR.save(f"{root_path}/{reactor_name}_CR.pkl")
reactor.DIV.save(f"{root_path}/{reactor_name}_DIV.pkl")
reactor.PF.save(f"{root_path}/{reactor_name}_PF.pkl")
reactor.PL.save(f"{root_path}/{reactor_name}_PL.pkl")
reactor.RS.save(f"{root_path}/{reactor_name}_RS.pkl")
reactor.TF.save(f"{root_path}/{reactor_name}_TF.pkl")
reactor.TS.save(f"{root_path}/{reactor_name}_TS.pkl")
reactor.VV.save(f"{root_path}/{reactor_name}_VV.pkl")

if __name__ == "__main__":
    # Plot the reactor to allow manual checks
    reactor.plot_xz()

    plt.show()
