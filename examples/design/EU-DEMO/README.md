# EU-DEMO Reactor Design

This folder contains the EU-DEMO reactor design. To run the reactor design you can run either the `EUDEMO_reactor.py` or `EUDEMO_reactor.ipynb` files. For information on how to run the notebook (`ipynb`) please see the examples README [here](../../README.md).

In future this will be moved to a separate repository. It should be used as a template for how we expect other reactor repositories to be structured. This boils down to the following:

```bash
.
├── reactorbuilders
│   ├── __init__.py
│   ├── equilibria.py
│   ├── pf_coils.py
│   ├── reactor.py
│   └── tf_coils.py
├── tests
│   ├── __init__.py
│   ├── build_config.json
│   ├── params.json
│   ├── template.json
│   ├── test_data
│   │   └── tf_coils_TripleArc_18.json
│   ├── test_equilibria.py
│   ├── test_pf_coils.py
│   ├── test_reactor.py
│   └── test_tf_coils.py
├── build_config.json
├── reactor.ipynb
├── reactor.py
├── params.json
├── README.md
└── template.json
```

Some important points:

- The `reactorbuilders` folder can be named as the developer chooses for example just the name of the reactor. All the design stages and builders are contained within this folder.

- The notebook (`.ipynb` file) is not needed but is a nice interface for a new user. We autogenerate it from the adjacent `reactor.py`  file which is the main run script of the reactor design.

- You may want to add a `setup.py` file in order to install your reactor as a package. Some subtle adjustments may be needed to the folder structure to make this easier to setup.

- We have renamed our tests folder to `EUDEMO_tests` to avoid pytest confusion with the other test folder. You do not need to do this.
