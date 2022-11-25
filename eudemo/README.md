# EU-DEMO Reactor Design

This folder contains the EU-DEMO reactor design.
To run the reactor design, or the EUDEMO tests,
you must add the `eudemo` folder to your python path:

```bash
export PYTHONPATH="<path/to/bluemira>/eudemo:${PYTHONPATH}"
```

Run the design using the `reactor.py` file:

```console
python eudemo/eudemo/reactor.py
```

In future this will be moved to a separate repository.
It should be used as a template for how we expect
other reactor repositories to be structured.
