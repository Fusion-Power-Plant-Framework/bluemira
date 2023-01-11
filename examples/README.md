# Running examples as Jupyter notebooks

The bluemira examples have been written with the intention that they will be run as Jupyter notebooks. This allows the code to be run in convenient blocks and displays most images within the Jupyter session. However, it is still possible to run the examples directly in Python, or to manually select and run sections of the files in a Python interpreter.

## Setting up and running a Jupyter server

Within your bluemira virtual environment, run the following to ensure Jupyter is installed with notebook support - this lets us run the examples within a web browser:

```bash
pip install notebook jupytext
```

When the install has completed, we can start a Jupyter server by running the following from your ~/code/bluemira directory:

```bash
python -m notebook --no-browser
```

Look at the output from the Jupyter start up and you will see a section like the below, where ... is replaced with some random characters:

```bash
To access the notebook, open this file in a browser:
    file:///home/{username}/.local/share/jupyter/runtime/nbserver-{notebook_id}-open.html
Or copy and paste one of these URLs:
    http://localhost:8888/?token=...
 or http://127.0.0.1:8888/?token=...
```

Open a web browser (e.g. Firefox or Chrome) and navigate to [your local Jupyter server](http://localhost:8888) (you don't actually need the token that Jupyter creates, because everything is running locally and your computer already knows who you are).

You should see a bunch of files, which correspond to what's in your bluemira directory.

You can stop the Jupyter server by double-pressing `ctrl+c` in your terminal.

## Maintaining the examples

To add a new example, or to edit an existing one, make your changes in the `.py` file
of the example. Use percentage blocks (`# %%` or `# %% [markdown]`) to mark cells
of the notebook. Your IDE or Jupytext if you are using jupyter will render the python file as a
notebook.

To include the examples in the rendered documentation please add the file to `examples.rst`.
The first line of the python file should be `# %% nbsphinx="hidden"` to hide the copyright notice
from the rendered script in the documentation (the copyright is in the footer of all documentation
pages).
