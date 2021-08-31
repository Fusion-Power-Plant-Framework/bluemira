# Running examples as Jupyter notebooks

The BLUEPRINT examples have been written with the intention that they will be run as Jupyter notebooks. This allows the code to be run in convenient blocks and displays most images within the Jupyter session. However, it is still possible to run the examples directly in Python, or to manually select and run sections of the files in a Python interpreter.

## Setting up and running a Jupyter server

Within your BLUEPRINT virtual environment, run the following to ensure Jupyter is installed with notebook support - this lets us run the examples within a web browser:

```bash
pip install notebook
```

When the install has completed, we can start a Jupyter server by running the following from your ~/code/BLUEPRINT directory:

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

You should see a bunch of files, which correspond to what's in your BLUEPRINT directory.

You can stop the Jupyter server by double-pressing `ctrl+c` in your terminal.

## Maintaining the examples

To add a new example, or to edit an existing one, make your changes in the `.py` version
of the example and then run the `convert_py_to_ipynb.py` script. This will convert all
`.py` files with notebook blocks e.g. `# %%` to the corresponding `.ipynb` file.
