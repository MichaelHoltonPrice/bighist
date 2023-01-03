# Installation
This installation guide assumes the use of Windows Powershell, but will work
for setups with minor modifications.

Clone this repository:

```bash
git clone https://github.com/MichaelHoltonPrice/bighist
cd bighist
```

If desired, create and activate a virtual environment named bighist_env:

```bash
python -m venv bighist_env
Set-ExecutionPolicy Unrestricted -Scope Process
.\bighist_env\Scripts\activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

Do the actual installation:

```bash
python setup.py install
```

Start Python and check the installation (then exit):

```bash
python
import bighist as bh
exit()
```

# Running an analysis
Some example analyses are available here:


My vision is that third party researchers will adopt a similar approach. That
is, bighist provides a core set of tools that multiple analyses and research
articles rely on and new analyses can use, with the stand-alone, project
specific code located elsewhere.

# Running the tests

```bash
python -m unittest tests/test_bighist.py
```

# Citing
For citing this Python package specifically:

TODO: add

For citing the seshat project generally:

TODO: add

For citing the classic seshat dataset:

TODO: add

For citing the Equinox seshat dataset:

TODO: add