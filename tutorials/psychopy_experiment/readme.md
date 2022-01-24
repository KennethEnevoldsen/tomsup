Here are some simple instructions for getting the psychopy experiment to run.

1) Install tomsup "pip install tomsup"

2) Install psychopy. This can also be done with pip install. Note that a lot of dependencies are installed this way, so in case one wants to avoid that, follow 2.a
    2.a) psychopy can be installed with "pip install psychopy --no-deps" (as by the website https://www.psychopy.org/download.html#pip-install). If this is done, some dependencies might need to be installed manually.
    These include
    future / pyyaml / wxpython / json_tricks / pyglet / python-bidi / freetype-py / requests
    Note also that the freetype library needs to be installed. If it isn't this can be done with homebrew on macs, and a google search should show how to do it on a windows.

3) Make sure to download the folder containing the python script as well as the 'images' folder.
The repo can be found here: https://github.com/KennethEnevoldsen/tomsup/tree/master/tutorials/psychopy_experiment

4) Make sure that if there is a 'data' folder in the directory, it doesn't contain anything except for files produced by this script. If there isn't a 'data' folder, the script will create one.

The python script should now run.
Note that:
- saving internal states is only possible for tom and QL agents.
- the syntax for specifying opponent parameters needs to follow the structure used by tomsup. For example:
        {'bias': 0.7} #for an RB agent
        {'volatility': -2, 'b_temp': -1, 'bias': 0, 'dilution': None} #for a TOM agent'
    Unspecified parameter values are set to tomsup's defaults.
- on some computers, there might be an issue with closing Psychopy after the experiment is finished. Here, cmd+Q or ctrl+Q can solve it 