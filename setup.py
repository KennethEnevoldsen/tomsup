from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='tomsup',
   version='1.0',
   description='An implementation of game theory of mind in a agent based framework following the implementation of Devaine, et al. (2017).',
   license='Apache License 2.0',
   long_description=long_description,
   long_description_content_type="text/markdown",
   author='Kenneth C. Enevoldsen and Peter T. Waade',
   author_email='kennethcenevoldsen@gmail.com',
   url="https://github.com/KennethEnevoldsen/tomsup",
   packages=['tomsup'],  #same as name
   install_requires=['numpy', 'pandas', 'scipy'], #external packages as dependencies

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    keywords='theory-of-mind tom game-theory',
)