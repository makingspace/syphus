from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))
setup(
    name='syphus',
    version='1.3.2',
    description='MakeSpace Hill-Climbing Algorithm',
    url='https://github.com/makingspace/syphus',
    author='Zach Smith',
    author_email='zach.smith@makespace.com',
    license='Proprietary software - not for distribution',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # Pick your license as you wish (should match "license" above)
        'License :: Other/Proprietary License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    packages=['syphus'],
    extras_require={'test': ['hypothesis'], })
