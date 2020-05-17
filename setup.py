"""
Yet another Covid model.
"""

import setuptools as st


setuptools.setup(
    name = 'Yacomo',
    author = 'Ernest Pusateri',
    author_email = 'erniep@gmail.com',
    long_descriptioin =__doc__,
    packages=st.find_packages(),
    zip_safe = False,
    install_requires = [
        'pyyaml==5.3.1',
        'click==7.1.2',
        'numpy==1.18.4',
        'scipy==1.4.1'
    ],
    python_requires='==3.7.7',
    entry_points = {
        'console_scripts': [
            'yacomo = yacomo.cli:main'
        ]
    }
)    
