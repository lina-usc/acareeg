# Authors: Christian O'Reilly <christian.oreilly@gmail.com>
# License: MIT

from setuptools import setup


if __name__ == "__main__":
    hard_dependencies = ('numpy', 'scipy')
    install_requires = list()
    with open('requirements.txt', 'r') as fid:
        for line in fid:
            req = line.strip()
            for hard_dep in hard_dependencies:
                if req.startswith(hard_dep):
                    install_requires.append(req)

    setup(
        name='acareeg',
        version="0.0.1",
        description='Code to work with EEG at ACAR.',
        python_requires='>=3.5',
        author="Christian O'Reilly",
        author_email='christian.oreilly@gmail.com',
        url='https://github.com/christian-oreilly/acareeg',
        install_requires=install_requires
    )
