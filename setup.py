from setuptools import setup


setup(
    name="szifi",
    version="0.1",
    description="SZ cluster finder",
    zip_safe=False,
    packages=["szifi"],
    author = 'Inigo Zubeldia',
    author_email = 'inigo.zubeldia@ast.cam.ac.uk',
    url = 'https://github.com/inigozubeldia/szifi',
    download_url = 'https://github.com/inigozubeldia/szifi',
    package_data={
        # "specdist": ["data/*txt"],
        #"data/ct_database/case_1_040520/*txt"]#,
    },

)
