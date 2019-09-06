import setuptools

setuptools.setup(
    name="myrtlespeech",
    version="0.0.1",
    author="Myrtle",
    description="Myrtle Speech Transcription Research",
    url="https://github.com/myrtlesoftware/myrtlespeech",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", exclude=["tests"]),
    include_package_data=True,
)
