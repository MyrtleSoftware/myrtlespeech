import setuptools

setuptools.setup(
    name="speech",
    version="0.0.1",
    author="Myrtle",
    description="Speech Transcription",
    url="https://github.com/myrtlesoftware/speech",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src", exclude=["tests"]),
)
