import torchaudio

# A call to initialize_sox is necessary for effect chaining needed for
# downsampling high rate audio samples to 16KHz. However, it must never be
# called more than once in a single runtime and we must call shutdown_sox
# once we are done with using effect chaining in the runtime.
# Hence, we import this file in any datasets that need effect chaining to make
# the necessary call. If another dataset has already imported this, the file
# will not be run again to avoid calling initialize_sox again. The _s variable
# is a reference that will prevent calling __del__ on Sox until the Python
# runtime is shutting down at which point we can safely break any effect
# chaining since we know that we won't be loading in any more samples.


class _Sox:
    def __init__(self):
        torchaudio.initialize_sox()

    def __del__(self):
        torchaudio.shutdown_sox()


_s = _Sox()
