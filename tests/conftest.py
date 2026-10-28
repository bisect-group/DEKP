def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: bf16/AMP tolerance tests that need a CUDA device")
