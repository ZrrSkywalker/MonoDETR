import os
import torch
import logging
import subprocess
import pynvml

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def get_cuda_version_nvcc():
    """Get CUDA version using nvcc --version"""
    try:
        result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            # Parse the version number from the nvcc output
            for line in result.stdout.splitlines():
                if "release" in line:
                    cuda_version = line.split()[-1]  # Extract version
                    return cuda_version
        logging.warning("Unable to fetch CUDA version from nvcc.")
        return None
    except Exception as e:
        logging.error(f"Error while fetching CUDA version using nvcc: {e}")
        return None

def get_gpu_info():
    """Fetch GPU information including memory usage"""
    try:
        pynvml.nvmlInit()  # Initialize the NVML library

        device_count = pynvml.nvmlDeviceGetCount()
        logging.info(f"Number of GPU devices: {device_count}")

        # Fetch information for each GPU device
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)  # No need to decode here
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = memory_info.total / (1024 ** 2)  # Convert to MB
            used_memory = memory_info.used / (1024 ** 2)    # Convert to MB
            free_memory = memory_info.free / (1024 ** 2)    # Convert to MB

            logging.info(f"GPU {i}: {name}")
            logging.info(f"   Total Memory: {total_memory:.2f} MB")
            logging.info(f"   Used Memory: {used_memory:.2f} MB")
            logging.info(f"   Free Memory: {free_memory:.2f} MB")

        pynvml.nvmlShutdown()  # Shutdown the NVML library
    except Exception as e:
        logging.error(f"Error while fetching GPU information: {e}")
        logging.warning("Unable to fetch GPU information.")


def check_pytorch_cuda():
    try:
        # Log the PyTorch version
        logging.info(f"PyTorch Version: {torch.__version__}")

        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        logging.info(f"CUDA Available: {cuda_available}")

        # Set a default value for cuda_version
        cuda_version = None
        
        # Log the CUDA version if available
        if cuda_available:
            cuda_version = torch.version.cuda
            logging.info(f"CUDA Version (PyTorch): {cuda_version}")
        else:
            logging.warning("CUDA is not available in PyTorch. Please check your installation.")

        # Get CUDA version from nvcc
        cuda_version_nvcc = get_cuda_version_nvcc()
        logging.info(f"CUDA Version (nvcc): {cuda_version_nvcc}")

        # Compare the CUDA versions
        if cuda_version and cuda_version_nvcc:
            if cuda_version == cuda_version_nvcc:
                logging.info(f"CUDA versions match: {cuda_version}")
            else:
                logging.warning(f"CUDA versions do not match: PyTorch uses {cuda_version}, nvcc reports {cuda_version_nvcc}")

        # Get GPU information
        get_gpu_info()

        return cuda_available, torch.__version__, cuda_version, cuda_version_nvcc

    except Exception as e:
        logging.error(f"Error while checking PyTorch and CUDA: {e}")
        return None, None, None, None

def get_cuda_versions():
    # Check available CUDA versions from the CUDA directory
    cuda_versions = []
    cuda_base_path = '/usr/local/'

    for folder in os.listdir(cuda_base_path):
        if folder.startswith("cuda-"):
            cuda_versions.append(folder)

    if cuda_versions:
        logging.info("Available CUDA Versions on the system:")
        for version in cuda_versions:
            logging.info(f"- {version}")
    else:
        logging.warning("No CUDA versions found on the system.")

    return cuda_versions

if __name__ == "__main__":
    # Check PyTorch CUDA availability
    check_pytorch_cuda()

    # List available CUDA versions on the system
    get_cuda_versions()

