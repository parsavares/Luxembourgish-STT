https://colab.research.google.com/drive/1ibPONCmdKUyjRJVcOE8fUN61mSMEbM2n?usp=sharing

## pre-requirements:
### 1. 
Enable Long Paths via Group Policy (for Windows Pro and Enterprise users):

    Press Win + R, type gpedit.msc, and press Enter to open the Group Policy Editor.
    Navigate to Computer Configuration > Administrative Templates > System > Filesystem.
    Double-click on the "Enable Win32 long paths" option.
    Set it to "Enabled" and click "OK".

Enable Long Paths via Registry Editor (for Windows Home users):

    Press Win + R, type regedit, and press Enter to open the Registry Editor.
    Navigate to HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem.
    Find the LongPathsEnabled entry. If it doesn’t exist, create a new DWORD (32-bit) Value named LongPathsEnabled.
    Set the value to 1.
    Restart your computer to apply the changes.
### 2.
Install NVIDIA Drivers

Ensure that you have the latest NVIDIA drivers installed. You can download and install them from the NVIDIA Driver Downloads page.
Step 2: Install CUDA Toolkit

Since you want to use CUDA, you’ll need to install the CUDA Toolkit. Here’s how:

    Download CUDA Toolkit:
        Go to the NVIDIA CUDA Toolkit download page.
        Select your operating system and version, and download the installer for CUDA 11.x (e.g., CUDA 11.8 is recommended as it is compatible with most recent PyTorch versions).

    Install CUDA Toolkit:
        Run the installer and follow the on-screen instructions.
        During installation, make sure to install both the CUDA toolkit and the NVIDIA drivers if prompted.

    Verify CUDA Installation:
        After installation, you should be able to run nvcc --version in your terminal to verify the installation.
        Additionally, running nvidia-smi should give you detailed information about your GPU and the installed CUDA version.

        
=> Run `nvidia-smi` in your Python terminal of your VS code
```
PS D:\GIT\Luxembourgish-STT> nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.94                 Driver Version: 560.94         CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce GTX 960M      WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A    0C    P8             N/A /  200W |       0MiB /   4096MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```


### 3.
Install PyTorch with CUDA Support

Once CUDA is installed, you can install PyTorch with CUDA support:

    Install PyTorch with CUDA:
        Install PyTorch using the following command, which will automatically select the appropriate CUDA version (replace cu118 with cu116 or cu113 if you installed a different version of CUDA):

    bash

    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

Step 4: Verify PyTorch Installation

After installation, verify that PyTorch is using CUDA:

    Run a Test Script:

    python

import torch

print("PyTorch version:", torch.__version__)
print("Is CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

    If torch.cuda.is_available() returns True, your setup is correct, and you can use CUDA with PyTorch.
