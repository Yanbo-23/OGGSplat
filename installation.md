# Installation

Follow the steps below to set up the environment for **OGGSplat**:


## Step 1: Clone the Repository

```bash
git clone https://github.com/Yanbo-23/OGGSplat.git
cd OGGSplat
```

## Step 2: Install the Environment from Splatt3R

Please follow the instructions in the [Splatt3R repository](https://github.com/btsmart/splatt3r) to set up the required environment. This typically involves creating a conda environment and installing PyTorch and other base dependencies.

## Step 3: Install the Dependencies of Modeified APE
Please follow the instructions in the [APE](https://github.com/Atrovast/APE) to set up the required environment.


## Step 4: Install the UniDepthV2
Please follow the instructions in the [UniDepthV2](https://github.com/lpiccinelli-eth/UniDepth) to set up the UniDepthV2.

## Step 5: Install the Rasterization Module

```bash
cd oggsplat_rasterization
pip install -e .
cd ..
```

## Step 6: Install Remaining Dependencies

```bash
pip install -r requirements.txt
```


