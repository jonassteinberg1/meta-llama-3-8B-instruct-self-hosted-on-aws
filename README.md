# Meta LLaMA 3 8B Instruct Self Hosted on AWS

## Overview
This repository contains the setup and scripts to self-host Meta LLaMA 3 8B Instruct model on AWS using Runhouse.

## Contents
- **llama.py**: Script for interacting with the Meta LLaMA 3 8B Instruct model.
- **meta-llama-3-8b-instruct-runhouse.py**: Script for setting up and running the model on AWS using Runhouse.
- **.gitignore**: Specifies files to be ignored by git.
- **LICENSE**: Contains the Apache-2.0 license.

## Requirements
- Python 3.8 or higher
- AWS account
- Runhouse

## Setup
1. Clone the repository.
    ```bash
    git clone https://github.com/jonassteinberg1/meta-llama-3-8B-instruct-self-hosted-on-aws.git
    cd meta-llama-3-8B-instruct-self-hosted-on-aws
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Configure AWS credentials.
4. Run the setup script:
    ```bash
    python meta-llama-3-8b-instruct-runhouse.py
    ```

## Usage
Run the model using:
```bash
python llama.py

