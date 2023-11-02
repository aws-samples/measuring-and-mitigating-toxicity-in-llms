## Measuring and Mitigating Toxicity in LLMs

Building and operating machine learning applications responsibly requires an active, consistent approach to prevent, assess, and mitigate harm. This workshop guides you through how to identify toxicity in traditional and generative AI applications, from understanding the causes and consequences of toxicity to developing and implementing mitigation strategies, including dataset and metric selection.

## Setup - local

Execute the following commands to set up the Jupyter environment; if working on Windows, make sure [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) is up-to-date before executing the below. The setup instructions assume that the repository was already cloned with e.g., ```git clone https://github.com/aws-samples/measuring-and-mitigating-toxicity-in-llms.git```.

```
conda create -n toxicity python=3.10
conda activate toxicity
pip install -r requirements.txt
python -m ipykernel install --user --name=toxicity
```

Troubleshooting for `FileNotFoundError`: Make sure you are inside the repository; if not use `cd measuring-and-mitigating-toxicity-in-llms`.

## Setup - Studio Lab

Instead of installing the environment locally, you can use the links below to open the notebooks using a pre-configured environment.

For this, you will need to provide an email and wait for SageMaker Studio Lab to approve the account. Account availability varies, so instant access cannot be guaranteed. After launching SageMaker Studio Lab the installation will take approx. 10-15'.

| Notebook Name | Studio lab |
| :---: | ---: |
| Explainability (Tabular Data)| [![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/aws-samples/measuring-and-mitigating-toxicity-in-llms/blob/main/Measuring_Mitigating_Toxicity.ipynb)|

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
