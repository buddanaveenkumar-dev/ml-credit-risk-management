# Build on AWS, Google Cloud or NVIDIA GPUs

**Planning guide — these options have not been deployed or benchmarked.** Published results still come from local server CPU training. NVIDIA provides GPU hardware and software that can run on a local server, AWS or Google Cloud; it is not a mutually exclusive cloud choice.

## AWS server and tools

| Purpose | Tools |
|---|---|
| First server | Amazon EC2 Linux VM with Python and the existing Streamlit app |
| Data and artifacts | Private Amazon S3 bucket for permitted datasets, saved models and reports |
| Managed training option | Amazon SageMaker AI custom training jobs; Python SDK / Boto3 |
| Container packaging | Docker and Amazon ECR |
| Access and operations | IAM roles, AWS Secrets Manager, CloudWatch; HTTPS reverse proxy for the app |

Build sequence:

1. Create a project environment, choose a region and budget, and provision a CPU EC2 server. Attach an IAM role limited to the required storage paths.
2. Clone this repository, install its dependencies and download permitted data to the existing `data/raw/` paths. For private S3 inputs, add a staging step to copy them into those paths.
3. Run `python scripts/run_all.py`. Save generated reports and models to private S3; provide only aggregate reports to the viewer.
4. Run Streamlit as a managed service behind HTTPS and authentication. Keep its internal port private.
5. For repeatable managed training, package the training script in a container, adapt input/output paths for SageMaker, and submit a training job. This integration is additional work, not included in the current repository.

SageMaker can run custom training code and provisions training compute. See [AWS training documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html) and [execution roles](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).

## Google Cloud (GCP) server and tools

| Purpose | Tools |
|---|---|
| First server | Compute Engine Linux VM with Python and Streamlit |
| Data and artifacts | Private Cloud Storage bucket |
| Managed training option | Google Cloud managed custom training (Vertex AI training documentation; now redirects to Agent Platform training) |
| Container packaging and app hosting | Docker, Artifact Registry; Cloud Run as an optional container hosting route |
| Access and operations | IAM service accounts, Secret Manager, Cloud Logging and Cloud Monitoring |

Build sequence:

1. Create a Google Cloud project, choose a region and budget, and provision a CPU Compute Engine VM with a narrowly scoped service account.
2. Clone the repository and install dependencies. Stage permitted data from Cloud Storage into `data/raw/` before running the training script.
3. Run the experiment and upload reports and models to private Cloud Storage. Keep training separate from the results viewer.
4. Serve the viewer on the VM behind authenticated HTTPS. Alternatively, build a container, publish it to Artifact Registry and deploy the viewer to Cloud Run, binding Streamlit to the configured container port and `0.0.0.0`.
5. For managed custom training, package the script and explicitly configure input staging and output uploads. Reproduce the local evaluation before adopting the cloud results.

See [Google Cloud custom training](https://docs.cloud.google.com/vertex-ai/docs/training/overview). Streamlit uses WebSockets: a Cloud Run deployment must handle request timeouts and reconnects; session affinity is best effort. See [Cloud Run WebSockets](https://docs.cloud.google.com/run/docs/triggering/websockets).

## NVIDIA GPU server and tools

| Purpose | Tools |
|---|---|
| Compute | Compatible NVIDIA GPU on a local Linux server or an AWS/GCP GPU VM |
| Runtime | NVIDIA driver, compatible CUDA runtime; NVIDIA Container Toolkit for Docker |
| Boosted-tree training | GPU-enabled XGBoost |
| Optional data processing | RAPIDS cuDF; cuML for selected additional algorithms |
| Optional model serving | NVIDIA Triton Inference Server with a compatible tree-model backend |

Build sequence:

1. Choose a supported GPU and a compatible driver/CUDA/library combination. Check the current installation matrix rather than assuming any GPU will work.
2. First reproduce the CPU baseline. Then create a separate GPU experiment using XGBoost settings `tree_method="hist", device="cuda"`.
3. Keep the same inputs, outcome definition and temporal cohorts. Compare elapsed time, memory use, cost, ranking and calibration. GPU training does not automatically improve predictive quality.
4. Only migrate preprocessing to cuDF/cuML when measurements justify it; the existing pandas/scikit-learn pipeline is not automatically GPU accelerated.
5. Add Triton only if an inference service is needed. Export a supported model and reproduce all preprocessing in the serving path; the current Streamlit app only displays saved results.

See [XGBoost GPU configuration](https://xgboost.readthedocs.io/en/stable/gpu/index.html), [RAPIDS requirements](https://docs.rapids.ai/install/) and [Triton tree-model serving](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/fil_backend/README.html).

## What changes in this repository

For a first cloud run, use the existing Python commands on a VM. Managed training or container hosting additionally needs configurable input/output locations, a Dockerfile, cloud storage transfer steps, dependency version locking and service configuration. None of those cloud resources are provisioned by this guide.

Start with CPU compute for this project's current scale; benchmark before paying for GPUs. Account access, provider data-use permissions, regional availability and a chosen budget are prerequisites. Hosting changes do not resolve missing mortgage data, HMDA download restrictions, model calibration or regulatory validation.
