#======================#
# Install, clean, test #
#======================#
reinstall_package:
	@pip uninstall -y cf_copilot || :
	@pip install -e .

install_requirements:
	@pip install -r requirements.txt

install:
	@pip install . -U

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr proj-*.dist-info
	@rm -fr proj.egg-info

test_structure:
	@bash tests/test_structure.sh

#======================#
#          API         #
#======================#

run_api:
	uvicorn cf_copilot.api.fast:app --reload --port 8000


#======================#
#          GCP         #
#======================#

gcloud-set-project:
	gcloud config set project $(GCP_PROJECT)



#======================#
#         Docker       #
#======================#

# Uses DOCKER_IMAGE_NAME from .env
# Local images - using local computer's architecture
# i.e. linux/amd64 for Windows / Linux / Apple with Intel chip
#      linux/arm64 for Apple with Apple Silicon (M1 / M2 chip)

docker_build_local:
	docker build --tag=$(DOCKER_IMAGE_NAME):local .

docker_run_local:
	docker run \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_NAME):local

docker_run_local_interactively:
	docker run -it \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_NAME):local \
		bash

# Cloud images - using architecture compatible with cloud, i.e. linux/amd64

DOCKER_IMAGE_PATH := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/$(DOCKER_REPO_NAME)/$(DOCKER_IMAGE_NAME)

docker_show_image_path:
	@echo $(DOCKER_IMAGE_PATH)

docker_build:
	docker build \
		--platform linux/amd64 \
		-t $(DOCKER_IMAGE_PATH):prod .

# Alternative if previous doesn´t work. Needs additional setup.
# Probably don´t need this. Used to build arm on linux amd64
docker_build_alternative:
	docker buildx build --load \
		--platform linux/amd64 \
		-t $(DOCKER_IMAGE_PATH):prod .

docker_run:
	docker run \
		--platform linux/amd64 \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_PATH):prod

docker_run_interactively:
	docker run -it \
		--platform linux/amd64 \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_PATH):prod \
		bash

#======================#
#  GCP CLOUD PROTOCAL  #
#======================#

# 1) Configure Docker auth
docker_allow:
	gcloud auth configure-docker $(GCP_REGION)-docker.pkg.dev

# 2) Create Artifact Registry repo
docker_create_repo:
	gcloud artifacts repositories create $(DOCKER_REPO_NAME) \
		--repository-format=docker \
		--location=$(GCP_REGION) \
		--description="Repository for storing docker images"

# 3) Build image for Cloud Run (linux/amd64, prod tag)
# use docker_build mentioned above uses, $(DOCKER_IMAGE_PATH)

# 4) Push image to Artifact Registry
docker_push:
	docker push $(DOCKER_IMAGE_PATH):prod

# 5) Deploy to Cloud Run with memory + env file
docker_deploy:
	gcloud run deploy \
		--image $(DOCKER_IMAGE_PATH):prod \
		--memory $(GAR_MEMORY) \
		--region $(GCP_REGION) \
		--env-vars-file .env.yaml

#======================#
#      GCP CLEANUP     #
#======================#

# Delete the Artifact Registry repository (and all images inside)
# WARNING: This is destructive.
delete_artifact_repo:
	gcloud artifacts repositories delete $(DOCKER_REPO_NAME) \
	  --location=$(GCP_REGION) \
	  --quiet

# List Cloud Run services in the configured region (for inspection)
list_cloud_run_services:
	gcloud run services list --region=$(GCP_REGION)

# The SERVICE column there is what you should pass as SERVICE_NAME:
# make delete_cloud_run_service SERVICE_NAME=<SERVICE from list>
# Delete a Cloud Run service by name
# Usage: make delete_cloud_run_service SERVICE_NAME=your-service-name
delete_cloud_run_service:
	gcloud run services delete $(SERVICE_NAME) \
	  --region=$(GCP_REGION) \
	  --quiet


#======================#
#         TESTS        #
#======================#

test_api_root:
	pytest \
	  tests/api/test_endpoints.py::test_root_is_up \
	  tests/api/test_endpoints.py::test_root_returns_greeting \
	  --asyncio-mode=strict -W "ignore"

test_api_predict:
	pytest \
	tests/api/test_endpoints.py::test_predict_is_up --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_predict_is_dict --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_predict_has_key --asyncio-mode=strict -W "ignore" \
	tests/api/test_endpoints.py::test_predict_val_is_float --asyncio-mode=strict -W "ignore"

test_api_on_docker:
	pytest \
	tests/api/test_docker_endpoints.py --asyncio-mode=strict -W "ignore"

test_mlflow_config:
	@pytest \
	tests/api/test_mlflow.py::TestMlflow::test_model_target_is_mlflow \
	tests/api/test_mlflow.py::TestMlflow::test_mlflow_experiment_is_not_null \
	tests/api/test_mlflow.py::TestMlflow::test_mlflow_model_name_is_not_null
