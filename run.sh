PROJECT_NAME=test

# Build docker file
#docker build -t abdullahaleem/projects:$PROJECT_NAME .

# Run with GPU
# sudo docker run --rm --gpus '"device=1,3"' --shm-size=1gb -v ~+/src:/home/src -v /home/darvin/NAS/Projects/Ptosis:/home/data/ -it abdullahaleem/projects:ptosis

# Run without GPU
docker run --rm -v ~+/:/home/ -it abdullahaleem/projects:$PROJECT_NAME


# Deploy docker file
#sudo docker push abdullahaleem/projects:$PROJECT_NAME
