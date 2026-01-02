# Docker Setup Guide for SEGALE

This guide will help you set up and use Docker to run SEGALE evaluation, even if you're new to Docker.

## What is Docker?

Docker is a tool that packages software and its dependencies into "containers" - isolated environments that run the same way on any computer. Think of it like a virtual machine, but lighter and faster.

## Step 1: Install Docker

### macOS:
1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop/
2. Install the `.dmg` file
3. Open Docker Desktop from Applications
4. Wait for Docker to start (you'll see a whale icon in the menu bar)

### Linux:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to docker group (to run without sudo)
sudo usermod -aG docker $USER
# Log out and back in for this to take effect
```

### Verify Installation:
```bash
docker --version
# Should show something like: Docker version 24.0.0 or higher
```

## Step 2: Build the SEGALE Docker Image

1. **Navigate to the SEGALE directory:**
```bash
cd other_repos/SEGALE
```

2. **Optional: Add your HuggingFace token** (if you have one):
   - Open `Dockerfile` in a text editor
   - Find line 52: `ENV HF_TOKEN=""`
   - Replace with: `ENV HF_TOKEN="your_token_here"`
   - Save the file
   
   > **Why?** Some models require authentication. If you don't have a token, you can skip this - the models will download when needed.

3. **Build the Docker image:**
```bash
docker build -t segale:latest .
```

   > **What this does:** This reads the `Dockerfile` and creates a Docker "image" (like a template) with all SEGALE dependencies installed.

   > **How long?** This can take 10-30 minutes the first time, as it downloads and installs everything. Subsequent builds are faster.

   > **Expected output:** You'll see lots of installation messages. At the end, you should see:
   ```
   Successfully built abc123def456
   Successfully tagged segale:latest
   ```

4. **Verify the image was created:**
```bash
docker images | grep segale
# Should show: segale   latest   abc123def456   2 minutes ago   5.2GB
```

## Step 3: Run the Test Script with Docker

Now you can use the test script with Docker:

```bash
# Go back to project root
cd ../..

# Run test with Docker
python segale/test_segale.py --output-dir outputs/wmt25/en-es/IRB/gpt-4-1 --use-docker --max-samples 5
```

The `--use-docker` flag tells the script to run SEGALE commands inside the Docker container.

## Common Docker Commands (Reference)

### Check if Docker is running:
```bash
docker ps
# Should show running containers (or be empty if nothing is running)
```

### List all Docker images:
```bash
docker images
```

### Remove an image (if you need to rebuild):
```bash
docker rmi segale:latest
```

### View Docker logs (if something goes wrong):
```bash
docker logs <container_id>
```

### Stop all containers:
```bash
docker stop $(docker ps -aq)
```

## Troubleshooting

### Problem: "Cannot connect to the Docker daemon"
**Solution:** Make sure Docker Desktop is running (macOS) or Docker service is started (Linux):
```bash
# Linux
sudo systemctl start docker
```

### Problem: "Permission denied" (Linux)
**Solution:** Add your user to the docker group:
```bash
sudo usermod -aG docker $USER
# Then log out and log back in
```

### Problem: "No space left on device"
**Solution:** Docker images can be large. Clean up unused images:
```bash
docker system prune -a
# This removes unused images, containers, and networks
```

### Problem: Build fails with "HF_TOKEN" error
**Solution:** Either:
1. Add your HuggingFace token to the Dockerfile (line 52)
2. Or remove/comment out the login line (line 54) - models will download when needed

### Problem: "segale:latest" image not found
**Solution:** Make sure you built the image:
```bash
cd other_repos/SEGALE
docker build -t segale:latest .
```

### Problem: Docker is slow
**Solution:** 
- Make sure you have enough RAM allocated to Docker (Docker Desktop → Settings → Resources)
- Close other applications to free up memory

## Understanding the Docker Command

When you use `--use-docker`, the script runs something like:

```bash
docker run --rm \
  -v /path/to/SEGALE:/workspace \
  -v /path/to/data:/data \
  -w /workspace \
  segale:latest \
  segale-align --system_file /data/system.jsonl ...
```

**Breaking it down:**
- `docker run`: Start a new container
- `--rm`: Automatically remove the container when it finishes
- `-v /path/to/SEGALE:/workspace`: Mount the SEGALE directory into the container
- `-v /path/to/data:/data`: Mount your data directory into the container
- `-w /workspace`: Set working directory inside container
- `segale:latest`: Use the image we built
- `segale-align ...`: Run the command inside the container

## Alternative: Run SEGALE Commands Directly in Docker

If you want to run SEGALE commands manually:

```bash
# Run segale-align
docker run --rm \
  -v $(pwd)/other_repos/SEGALE:/workspace \
  -v $(pwd)/outputs:/data \
  -w /workspace \
  segale:latest \
  segale-align --system_file /data/wmt25/en-es/IRB/gpt-4-1/segale_test/system.jsonl \
               --ref_file /data/wmt25/en-es/IRB/gpt-4-1/segale_test/reference.jsonl \
               --segmenter spacy \
               --task_lang es \
               --proc_device cpu \
               -v

# Run segale-eval (COMET metrics)
docker run --rm \
  -v $(pwd)/other_repos/SEGALE:/workspace \
  -v $(pwd)/outputs:/data \
  -w /workspace \
  segale:latest \
  segale-eval --input_file /data/wmt25/en-es/IRB/gpt-4-1/segale_test/system/aligned_spacy_system.jsonl
```

## Next Steps

Once Docker is working:
1. Test with a small sample: `--max-samples 5`
2. If successful, run on full dataset
3. Review the evaluation results
4. Integrate into the main codebase

## Getting Help

- Docker documentation: https://docs.docker.com/
- SEGALE repository: Check `other_repos/SEGALE/README.md`
- Check Docker logs if commands fail: `docker logs <container_id>`

