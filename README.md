# 5G Network Forecasting

This repository contains the source code and workflow for a research study comparing statistical and machine learning models for predicting 5G network signal quality metrics, such as RSRP and RSRQ. These predictions aim to optimize Radio Resource Management (RRM) parameters, improving handover efficiency, network reliability, and overall user experience in dynamic 5G environments.

---

## How to Run? 🏃‍♂️🏃‍♀️

### Using Docker 🐋

#### 1. Start a Training Service
To run a training service inside Docker, use the following command:
```bash
docker compose up -d {service-name}
```

#### 2. Export Logs to a File
You can export the logs from a container to a file for review or debugging:
```bash
docker logs --tail 1000 {container-id} > {file-name}.log 2>&1
```

#### 3. Run JupyterLab in Docker
To start JupyterLab in Docker, run:
```bash
docker compose up -d jupyterlab
```

#### 4. Running GPU Services  
As there is currently an unresolved issue with **Docker Compose** and GPU support, GPU-enabled services should be run using `docker run`. For example, to run a GPU-enabled training process:

Building:
```bash
docker build -t darts_gpu_image -f Dockerfile.gpu .
```

Running:
```bash
docker run --rm --gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v $(pwd)/data:/app/data -v $(pwd)/src:/app darts_gpu_image python3 train_models_no_sliding_window.py --models deep_learning
```

---

### Handling File Manipulation in the `data` Folder
If you need to manipulate or delete files in the `data` folder but encounter permission issues, you can run a temporary Docker container with elevated permissions to access the folder directly. Use the following command:

```bash
docker run -it --rm -v $(pwd)/data:/data ubuntu /bin/bash
```

This command opens an interactive terminal in a lightweight **Ubuntu** container with the `data` folder mounted, allowing you to perform any required file operations (like deleting files). Make sure to exit the container once done to remove it automatically.

---

### Accessing Jupyter Notebooks in VSCode ✨
Once JupyterLab is running, you can connect to the notebook interface in **VSCode**.

#### Step-by-Step Instructions:

1. **Open the Jupyter Notebook in a Browser**  
   Copy and paste the following URL into your browser:
   ```
   http://localhost:8888/?token=token_here
   ```

2. **Connect VSCode to the Jupyter Server**  
   In VSCode:
   - Install the **Jupyter** and **Python** extensions from the Extensions Marketplace.
   - Open a notebook (`.ipynb` file) from your project.
   - When prompted to select a kernel, click **"Select Kernel"** at the top of the notebook.
   - Choose **"Existing Jupyter Server"** and enter the URL of the JupyterLab instance running in Docker:
     ```
     http://localhost:8888/?token=token_here
     ```

Now you can edit and run the Jupyter notebooks inside **VSCode**, with the backend executing on the Docker container.

---

### Datasets 🎲

- [5G Production Dataset][def]

[def]: https://github.com/uccmisl/5Gdataset

---

This workflow ensures compatibility across GPU and non-GPU systems, providing flexibility and efficiency for your experiments.