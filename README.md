Aqui está a versão aprimorada do markdown, explicando a conexão ao notebook pelo VSCode:

---

# nbeats-5g-forecasting
This repository contains the code for a research study exploring the use of the N-BEATS algorithm for forecasting 5G network quality.

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

This approach allows you to work seamlessly between Docker, JupyterLab, and VSCode, making your workflow smooth and efficient!

