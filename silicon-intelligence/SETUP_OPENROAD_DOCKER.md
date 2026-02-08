# OpenROAD Installation Guide (Docker on Windows)

Since you have **WSL** and **Docker** installed, the easiest way to run OpenROAD is via a Docker container. This avoids complex compilation steps.

## High-Speed Setup

1.  **Pull the Image**:
    Run this command in your terminal (PowerShell or Command Prompt):
    ```powershell
    docker pull openroad/openroad
    ```

2.  **Run OpenROAD**:
    Use the provided wrapper script located in the `scripts/` folder:
    ```powershell
    .\scripts\run_openroad_docker.bat
    ```
    This script automatically mounts your current directory into the container, so OpenROAD can see your design files.

## Manual Usage
If you prefer identifying the command yourself:
```powershell
docker run -it -v ${PWD}:/OpenROAD/design -w /OpenROAD/design openroad/openroad
```

## Verification
Once inside the container (you should see a prompt like `root@...`), type:
```bash
openroad -help
```
If you see the help message, OpenROAD is ready to receive commands from the `Silicon Intelligence` system!
