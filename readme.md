# Hybrid Taxi System Simulation.
## Overview
This project simulates a hybrid taxi system. The system relies on multiple components, including the Gurobi optimizer 
and OSRM (Open Source Routing Machine) for routing. The following sections guide you through the environment setup, 
dependency installation, and how to run the simulation.

## Dependency installation

### 1. Create the virtual environment

Create a new virtual environment in an Anaconda Prompt with all the packages listed in `hybrid_taxi_system.yml`. 
The default name of the environment is called `hybrid-taxi-system`

    conda env create -f hybrid_taxi_system.yml

Check out the created virtual environment.

    conda activate hybrid-taxi-system

### 2. Optimizer(Gurobi)
Gurobi: Set gurobi channel on top of your channel list by twice calling

    conda config --add channels http://conda.anaconda.org/gurobi

Install gurobi package by

    conda install gurobi

Free academic licenses of Gurobi can be acquired. See https://www.gurobi.com/academia/academic-program-and-licenses/ 
for more details in installation instructions.

### 3. Pull the osrm-backend image

    docker pull osrm/osrm-backend

## Dataset

Due to the confidentiality of the original trip data, we are unable to provide the exact dataset used in our paper. 
We have generated a set of hypothetical data, `customers_HZ.csv`, located in the `vehicle_demo_hybrid/data`, based on 
the temporal characteristics of the trip requests. This simulated dataset yield results consistent with the conclusions presented in the paper.

## How to run

### 1. Start the OSRM server using Docker
Pre-process the `hangzhou.osm` with the car profile and start a routing engine HTTP server on port 5000

Enter the `vehicle_demo_hybrid/data/osrm` folder

    docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-extract -p /opt/car.lua /data/hangzhou.osm

The flag `-v "${PWD}:/data"` creates the directory `/data` inside the docker container and makes the current working directory 
`"${PWD}"`available there. The file `/data/hangzhou.osm` inside the container is referring to `"${PWD}/hangzhou.osm"` on the host.

Then run:

    docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-partition /data/hangzhou.osrm
    docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-customize /data/hangzhou.osrm

Start the OSRM server:

    docker run -t -i -p 5000:5000 -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-routed --algorithm mld /data/hangzhou.osrm --max-table-size 10000

The default port number is `5000`, if port 5000 is already in use by another program, consider changing it to an available port, such as `8080`:
    
    docker run -t -i -p 8080:5000 -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-routed --algorithm mld /data/hangzhou.osrm --max-table-size 10000

**Remember to update the `osrm_port` in `default_gui.json` accordingly if you make this change.**

Test request:

    curl "http://localhost:5000/route/v1/driving/120.1760299,30.2668853;120.1759848,30.2655152?steps=true"

To get the server started a second time when you pick up your work, only the last container needs to be started again. If you named it (with the flag --name osrm), you can run:

    docker start osrm

Further useful links:

- Project website: http://project-osrm.org/
- GitHub Link: https://github.com/Project-OSRM/osrm-backend
- API Documentation: http://project-osrm.org/docs/v5.5.1/api/#general-options
- Docker backend image: https://hub.docker.com/r/osrm/osrm-backend/

### 2. Run the Simulation
execute `main.exe`

### 3. Modify parameters
You can modify parameters by editing the `vehicle_demo_hybrid/default_gui.json`

Specifically, you can adjust:

- "optimization": `true` or `false`. If true, Gurobi will be used to optimize the assignment; otherwise, 
pedestrians will be assigned to the nearest vehicles.
- "parkinglot"
    - "num": Total number of parking lots (20/50/100/200/400).
- "vehicles"
    - "num": Total number of vehicles.
    - "percent_per_type":  List specifying the percentage of three vehicle types (AET, HET, HGT).

## Output
The simulation will generate two CSV files:

- `{}-{}-pedestrians.csv`
- `{}-{}-vehicles.csv`

These files will be saved in the `vehicle_demo_hybrid/output` folder. The `{}` placeholders are replaced with:

The first bracket: the number of parking lots.\
The second bracket: the percentage distribution of the three taxi types (AET:HET:HGT).