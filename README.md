# GRAND Neutrino Observatory - Automated Site Search Tool

## Overview

This tool is a high-performance topographic and radio-physics simulation engine designed to identify viable deployment sites for the **GRAND (Giant Radio Array for Neutrino Detection)** project.

Because identifying suitable mountain slopes for arrays of 10,000+ antennas requires evaluating billions of pixels, this script utilizes **out-of-core memory mapping** (to prevent RAM exhaustion) and **Numba JIT compilation** (for C-speed parallel ray-tracing). It evaluates geographic coordinates against physical slope constraints, Line-of-Sight (LoS) radio propagation physics, dynamic exclusion zones (RFI), and logistical constraints.

---

## 1. Requirements

The script is built for **Python 3.8+**. Due to the heavy reliance on C-compiled math and geospatial array processing, using a virtual environment (like Conda) is highly recommended.

### Core Dependencies:

* `numpy` (Core matrix math)
* `scipy` (Morphological image processing and connected-component labeling)
* `numba` (JIT compiler for parallelized physics kernels)
* `tifffile` (Reading/Writing large geospatial TIFFs)
* `matplotlib` (Generating visualization maps and extracting KML contours)
* `joblib` (CPU multi-processing orchestration)
* `tqdm` (Progress bars)
* `psutil` (Optional, used for printing system RAM diagnostics)

**Installation via pip:**

```bash
pip install numpy scipy numba tifffile matplotlib joblib tqdm psutil

```

### Automated Conda Environment Generation:

Instead of installing packages manually, you can use the included `generate_env.py` script. This tool parses the main script using Python's Abstract Syntax Tree (AST) to securely identify all required third-party dependencies, checks what is missing from your active environment, and generates a clean, `conda-forge` prioritized `environment.yml` file.

**Usage:**

```bash
# Generate the environment.yml file based on the main script's imports
python generate_env.py

# Create the new Conda environment from the generated file
conda env create -f environment.yml

# Activate the environment
conda activate grand_site_search

```

---

## 2. Setup & Data Acquisition

The script requires a **Digital Elevation Model (DEM)** in `.tif` format, optimized for ~30-meter resolution models. We highly recommend using **[OpenTopography](https://opentopography.org/)** to acquire this data (e.g., SRTM or ALOS AW3D30).

### Automated Setup (Recommended)

We provide a `setup.py` script that verifies your environment dependencies, automatically downloads the required DEM files for the primary target regions (Lima and Arequipa), and generates ready-to-use configuration files.

**Step 1: Obtain an OpenTopography API Key**

1. Create a free account at [OpenTopography](https://portal.opentopography.org/myopentopo).
2. Log in and navigate to the **"myOpenTopo"** dashboard.
3. Click on **"Request an API Key"** to generate your unique authorization token.

**Step 2: Run the Setup Script**
Pass your API key to the setup script to begin the automated download and configuration process:

```bash
python setup.py --open_topography_api_key YOUR_API_KEY_HERE

```

This will securely download the `.tif` files into `../input/dem/` and generate the necessary JSON config files in `../config/`.

### Manual Setup

If you are targeting a region other than Lima or Arequipa:

* Download the required regional tiles manually via the OpenTopography web portal.
* **Preparation:** If your target region spans multiple tiles, merge them into a single `.tif` using a GIS tool like QGIS or GDAL (`gdal_merge.py`) before running the script.

---

## 3. Quick-Start Guide

### Example 1: Using the Built-In Config Generator

The easiest way to run the script is using a JSON configuration file. You can automatically generate a pre-filled template using the built-in generator.

```bash
# Generate a template specifically pre-configured for the Arequipa region
python site_searcher.py --generate_config arequipa_config.json --config_preset arequipa

# Run the script using the newly generated configuration file
python site_searcher.py --config_path arequipa_config.json

```

### Example 2: Running entirely via CLI Flags

If you prefer scripting environments (like bash scripts or Makefiles), you can supply all parameters directly to the command line:

```bash
python site_searcher.py \
    --dem_path "my_custom_map.tif" \
    --origin_lat -15.5 \
    --origin_lon -73.1 \
    --target_antennas 5000 \
    --grid_type hex \
    --min_slope_deg 5.0 \
    --max_slope_deg 20.0 \
    --generate_kml

```

### Example 3: Resuming a Failed Run

If a run crashes halfway through (e.g., your laptop runs out of battery during Step 5), you can instantly skip the expensive ray-tracing step by pointing the script to the failed run's directory.

```bash
python site_searcher.py --config_path my_config.json --resume --resume_dir ../output/20260227_153000

```

### Output Products

By default, all generated output files are saved into a unified run folder located under `../output/`. If you use a JSON config file, the folder is named after the config file. If not, a timestamped folder (e.g., `../output/YYYYMMDD_HHMMSS/`) is automatically generated.

A complete run will produce the following files inside that directory:

* **`log.txt`**: A full transcript of the terminal execution, including settings used, memory usage, and runtime.
* **`*.json`**: A serialized summary containing the exact parameter values used for the run and a detailed breakdown of every valid site found (including area, capacity, and facing direction).
* **`*.png`**: A high-resolution, annotated visualization map displaying the target terrain, overlaid RFI exclusion zones, and color-coded valid array sites. *(Format can be changed to PDF/SVG via parameters).*
* **`*.tif`**: A binary raster mask where `1` represents valid antenna deployment pixels and `0` is excluded terrain.
* **`*.tfw`**: An ESRI World File ensuring the `.tif` mask is properly georeferenced when loaded into GIS software (like QGIS or ArcGIS).
* **`*.kml`**: *(If `--generate_kml` is flagged)* A Google Earth compatible file containing the bounding polygons of all valid sites.

---

## 4. Parameter Configuration Hierarchy

Parameters can be supplied in three ways. The script resolves parameters in the following strict order of priority (1 overrides 2, 2 overrides 3):

1. **Config File (`--config_path`)**: Explicit JSON key-value pairs.
2. **Fallback File**: Automatically looks for `../config/fallbacks.json`. Useful for setting lab-wide defaults.
3. **CLI Arguments**: Standard command-line flags.

### Complete List of Options

#### Required Data Inputs

| Argument | Type | Description |
| --- | --- | --- |
| `--dem_path` | String | Path to the input elevation `.tif` file. |
| `--origin_lat` | Float | Reference origin latitude of the top-left corner of the DEM. |
| `--origin_lon` | Float | Reference origin longitude of the top-left corner of the DEM. |

#### Physical Layout & Geometry

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--target_antennas` | Int | `10000` | Total desired number of antennas across all sub-arrays. |
| `--min_width_km` | Float | `2.0` | Minimum physical width to prune out narrow "tendril" ridges. |
| `--antenna_spacing_km` | Float | `1.0` | Distance between antennas. |
| `--grid_type` | String | `hex` | Pattern for antenna deployment (`hex` or `square`). |
| `--min_sub_array_size` | Int | `500` | Minimum capacity required for a disconnected sub-array to be considered viable. |

#### Topographic & Physics Bounds

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--min_slope_deg` | Float | `3.0` | Minimum allowable terrain steepness. |
| `--max_slope_deg` | Float | `25.0` | Maximum allowable terrain steepness. |
| `--min_dist_km` | Float | `10.0` | Minimum required distance to the target interaction mountain. |
| `--max_dist_km` | Float | `80.0` | Maximum allowable distance to the target interaction mountain. |
| `--min_altitude` | Float | `None` | Minimum absolute altitude of the site in meters. |
| `--max_altitude` | Float | `None` | Maximum absolute altitude of the site in meters. |
| `--min_aspect_deg` | Float | `None` | Restricts sites to specific facing directions (0-360). |
| `--max_aspect_deg` | Float | `None` | Restricts sites to specific facing directions (0-360). |
| `--fresnel_buffer` | Float | `200.0` | Vertical clearance margin (meters) added to the ray to prevent intermediate terrain scattering. |

#### Logistics & Exclusions

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--rfi_zones` | String | `'none'` | Radio exclusion zones. Accepts presets (`lima`, `arequipa`) or a JSON string/list of custom polygons/circles. |
| `--road_map_path` | String | `None` | Path to a secondary aligned `.tif` mapping distances to roads. |
| `--max_road_dist_km` | Float | `20.0` | If road map is provided, maximum allowed distance from a road. |

#### Compute & System Management

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--num_cores` | Int | `-1` | Number of CPU threads to allocate for ray-tracing. `-1` uses all available. |
| `--tile_size` | Int | `2048` | Dimension of square chunks loaded into RAM during operations. Reduce if experiencing memory crashes. |
| `--downsample_factor` | Int | `4` | Coarsening factor applied before final capacity counting to exponentially speed up labeling. |
| `--resume` | Flag | `False` | Triggers checkpoint loading. Bypasses ray-tracing if previous buffers exist. |
| `--resume_dir` | String | `None` | Explicit path to a failed run directory. Defaults to the current output dir if not provided. |

#### Output & Metadata

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--search_mode` | String | `distributed` | `single` forces one monolithic array. `distributed` allows combined sub-arrays. |
| `--region_name` | String | `None` | Cosmetic name printed on the final visualization map. |
| `--generate_kml` | Flag | `False` | Compiles bounding contours into a Google Earth `.kml` file. |
| `--output_directory_base_with_given_json` | String | `../output/` | Directory to store the unified run folders (containing logs, maps, and json). |
| `--output_image_format` | String | `png` | Format of the visual map (`png`, `pdf`, `svg`). |

---

## 5. Internal Workings: The 6-Step Pipeline

The script processes terrain logically through six distinct architectural phases.

### Step 1: Disk Setup & Memory Management

To handle massive DEM files (which can easily exceed 20GB of RAM if loaded natively), the script instantly converts the input `.tif` into a Numpy `.npy` file. It then uses `np.lib.format.open_memmap` to establish "Ping-Pong" buffers (`buffer_A.npy`, `buffer_B.npy`) on the hard drive. All subsequent operations read and write to the disk in chunks, allowing the script to run seamlessly on standard laptops.

### Step 2: Topographic Screening

The code steps through the DEM in defined RAM chunks (configured by `--tile_size`).

1. Uses `np.gradient` to establish raw `dy` and `dx` vectors.
2. Derives physical `slope` and `aspect` angles using trig arrays.
3. Filters the terrain by the bounds (`min_slope`, `altitude`, `aspect`, etc.).
4. Evaluates geographic spatial logic. Longitude degrees are scaled dynamically using `math.cos(lat) * 111.32` to ensure RFI exclusion circles remain perfectly circular on the map projection.
5. Surviving pixels are downsampled (default 5x) and passed forward as raw candidate coordinates.

### Step 3: Physics Simulation (Ray Tracing)

This is the most computationally expensive step. It distributes the candidate pixels across the user's CPU cores using `joblib`.

* **Numba JIT**: The core loop (`check_physics_chunk`) is compiled to native C-speed.
* **The Ray-Cast**: For each candidate pixel, a geometric ray is cast outward in its "aspect" direction up to the `max_dist_km`.
* **Earth Curvature**: The script applies real-world geometry. The target mountain's apparent height is reduced based on distance ($d^2 / 2R$) using an effective Earth radius of 8,500 km.
* **Fresnel Margin**: The target must exceed the detector's altitude + a 1km required interaction depth + the `fresnel_buffer` clearance margin to successfully count as a hit.

### Step 4: Spatial Pruning

A single pixel that sees a mountain is useless if a truck cannot deploy an antenna there. The script uses SciPy's morphological kernels (`binary_closing`, `binary_opening`) on the massive memory maps.

* **Closing:** Fills in small, unviable gaps (potholes) in otherwise good slopes.
* **Opening:** Erases narrow, thin ridge lines (tendrils) that do not meet the `min_width_km` requirement.

### Step 5: Capacity Analysis

The script isolates disconnected sub-arrays using `scipy.ndimage.label`.
Instead of estimating area, it invokes `count_grid_capacity` which physically simulates dropping bounding boxes in either a staggered `hex` pattern or a strict `square` grid. Sites that cannot fit the `min_sub_array_size` are discarded.

### Step 6: Output Generation

Everything is exported into a unified, dynamically generated, timestamped directory.

* **GeoTiff:** The final binary mask is saved alongside a `.tfw` (World File), allowing direct drag-and-drop into QGIS/ArcGIS.
* **KML:** Bounding polygons are extracted using Matplotlib's contour tool and formatted as yellow overlays for Google Earth.
* **JSON:** A complete, serialized summary of all parameters used and the specific capacities/areas of every array found.