import argparse
import sys
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as MplPolygon
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
from scipy.ndimage import binary_closing, binary_opening, label, sum as ndi_sum, find_objects
import os
import shutil
import tempfile
import math
import time
import json
import xml.etree.ElementTree as ET
from datetime import datetime

# Try to import psutil for RAM stats
try:
    import psutil
except ImportError:
    psutil = None

# ==========================================
#               CONFIGURATION
# ==========================================
# Pre-defined Radio Frequency Interference (RFI) exclusion zones
AREQUIPA_RFI_ZONES = [
    ('circle', -16.409, -71.537, 25.0, "Arequipa"),
    ('circle', -16.264, -71.956, 10.0, "Majes"),
    ('circle', -16.533, -71.658, 15.0, "Cerro Verde"),
    ('circle', -16.480, -71.930, 8.0, "La Joya"),
    ('circle', -17.015, -72.015, 10.0, "Mollendo"),
]
LIMA_RFI_ZONES = [
    # 1. Metropolitan Area (Consolidated, Coastal)
    ('circle', -12.080, -77.010, 40.0, "Greater Lima Urban Area"),
    # 2. Inland Valleys (Rímac/Sierra Hubs)
    ('circle', -11.950, -76.680, 8.0, "Chosica"),
    ('circle', -11.850, -76.360, 6.0, "Matucana"),
    ('circle', -11.750, -76.220, 5.0, "San Mateo"),
    ('circle', -11.470, -76.630, 7.0, "Canta"),
    # 3. Provincial Coastal Hubs (North/South)
    ('circle', -11.100, -77.600, 12.0, "Huacho"),
    ('circle', -13.060, -76.380, 10.0, "Cañete"),
    ('circle', -11.080, -77.560, 6.0, "Huaura / Carquin"),
    ('circle', -10.750, -77.750, 7.0, "Barranca"),
    ('circle', -12.480, -76.650, 8.0, "Chilca"),
    ('circle', -11.480, -77.200, 8.0, "Huaral"),
    ('circle', -12.670, -76.620, 5.0, "Mala"),
    ('circle', -13.000, -76.350, 4.0, "Asia"),
    ('circle', -11.560, -77.270, 6.0, "Chancay"),
]

ORIGIN_LAT_AREQUIPA = -14.555380967667489
ORIGIN_LON_AREQUIPA = -73.58612537384033

ORIGIN_LAT_LIMA = -10.228479499469358
ORIGIN_LON_LIMA = -78.07665824890137

# ==========================================
#           NUMBA & PHYSICS KERNELS
# ==========================================
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func): return func
        return decorator

@jit(nopython=True, nogil=True, fastmath=True)
def check_physics_chunk(candidate_subset, elevation, cell_size, rows, cols, fresnel_buffer, min_dist_km, max_dist_km):
    """
    Core Physics Engine: Simulates Line-of-Sight (LoS) from detector pixels to target mountains.
    
    This function utilizes Numba for C-level execution speeds. It casts a geometric ray outward 
    from each candidate pixel in the direction it faces (aspect). It calculates if a target 
    mountain interrupts the ray within the required distance limits, actively accounting for 
    the curvature of the Earth and a Fresnel zone clearance buffer.
    
    Parameters:
    - candidate_subset (ndarray): An Nx3 array of [row, col, aspect_degrees] for candidate pixels.
    - elevation (ndarray): The full 2D array of the Digital Elevation Model (DEM).
    - cell_size (float): The physical size of one pixel side in meters.
    - rows, cols (int): Dimensions of the elevation array.
    - fresnel_buffer (float): Altitude buffer in meters added to account for radio wave scattering.
    - min_dist_km, max_dist_km (float): The valid distance bounds to find a target mountain.
    
    Returns:
    - tuple(list, list): Two lists containing the row and column indices of successful candidate pixels.
    """
    hits_r = []
    hits_c = []
    
    # Convert physical limits to pixel indices based on map resolution
    start_dist_px = int((min_dist_km * 1000) / cell_size) 
    end_dist_px   = int((max_dist_km * 1000) / cell_size) 
    step_px       = int(1000 / cell_size)
    
    # Precompute Earth Curvature coefficient: 1 / (2 * Earth_Radius_in_meters)
    # Using a standard 8500km effective radius often used in radio propagation models
    inv_2R = 1.0 / (2 * 8500000.0) 

    n = candidate_subset.shape[0]
    for i in range(n):
        r = int(candidate_subset[i, 0])
        c = int(candidate_subset[i, 1])
        aspect_val = candidate_subset[i, 2]
        
        # Calculate ray directional vectors based on pixel aspect
        look_rad = np.radians(aspect_val)
        sin_look = np.sin(look_rad)
        cos_look = np.cos(look_rad)
        my_elev = elevation[r, c]
        has_target = False
        
        # Ray casting loop: step outward from the pixel within the specified distance bounds
        for dist_px in range(start_dist_px, end_dist_px, step_px):
            dist_m = dist_px * cell_size
            dx = int(dist_px * sin_look)
            dy = int(dist_px * cos_look)
            target_r = r - dy # Subtract dy because image y-axis points downwards
            target_c = c + dx
            
            # Ensure target ray index remains within DEM boundaries
            if target_r >= 0 and target_r < rows and target_c >= 0 and target_c < cols:
                target_elev = elevation[target_r, target_c]
                if np.isnan(target_elev): continue
                
                # Apply Earth Curvature drop calculation: drop = d^2 / 2R
                curvature_drop = (dist_m ** 2) * inv_2R
                apparent_height = target_elev - curvature_drop
                
                # Condition: Target must be taller than detector + required 1km interaction depth + Fresnel buffer
                if apparent_height > (my_elev + 1000 + fresnel_buffer):
                    has_target = True
                    break
                    
        # If the ray successfully hit a valid target mountain, record the pixel
        if has_target:
            hits_r.append(r)
            hits_c.append(c)
            
    return hits_r, hits_c

@jit(nopython=True, fastmath=True)
def is_point_in_poly(x, y, poly_verts):
    """
    Determines if a given 2D point lies inside a polygon using the Ray-Casting algorithm.
    Optimized for Numba execution.
    
    Parameters:
    - x, y (float): Coordinates of the test point.
    - poly_verts (ndarray): A list of (x, y) coordinates defining the polygon vertices.
    
    Returns:
    - bool: True if the point is inside the polygon, False otherwise.
    """
    n = len(poly_verts)
    inside = False
    p1x, p1y = poly_verts[0]
    
    for i in range(n + 1):
        p2x, p2y = poly_verts[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
        
    return inside

@jit(nopython=True, parallel=True)
def apply_poly_mask_numba(valid_rows, valid_cols, poly_verts, mask_out):
    """
    Parallelized application of the polygon ray-casting check across an array of coordinates.
    Used for excluding regions defined by arbitrary polygonal RFI zones.
    
    Parameters:
    - valid_rows, valid_cols (ndarray): Arrays containing the row and col coordinates to check.
    - poly_verts (ndarray): Polygon vertices.
    - mask_out (ndarray): A boolean array modified in-place. Sets to False if point is inside polygon.
    """
    n = len(valid_rows)
    for i in prange(n):
        if is_point_in_poly(valid_cols[i], valid_rows[i], poly_verts):
            mask_out[i] = False

@jit(nopython=True, fastmath=True)
def count_grid_capacity(mask_chunk, spacing_px, grid_type_code):
    """
    Simulates the placement of physical antennas on the validated terrain map to count 
    total array capacity based on specific geometry.
    
    Parameters:
    - mask_chunk (ndarray): A 2D boolean array where True indicates valid terrain.
    - spacing_px (int): The distance between antennas in pixels.
    - grid_type_code (int): 0 for 'square' grid, 1 for 'hexagonal' grid.
    
    Returns:
    - int: The total number of antennas that successfully fit inside the valid terrain.
    """
    h, w = mask_chunk.shape
    count = 0
    if grid_type_code == 0: # Square Grid layout
        for r in range(0, h, spacing_px):
            for c in range(0, w, spacing_px):
                if mask_chunk[r, c]:
                    count += 1
    elif grid_type_code == 1: # Hexagonal Grid layout
        # Vertical spacing for hex is sin(60 deg) * spacing
        v_step = int(spacing_px * 0.866025)
        if v_step < 1: v_step = 1
        row_idx = 0
        for r in range(0, h, v_step):
            # Stagger every other row by half the spacing distance
            offset = (spacing_px // 2) if (row_idx % 2 == 1) else 0
            for c in range(offset, w, spacing_px):
                if mask_chunk[r, c]:
                    count += 1
            row_idx += 1
    return count

# ==========================================
#           HELPERS
# ==========================================

def get_candidates_chunked(elevation, cell_size, rfi_zones, origin_lat, origin_lon, 
                           min_alt=None, max_alt=None, min_aspect_deg=None, max_aspect_deg=None, 
                           road_map_path=None, max_road_dist_km=None, 
                           tile_size=2048):
    """
    Memory-efficient topographic screening. Iterates over the large DEM in chunks (tiles) 
    to find pixels that meet the primary geometrical criteria (slope, aspect, altitude) 
    and logistics constraints (RFI distance, road distance) prior to running ray-tracing.
    
    Parameters:
    - elevation (ndarray): Full DEM array (usually memory-mapped).
    - cell_size (float): Physical pixel size in meters.
    - rfi_zones (list): List of configured exclusion zones (circles/polygons).
    - origin_lat, origin_lon (float): Reference coordinates for converting km to pixels.
    - min_alt, max_alt (float): Elevation restrictions.
    - min_aspect_deg, max_aspect_deg (float): Required facing directions for slopes.
    - road_map_path (str): Path to an aligned TIFF containing distance-to-road values.
    - max_road_dist_km (float): Maximum allowed distance from a road.
    - tile_size (int): Size of the square chunk to process in RAM at one time.
    
    Returns:
    - ndarray: Nx3 array of valid candidate pixels formatted as [row, col, aspect_degrees].
    """
    rows, cols = elevation.shape
    candidates_list = []
    
    # Rough approximations for translating lat/lon to pixels locally
    deg_per_px_lat = (cell_size / 1000.0) / 110.6
    deg_per_px_lon = (cell_size / 1000.0) / 107.0
    
    # Load Logistics Road map if provided
    road_dist_map = None
    if road_map_path and max_road_dist_km:
        if os.path.exists(road_map_path):
            try:
                road_dist_map = tiff.imread(road_map_path, out='memmap')
                print(f"   -> Logistics: Loaded Road Distance Map ({road_map_path})")
            except:
                print(f"   -> WARNING: Could not load road map.")
        else:
            print(f"   -> WARNING: Road map file not found.")

    # Convert geographic RFI definitions into pixel coordinates for local checking
    rfi_circles = [] 
    rfi_polys = []   
    if rfi_zones:
        for item in rfi_zones:
            type_tag = item[0]
            if type_tag == 'circle':
                _, zlat, zlon, zrad_km, _ = item 
                z_r = (origin_lat - zlat) / deg_per_px_lat
                z_c = (zlon - origin_lon) / deg_per_px_lon
                z_rad_px = int((zrad_km * 1000) / cell_size)
                rfi_circles.append((z_r, z_c, z_rad_px**2)) # Store radius squared for faster math
            elif type_tag == 'poly':
                _, coords, _ = item 
                pixel_verts = []
                for (plat, plon) in coords:
                    pr = (origin_lat - plat) / deg_per_px_lat
                    pc = (plon - origin_lon) / deg_per_px_lon
                    pixel_verts.append((pc, pr)) 
                rfi_polys.append(np.array(pixel_verts, dtype=np.float64))

    r_steps = range(0, rows, tile_size)
    c_steps = range(0, cols, tile_size)
    
    # Process the map in chunks to avoid blowing out system RAM
    with tqdm(total=len(r_steps)*len(c_steps), desc="   Scanning Topography", unit="tile") as pbar:
        for r in r_steps:
            for c in c_steps:
                r_end = min(r + tile_size, rows)
                c_end = min(c + tile_size, cols)
                chunk = elevation[r:r_end, c:c_end]
                
                # Calculate topographic gradient (change in Z)
                dy, dx = np.gradient(chunk, cell_size)
                
                # Derive Slope (steepness) and Aspect (direction)
                slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
                aspect = np.degrees(np.arctan2(-dx, dy)) % 360
                
                # Filter 1: Fundamental detector slope requirement (3-25 degrees)
                mask = (slope >= 3) & (slope <= 25)
                
                # Filter 2: Altitude bounds
                if min_alt is not None: mask &= (chunk >= min_alt)
                if max_alt is not None: mask &= (chunk <= max_alt)
                
                # Filter 3: Aspect bounds (handle wrapping around 360 degrees)
                if min_aspect_deg is not None and max_aspect_deg is not None:
                    min_a, max_a = min_aspect_deg, max_aspect_deg
                    if min_a > max_a:
                        mask &= (aspect >= min_a) | (aspect <= max_a) 
                    else:
                        mask &= (aspect >= min_a) & (aspect <= max_a) 
                
                # Filter 4: Road Logistics
                if road_dist_map is not None:
                    road_chunk = road_dist_map[r:r_end, c:c_end]
                    mask &= (road_chunk <= (max_road_dist_km * 1000))

                # Filter 5: Dynamic Exclusion Zones (RFI)
                if rfi_zones:
                    valid_y, valid_x = np.where(mask)
                    if len(valid_y) > 0:
                        abs_r = r + valid_y
                        abs_c = c + valid_x
                        
                        # Process Circular exclusion zones
                        for (zr, zc, zrad_sq) in rfi_circles:
                            dist_sq = (abs_r - zr)**2 + (abs_c - zc)**2
                            mask[valid_y[np.where(dist_sq < zrad_sq)], valid_x[np.where(dist_sq < zrad_sq)]] = False
                        
                        valid_idx = np.where(mask.ravel())[0]
                        if len(valid_idx) > 0:
                            subset_mask = np.ones(len(valid_y), dtype=bool)
                            # Process Polygonal exclusion zones
                            for poly in rfi_polys:
                                apply_poly_mask_numba(abs_r.astype(np.float64), abs_c.astype(np.float64), poly, subset_mask)
                            bad_idx = np.where(~subset_mask)[0]
                            mask[valid_y[bad_idx], valid_x[bad_idx]] = False

                # Extract surviving pixels for the physics simulation step
                cr, cc = np.where(mask)
                if len(cr) > 0:
                    chunk_cands = np.column_stack((cr + r, cc + c, aspect[cr, cc]))
                    # Down-sample candidates 5x to speed up ray tracing; assumption is terrain is continuous
                    candidates_list.append(chunk_cands[::5]) 
                    pbar.set_postfix(candidates=f"{len(chunk_cands[::5]):,}")
                pbar.update(1)

    if not candidates_list: return np.zeros((0, 3))
    return np.vstack(candidates_list)

def apply_morphology_pingpong(source_path, dest_path, shape, dtype, operation_func, structure, desc="Processing", tile_size=2048):
    """
    Applies image morphology operations (closing/opening) on a massive memory-mapped array 
    without loading the whole array into RAM. It reads from one file and writes to another ("ping-pong").
    Used to prune out "tendrils" (narrow, unusable ridges) and fill in small gaps in the detected sites.
    
    Parameters:
    - source_path, dest_path (str): File paths to the input and output .npy memmap files.
    - shape (tuple): Dimensions of the array.
    - dtype (type): Data type of the array elements (usually bool).
    - operation_func (function): The SciPy morphology function to apply (e.g., binary_closing).
    - structure (ndarray): The structuring element (kernel) dictating the pixel radius of the operation.
    """
    source = np.lib.format.open_memmap(source_path, mode='r')
    dest = np.lib.format.open_memmap(dest_path, mode='r+', shape=shape, dtype=dtype)
    rows, cols = shape
    
    with tqdm(total=(rows//tile_size + 1)*(cols//tile_size + 1), desc=f"   {desc}", unit="tile") as pbar:
        # Pad the chunk by half the structure size to prevent edge artifacts between chunks
        pad = max(structure.shape) // 2
        for r in range(0, rows, tile_size):
            for c in range(0, cols, tile_size):
                r_end = min(r + tile_size, rows)
                c_end = min(c + tile_size, cols)
                r_start = max(0, r - pad)
                c_start = max(0, c - pad)
                
                chunk = source[r_start:min(rows, r_end+pad), c_start:min(cols, c_end+pad)]
                processed = operation_func(chunk, structure=structure)
                
                loc_r_start = r - r_start
                loc_c_start = c - c_start
                # Write back only the non-padded, processed core of the chunk
                dest[r:r_end, c:c_end] = processed[loc_r_start:loc_r_start + (r_end - r), loc_c_start:loc_c_start + (c_end - c)]
                pbar.update(1)
    dest.flush()

def create_world_file(tif_filename, top_left_lat, top_left_lon, cell_size_deg):
    """
    Creates an ESRI World File (.tfw) which accompanies a standard TIFF image, 
    allowing GIS software (like QGIS or ArcGIS) to project it correctly on a map.
    """
    tfw_name = os.path.splitext(tif_filename)[0] + ".tfw"
    try:
        with open(tfw_name, "w") as f:
            # Format: Pixel X size, Rotation, Rotation, Negative Pixel Y size, Top-Left X, Top-Left Y
            f.write(f"{cell_size_deg:.10f}\n0.0\n0.0\n-{cell_size_deg:.10f}\n{top_left_lon:.10f}\n{top_left_lat:.10f}\n") 
    except: pass

def generate_kml_file(mask, elevation, filename, origin_lat, origin_lon, cell_size_deg, downsample=1):
    """
    Generates a Google Earth compatible KML file representing the valid site polygons.
    It extracts polygon contours from the binary mask using Matplotlib's contour tool.
    
    Parameters:
    - mask (ndarray): Binary mask indicating valid deployment sites.
    - filename (str): Output path for the KML file.
    - origin_lat, origin_lon, cell_size_deg: Used to convert array pixel indices to GPS coordinates.
    """
    print(f"   -> Generating KML: {filename} ...")
    
    try:
        root = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
        doc = ET.SubElement(root, "Document")
        
        # Define visual styles for the KML polygons (Yellow, semi-transparent)
        style = ET.SubElement(doc, "Style", id="grand_site")
        lstyle = ET.SubElement(style, "LineStyle")
        ET.SubElement(lstyle, "color").text = "ffffff00" 
        ET.SubElement(lstyle, "width").text = "3"
        pstyle = ET.SubElement(style, "PolyStyle")
        ET.SubElement(pstyle, "color").text = "40ffff00" 
        
        # Use Matplotlib to trace the boundaries of the mask areas
        fig = plt.figure()
        ax = fig.add_subplot(111)
        contours = ax.contour(mask, levels=[0.5])
        plt.close(fig)
        
        # Iterate over traced contours to build KML geometry blocks
        site_idx = 1
        for path in contours.get_paths():
            placemark = ET.SubElement(doc, "Placemark")
            ET.SubElement(placemark, "name").text = f"GRAND Site {site_idx}"
            ET.SubElement(placemark, "styleUrl").text = "#grand_site"
            
            poly = ET.SubElement(placemark, "Polygon")
            outer = ET.SubElement(poly, "outerBoundaryIs")
            ring = ET.SubElement(outer, "LinearRing")
            coords_str = ""
            
            # Map Matplotlib vertices back to real-world Long/Lat coordinates
            for (c, r) in path.vertices:
                r_full = r * downsample
                c_full = c * downsample
                lat = origin_lat - (r_full * cell_size_deg)
                lon = origin_lon + (c_full * cell_size_deg)
                coords_str += f"{lon},{lat},0 "
            
            ET.SubElement(ring, "coordinates").text = coords_str
            site_idx += 1
                
        ET.ElementTree(root).write(filename, encoding='UTF-8', xml_declaration=True)
        print(f"   -> KML saved to '{filename}'")
    except Exception as e:
        print(f"   -> WARNING: KML generation failed (Skipping KML). Error: {e}")

def print_tool_explanation():
    """Outputs a formatted explanation of the tool's capabilities and logic to the console."""
    print("""
================================================================================
GRAND NEUTRINO OBSERVATORY - AUTOMATED SITE SEARCH TOOL
================================================================================
This tool performs a high-performance topographic and physics simulation to 
identify suitable deployment sites for the GRAND array.

Core Workflow:
1. Topographic Filtering: Scans the DEM for terrain with suitable slopes (3-25 degrees),
   enforcing altitude limits and specified facing directions (Aspect).
2. Logistics & RFI: Masks out areas overlapping populated centers and, optionally,
   areas situated too far from road infrastructure.
3. Ray-Tracing (Physics): Simulates line-of-sight from candidates to target 
   mountain ranges. It actively accounts for Earth's curvature and maintains a 
   clearance buffer over intermediate terrain.
4. Spatial Pruning: Implements morphological math (Closing/Opening) to remove 
   isolated "tendril" ridges that are unsuitable for wide array deployments.
5. Grid Packing: Simulates placing antennas in 'hex' or 'square' grids to calculate 
   the true physical capacity of the resulting sites.
   
Customizable Constraints & Processing Parameters:
- RFI Zones: Accept pre-defined sets ('lima', 'arequipa') or custom geometry lists via JSON config.
- Fresnel Buffer: Adds a vertical clearance margin (in meters) to the line-of-sight ray.
- Downsample Factor: Modifies the internal resolution of the capacity masking, speeding up processing.
- Output Paths: Export custom image paths and formats (png, pdf, svg) directly via CLI or config.
   
Output:
The script outputs georeferenced TIFFs, a KML file for Google Earth, an annotated 
PNG (or preferred format) map, and a JSON summary of all viable sites found.
================================================================================
    """)

def validate_parameters(params):
    """
    Pre-flight validation checks to enforce 'Fail Fast' mechanisms. 
    Verifies the existence of critical files and the physical logic of search bounds 
    before engaging the memory-heavy processing loops.
    """
    errors = []
    
    # 1. Check DEM path existence
    if not os.path.exists(params['dem_path']):
        errors.append(f"DEM file not found: {params['dem_path']}")
    
    # 2. Check physical layout impossibilities
    if params['min_width_km'] <= 0:
        errors.append("min_width_km must be strictly positive (> 0).")
        
    if params['target_antennas'] <= 0:
        errors.append("target_antennas must be strictly positive (> 0).")
        
    if params['antenna_spacing_km'] <= 0:
        errors.append("antenna_spacing_km must be strictly positive (> 0).")
        
    # 3. Verify Road Map existence if specified
    if params['road_map_path'] is not None and not os.path.exists(params['road_map_path']):
        errors.append(f"Road map file not found: {params['road_map_path']}")
        
    # 4. Verify Ray-Tracing bounds logic
    if params['min_dist_km'] >= params['max_dist_km']:
        errors.append("min_dist_km must be strictly less than max_dist_km.")
        
    # 5. Verify Altitude bounds logic
    if params['min_altitude'] is not None and params['max_altitude'] is not None:
        if params['min_altitude'] >= params['max_altitude']:
            errors.append("min_altitude must be strictly less than max_altitude.")

    # Execute Fail-Fast
    if errors:
        print("\n================================================================================")
        print("PRE-FLIGHT VALIDATION FAILED:")
        for error in errors:
            print(f"  - {error}")
        print("Please correct the parameters in your config file or CLI and try again.")
        print("================================================================================\n")
        sys.exit(1)

# ==========================================
#             MAIN EXECUTION
# ==========================================
def find_grand_regions_interactive(dem_path, cell_size=30, target_antennas=1000, 
                            rfi_zones=None, origin_lat=-15.0, origin_lon=-73.0,
                            min_width_km=2.0, min_altitude=None, max_altitude=None,
                            antenna_spacing_km=1.0, min_dist_km=30.0, max_dist_km=80.0,
                            road_map_path=None, max_road_dist_km=None,
                            grid_type='square', generate_kml=False,
                            search_mode='single', min_sub_array_size=100,
                            min_aspect_deg=None, max_aspect_deg=None,
                            region_name=None, fresnel_buffer=200.0, 
                            downsample_factor=4, output_image_path=None, 
                            output_image_format='png'):
    """
    The orchestrator function. Manages memory, coordinates chunking algorithms, runs 
    the ray-tracer, performs physical capacity counting, and manages output generation.
    """
    
    # Cast safety to ensure slice logic doesn't fail if passed as float via JSON
    downsample_factor = int(downsample_factor)
    
    print(f"\n=============================================")
    print(f"   GRAND SITE SEARCH: RUN PARAMETERS")
    print(f"=============================================")
    print(f"   -> DEM File: {dem_path}")
    print(f"   -> Origin: {origin_lat}, {origin_lon}")
    print(f"   -> Target: {target_antennas} antennas")
    print(f"   -> Spacing: {antenna_spacing_km} km ({grid_type} grid)")
    print(f"   -> Min Width: {min_width_km} km")
    print(f"   -> Target Dist: {min_dist_km} - {max_dist_km} km")
    print(f"   -> Physics: Fresnel Buffer {fresnel_buffer}m | Downsample Factor {downsample_factor}")
    if road_map_path:
        print(f"   -> Logistics: Require road within {max_road_dist_km} km")
    if min_altitude or max_altitude:
        min_s = f"{min_altitude}m" if min_altitude else "0m"
        max_s = f"{max_altitude}m" if max_altitude else "Inf"
        print(f"   -> Altitude: {min_s} < h < {max_s}")
    if min_aspect_deg is not None and max_aspect_deg is not None:
        print(f"   -> Aspect Range: {min_aspect_deg}° to {max_aspect_deg}°") 
    print(f"   -> RFI Zones: {len(rfi_zones) if rfi_zones else 0} active (Numba Optimized)")

    print(f"\n=============================================")
    print(f"   SYSTEM & RESOURCE REPORT")
    print(f"=============================================")
    num_cores = multiprocessing.cpu_count()
    print(f"   -> CPU Cores: {num_cores}")
    print(f"   -> Numba JIT: {'ENABLED' if HAS_NUMBA else 'DISABLED'}")
    if psutil:
        mem = psutil.virtual_memory()
        print(f"   -> System RAM: {mem.total/1024**3:.1f} GB (Free: {mem.available/1024**3:.1f} GB)")
    
    # Create a temporary directory for memory-mapped operations on large DEM files
    temp_dir = tempfile.mkdtemp()
    print(f"   -> Temp Dir: {temp_dir}")
    print(f"=============================================\n")

    t_start_total = time.time()
    site_details = []
    generated_files = []
    count = 0

    try:
        # Step 1: Disk Setup. Convert TIF to memory-mapped NPY for rapid random access.
        print("[1/6] Loading Map Data...")
        t0 = time.time()
        npy_path = dem_path.replace(".tif", ".npy")
        if not os.path.exists(npy_path):
            temp = tiff.imread(dem_path).astype(np.float32)
            temp[temp < -100] = np.nan # Nullify ocean/void areas
            np.save(npy_path, temp)
            del temp
        elevation = np.load(npy_path, mmap_mode='r')
        rows, cols = elevation.shape
        print(f"      Map: {rows} x {cols} pixels")
        
        est_disk_gb = (rows * cols * 2) / (1024**3) 
        print(f"      Estimated Temp Disk Usage: ~{est_disk_gb:.2f} GB")
        print(f"      Time: {time.time()-t0:.2f}s")
        
        path_A = os.path.join(temp_dir, "buffer_A.npy")
        path_B = os.path.join(temp_dir, "buffer_B.npy")
        buf_a = np.lib.format.open_memmap(path_A, mode='w+', shape=(rows, cols), dtype=bool)
        buf_b = np.lib.format.open_memmap(path_B, mode='w+', shape=(rows, cols), dtype=bool)
        del buf_b

        # Step 2: Basic Geometric and Geographic constraints filtering
        print("\n[2/6] Identifying Candidates...")
        t0 = time.time()
        candidates_arr = get_candidates_chunked(
            elevation, cell_size, rfi_zones, origin_lat, origin_lon, 
            min_alt=min_altitude, max_alt=max_altitude,
            road_map_path=road_map_path, max_road_dist_km=max_road_dist_km,
            min_aspect_deg=min_aspect_deg, max_aspect_deg=max_aspect_deg
        )
        total = candidates_arr.shape[0]
        print(f"      Time: {time.time()-t0:.2f}s")
        
        if total == 0: 
            print("No candidates found.")
            return

        # Step 3: Expensive Physics computation parallelized across CPU cores
        print(f"\n[3/6] Ray Tracing ({total} candidates)...")
        t0 = time.time()
        batches = np.array_split(candidates_arr, num_cores * 4)
        results = Parallel(n_jobs=-1)(
            delayed(check_physics_chunk)(batch, elevation, cell_size, rows, cols, fresnel_buffer, min_dist_km, max_dist_km) 
            for batch in tqdm(batches, desc="   Simulating", unit="batch")
        )
        # Reconstruct the boolean mask from returned ray-cast coordinates
        for r_list, c_list in results:
            buf_a[r_list, c_list] = True
        buf_a.flush()
        del buf_a
        print(f"      Time: {time.time()-t0:.2f}s")

        # Step 4: Prune spatial artifacts to ensure solid, block-like arrays
        print("\n[4/6] Cleaning Shapes...")
        t0 = time.time()
        close_r = int(antenna_spacing_km * 1000 / cell_size)
        tendril_r = int((min_width_km * 0.5 * 1000) / cell_size)
        apply_morphology_pingpong(path_A, path_B, (rows, cols), bool, binary_closing, np.ones((close_r, close_r)), desc="Closing")
        apply_morphology_pingpong(path_B, path_A, (rows, cols), bool, binary_opening, np.ones((tendril_r, tendril_r)), desc="Pruning")
        print(f"      Time: {time.time()-t0:.2f}s")

        # Step 5: Isolate unique sites and measure their capacity mathematically
        print("\n[5/6] Final Analysis...")
        t0 = time.time()
        final_map_disk = np.lib.format.open_memmap(path_A, mode='r')
        small_map = final_map_disk[::downsample_factor, ::downsample_factor]
        labeled, num = label(small_map) # Give unique integer IDs to disconnected array zones
        
        eff_cell = cell_size * downsample_factor
        px_area_km2 = (eff_cell / 1000.0)**2
        
        if search_mode == 'single':
            threshold_antennas = target_antennas
        else:
            threshold_antennas = min_sub_array_size
            
        req_pixels = int((threshold_antennas * antenna_spacing_km**2) / px_area_km2)
        small_final = np.zeros_like(labeled, dtype=np.uint8)
        cumulative_capacity = 0
        
        if num > 0:
            sizes = ndi_sum(small_map, labeled, index=np.arange(1, num+1))
            potential_ids = np.where(sizes >= req_pixels)[0] + 1
            valid_ids_final = []
            
            if len(potential_ids) > 0:
                dy_ds, dx_ds = np.gradient(elevation[::downsample_factor, ::downsample_factor], eff_cell)
                aspect_ds = np.degrees(np.arctan2(-dx_ds, dy_ds)) % 360
                
                spacing_px = int((antenna_spacing_km * 1000) / cell_size)
                grid_code = 1 if grid_type == 'hex' else 0 
                all_slices = find_objects(labeled)
                
                # Iterate through found blobs to calculate physical internal placement of DUs
                for site_id in potential_ids:
                    loc = all_slices[site_id - 1]
                    r_start = loc[0].start * downsample_factor
                    r_stop = loc[0].stop * downsample_factor
                    c_start = loc[1].start * downsample_factor
                    c_stop = loc[1].stop * downsample_factor
                    
                    r_stop = min(r_stop, rows)
                    c_stop = min(c_stop, cols)
                    
                    mask_chunk = final_map_disk[r_start:r_stop, c_start:c_stop]
                    antennas_fit = count_grid_capacity(mask_chunk, spacing_px, grid_code)
                    
                    if antennas_fit >= threshold_antennas:
                        valid_ids_final.append(site_id)
                        site_mask_ds = (labeled == site_id)
                        mean_aspect = np.mean(aspect_ds[site_mask_ds])
                        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                        aspect_str = dirs[round(mean_aspect / 45) % 8]
                        area_km2 = sizes[site_id-1] * px_area_km2
                        
                        site_details.append({
                            "site_id": int(site_id),
                            "area_km2": float(f"{area_km2:.2f}"),
                            "capacity_exact": int(antennas_fit),
                            "grid_type": grid_type,
                            "mean_aspect_deg": float(f"{mean_aspect:.1f}"),
                            "facing_direction": aspect_str
                        })

            site_details.sort(key=lambda x: x['capacity_exact'], reverse=True)
            final_selection_ids = []
            
            if search_mode == 'distributed':
                for site in site_details:
                    final_selection_ids.append(site['site_id'])
                    cumulative_capacity += site['capacity_exact']
                success = cumulative_capacity >= target_antennas
                print(f"   -> Distributed: {len(final_selection_ids)} sites found.")
                print(f"   -> Total Cap: {cumulative_capacity} (Target: {target_antennas})")
            else:
                final_selection_ids = [s['site_id'] for s in site_details]
                success = len(final_selection_ids) > 0
                print(f"   -> Single: {len(final_selection_ids)} valid sites found.")

            if len(final_selection_ids) > 0:
                labeled_viz = np.zeros_like(labeled, dtype=np.uint8)
                current_viz_id = 1
                for original_id in final_selection_ids:
                    labeled_viz[labeled == original_id] = current_viz_id
                    current_viz_id += 1
                small_final = np.isin(labeled, final_selection_ids).astype(np.uint8)
                count = len(final_selection_ids)

        # Step 6: Create Outputs (Images, KML, and JSON parameters file)
        print(f"\n[6/6] Saving & Visualization...")
        t0 = time.time()
        
        # Save TIF
        out_tif = "grand_search_results_"+os.path.splitext(os.path.basename(dem_path))[0]+".tif"
        tiff.imwrite(out_tif, small_final)
        generated_files.append(os.path.abspath(out_tif))
        
        # Save TFW
        new_res_deg = (1.0/3600.0) * downsample_factor
        create_world_file(out_tif, origin_lat, origin_lon, new_res_deg)
        generated_files.append(os.path.abspath(os.path.splitext(out_tif)[0] + ".tfw"))
        
        # Save KML
        if generate_kml:
            kml_name = "grand_search_results_"+os.path.splitext(os.path.basename(dem_path))[0]+".kml"
            generate_kml_file(small_final, elevation, kml_name, origin_lat, origin_lon, new_res_deg)
            generated_files.append(os.path.abspath(kml_name))

        # Save Custom Visualization
        try:
            fig, ax = plt.subplots(figsize=(14, 12))
            viz_ds = downsample_factor * 2 
            elev_viz = elevation[::viz_ds, ::viz_ds]
            mask_viz = small_final[::2, ::2] 
            
            mask_viz_labeled = labeled_viz[::2, ::2]
            
            mr = min(elev_viz.shape[0], mask_viz.shape[0])
            mc = min(elev_viz.shape[1], mask_viz.shape[1])
            elev_viz = elev_viz[:mr, :mc]
            mask_viz_labeled = mask_viz_labeled[:mr, :mc]
            
            im = ax.imshow(elev_viz, cmap='terrain', vmin=0, vmax=6000)
            
            legend_handles = []
            legend_labels = []

            if count > 0:
                cmap = plt.get_cmap('tab10') 
                for i in range(1, count + 1): 
                    color = cmap((i - 1) % 10)
                    ax.contour((mask_viz_labeled == i), levels=[0.5], colors=[color], linewidths=2.5)
                        
                    site_data = site_details[i - 1]
                    label_str = f"Site {site_data['site_id']}: {site_data['capacity_exact']} DUs ({site_data['area_km2']} km²)"
                    legend_handles.append(Line2D([0], [0], color=color, lw=2.5))
                    legend_labels.append(label_str)
            
            if rfi_zones:
                deg_viz = (1.0/3600.0) * viz_ds
                legend_handles.append(Line2D([0], [0], color='red', linestyle='--', lw=2))
                legend_labels.append("RFI exclusion zone")
                for item in rfi_zones:
                    type_tag = item[0]
                    if type_tag == 'circle':
                        _, lat, lon, radius_km, name = item
                        px_x = (lon - origin_lon) / deg_viz
                        px_y = (origin_lat - lat) / deg_viz
                        r_px = (radius_km / 111.0) / deg_viz
                        ax.add_patch(Circle((px_x, px_y), r_px, edgecolor='red', facecolor='none', ls='--', lw=2))
                        text = ax.text(px_x, px_y-r_px/2, name, color='red', fontsize=12, ha='center')
                        text.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'), path_effects.Normal()])
                    elif type_tag == 'poly':
                        _, coords, name = item
                        verts = []
                        for (plat, plon) in coords:
                            px = (plon - origin_lon) / deg_viz
                            py = (origin_lat - plat) / deg_viz
                            verts.append((px, py))
                        ax.add_patch(MplPolygon(verts, closed=True, edgecolor='red', facecolor='none', ls='--', lw=2))
                        cx = sum(p[0] for p in verts)/len(verts)
                        cy = sum(p[1] for p in verts)/len(verts)
                        text = ax.text(cx, cy, name, color='red', fontsize=8, ha='center')
                        text.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'), path_effects.Normal()])

            deg_viz = (1.0/3600.0) * viz_ds
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p: f"{origin_lon + x*deg_viz:.2f}"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y,p: f"{origin_lat - y*deg_viz:.2f}"))
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            cbar = plt.colorbar(im, fraction=0.035, pad=0.04)
            cbar.set_label('Altitude (m)', rotation=270, labelpad=15)
            ax.set_title(f"GRAND site search | {region_name if region_name is not None else ''} {'|' if region_name is not None else ''} {search_mode.title()} mode\nFound {count} sites | Total capacity: {cumulative_capacity if search_mode=='distributed' else 'N/A'} DUs | Grid: {grid_type} | Spacing: {antenna_spacing_km} km | Altitude restriction: {min_altitude}-{max_altitude} m")
            
            fs = 'small' if len(legend_labels) > 8 else 'medium'
            ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=fs, framealpha=0.8)
            
            if output_image_path is not None:
                img_name = output_image_path
            else:
                img_name = "grand_search_results_"+os.path.splitext(os.path.basename(dem_path))[0]+"."+output_image_format.strip('.')
            
            if os.path.dirname(img_name):
                os.makedirs(os.path.dirname(img_name), exist_ok=True)
                
            plt.savefig(img_name, format=output_image_format.strip('.'), dpi=150, bbox_inches='tight')
            generated_files.append(os.path.abspath(img_name))
            print(f"   -> Map saved.")
            print(f"      Time Elapsed: {time.time()-t0:.2f}s")
            
        except Exception as e:
            print(f"Viz Error: {e}")

        print(f"\n=============================================")
        print(f"   RESULTS SUMMARY")
        print(f"=============================================")
        print(f"   -> Sites Found: {count}")
        for site in site_details:
            print(f"      Site {site['site_id']}: {site['area_km2']} km² | Cap: {site['capacity_exact']} ({site['grid_type']}) | Faces: {site['facing_direction']}")
        
        # Save JSON output log
        out_data = {
            "timestamp": datetime.now().isoformat(),
            "mode": search_mode,
            "parameters": {
                "dem": dem_path, "origin": [origin_lat, origin_lon],
                "target": target_antennas, "spacing_km": antenna_spacing_km,
                "min_dist_km": min_dist_km, "max_dist_km": max_dist_km,
                "min_sub_array": min_sub_array_size,
                "grid_type": grid_type, "road_map": road_map_path,
                "fresnel_buffer": fresnel_buffer, "downsample_factor": downsample_factor
            },
            "results": {
                "total_sites": count,
                "total_capacity": cumulative_capacity if search_mode=='distributed' else 'N/A',
                "sites": site_details
            }
        }
        json_name = "grand_search_results_"+os.path.splitext(os.path.basename(dem_path))[0]+".json"
        with open(json_name, "w") as f:
            json.dump(out_data, f, indent=4)
        generated_files.append(os.path.abspath(json_name))
        print(f"   -> JSON saved.")
        
        # Print outputs generated block for log
        print(f"\n=============================================")
        print(f"   OUTPUTS GENERATED")
        print(f"=============================================")
        for fpath in generated_files:
            print(f"   -> {fpath}")

    finally:
        try: shutil.rmtree(temp_dir)
        except: pass
        print(f"\nTotal Execution Time: {time.time() - t_start_total:.2f} seconds")
        print("Done.")

# Custom Logger Interceptor
class TeeLogger:
    """Duplicates stream writes to both the original terminal and an attached log file."""
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRAND Neutrino Array - Automated Site Search Tool")
    
    # Made DEM and Origin optional here so they can be exclusively provided via config or fallbacks
    parser.add_argument("--dem_path", type=str, help="Path to the Digital Elevation Model (.tif) file.")
    parser.add_argument("--origin_lat", type=float, help="Reference origin latitude (e.g., -10.228).")
    parser.add_argument("--origin_lon", type=float, help="Reference origin longitude (e.g., -78.076).")
    
    # Configuration and Layout Arguments
    parser.add_argument("--target_antennas", type=int, default=10000, help="Total target capacity for the array (default: 10000).")
    parser.add_argument("--min_width_km", type=float, default=2.0, help="Minimum acceptable width of the array site in km (default: 2.0).")
    parser.add_argument("--min_altitude", type=float, default=None, help="Minimum allowable altitude in meters (optional).")
    parser.add_argument("--max_altitude", type=float, default=None, help="Maximum allowable altitude in meters (optional).")
    parser.add_argument("--antenna_spacing_km", type=float, default=1.0, help="Distance between antennas in km (default: 1.0).")
    parser.add_argument("--min_dist_km", type=float, default=10.0, help="Minimum required distance to target mountain in km (default: 10.0).")
    parser.add_argument("--max_dist_km", type=float, default=80.0, help="Maximum required distance to target mountain in km (default: 80.0).")
    parser.add_argument("--grid_type", type=str, choices=['square', 'hex'], default='hex', help="Antenna layout grid type (default: 'hex').")
    
    # Internal Math & Physics Parameters
    parser.add_argument("--fresnel_buffer", type=float, default=200.0, help="Clearance margin (in meters) for line-of-sight ray tracing (default: 200.0).")
    parser.add_argument("--downsample_factor", type=int, default=4, help="Internal capacity mask downsampling factor for processing speed (default: 4).")
    
    # Logistics and Geography Arguments
    parser.add_argument("--rfi_zones", type=str, default='none', help="Can be preset ('lima', 'arequipa') or a valid JSON string outlining custom exclusion zones.")
    parser.add_argument("--road_map_path", type=str, default=None, help="Path to a raster mapping distance-to-roads (optional).")
    parser.add_argument("--max_road_dist_km", type=float, default=20.0, help="Maximum distance allowed from a road in km (default: 20.0).")
    
    # Execution modes and constraints
    parser.add_argument("--search_mode", type=str, choices=['single', 'distributed'], default='distributed', help="'single' finds one monolithic site, 'distributed' allows sub-arrays.")
    parser.add_argument("--min_sub_array_size", type=int, default=500, help="Minimum required capacity for a sub-array to be considered valid (default: 500).")
    parser.add_argument("--min_aspect_deg", type=float, default=None, help="Minimum bound for site facing direction in degrees (0-360).")
    parser.add_argument("--max_aspect_deg", type=float, default=None, help="Maximum bound for site facing direction in degrees (0-360).")
    
    # Metadata and output flags
    parser.add_argument("--region_name", type=str, default=None, help="Cosmetic region name to print on the final visualization chart.")
    parser.add_argument("--generate_kml", action="store_true", help="Include this flag to generate a Google Earth KML file of the findings.")
    parser.add_argument("--no_print_info", action="store_false", dest="print_info", help="Include this flag to skip printing the detailed explanatory text.")
    
    # IO / Configs mapping & Tools
    parser.add_argument("--config_path", type=str, default=None, help="Path to external JSON configuration file.")
    parser.add_argument("--log_path", type=str, default="../output/logs/log.txt", help="Path to store execution log (default: ../output/logs/log.txt).")
    parser.add_argument("--output_image_path", type=str, default=None, help="Specific file path to save the generated map visual (optional).")
    parser.add_argument("--output_image_format", type=str, default="png", help="Format of the saved map visual, e.g., png, pdf, svg (default: png).")
    
    # Tool Generation Arguments
    parser.add_argument("--generate_config", type=str, default=None, help="Supply a filepath to generate a default JSON config template and exit.")
    parser.add_argument("--config_preset", type=str, choices=['default', 'lima', 'arequipa'], default='default', help="Optional presets to inject when using --generate_config.")

    args = parser.parse_args()

    # --- Tool Execution: Generate Config Template ---
    if args.generate_config:
        preset = args.config_preset
        default_config = {
            "dem_path": "path_to_your_dem.tif",
            "origin_lat": 0.0,
            "origin_lon": 0.0,
            "target_antennas": 10000,
            "min_width_km": 2.0,
            "min_altitude": None,
            "max_altitude": None,
            "antenna_spacing_km": 1.0,
            "min_dist_km": 10.0,
            "max_dist_km": 80.0,
            "grid_type": "hex",
            "fresnel_buffer": 200.0,
            "downsample_factor": 4,
            "rfi_zones": "none",
            "road_map_path": None,
            "max_road_dist_km": 20.0,
            "search_mode": "distributed",
            "min_sub_array_size": 500,
            "min_aspect_deg": None,
            "max_aspect_deg": None,
            "region_name": "Custom Region",
            "generate_kml": True,
            "print_info": True,
            "log_path": "../output/logs/log.txt",
            "output_image_path": None,
            "output_image_format": "png"
        }
        
        # Inject presets if specifically requested
        if preset == 'lima':
            default_config['origin_lat'] = ORIGIN_LAT_LIMA
            default_config['origin_lon'] = ORIGIN_LON_LIMA
            default_config['rfi_zones'] = 'lima'
            default_config['region_name'] = 'Lima, Peru'
            default_config['dem_path'] = 'lima_AW3D30.tif'
        elif preset == 'arequipa':
            default_config['origin_lat'] = ORIGIN_LAT_AREQUIPA
            default_config['origin_lon'] = ORIGIN_LON_AREQUIPA
            default_config['rfi_zones'] = 'arequipa'
            default_config['region_name'] = 'Arequipa, Peru'
            default_config['dem_path'] = 'arequipa_SRTMGL1.tif'

        # Safely create directory structure if necessary and save
        dir_name = os.path.dirname(os.path.abspath(args.generate_config))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        with open(args.generate_config, 'w') as f:
            json.dump(default_config, f, indent=4)
        print(f"Configuration file generated successfully at: {args.generate_config} (Preset: {preset})")
        sys.exit(0)


    # 1. Initialize Configuration Maps
    config_params = {}
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            config_params = json.load(f)

    # 2. Retrieve Fallbacks
    fallback_path = os.path.join("..", "config", "fallbacks.json")
    fallback_params = {}
    if os.path.exists(fallback_path):
        with open(fallback_path, 'r') as f:
            fallback_params = json.load(f)

    # Determine logging location hierarchically 
    log_path = args.log_path
    if "log_path" in config_params:
        log_path = config_params["log_path"]
    elif "log_path" in fallback_params:
        log_path = fallback_params["log_path"]

    # 3. Apply Custom Standard-Out / Standard-Error interceptors for the log file
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    
    sys.stdout = TeeLogger(sys.stdout, log_file)
    sys.stderr = TeeLogger(sys.stderr, log_file)

    # Ensure log captures initiation context
    print(f"\n================================================================================")
    print(f"Execution started at: {datetime.now().isoformat()}")
    if args.config_path:
        print(f"Using config file: {os.path.abspath(args.config_path)}")
    else:
        print(f"No config file provided. Relying on CLI arguments and fallbacks.")
    print(f"Using fallbacks file: {os.path.abspath(fallback_path)}")
    print(f"Log file initialized at: {os.path.abspath(log_path)}")
    print(f"================================================================================\n")

    # 4. Reconcile Configuration Strategy (Config > Fallback > CLI / Standard defaults)
    final_params = {}
    
    # Collect all available arguments parsed from CLI framework
    param_names = [action.dest for action in parser._actions if action.dest not in ('help', 'config_path', 'log_path', 'generate_config', 'config_preset')]
    
    for param in param_names:
        if param in config_params:
            final_params[param] = config_params[param]
        elif param in fallback_params:
            final_params[param] = fallback_params[param]
            print(f"WARNING: Parameter '{param}' not explicitly specified in config. Using fallback value: {fallback_params[param]}")
        else:
            final_params[param] = getattr(args, param)

    # Post-validation of absolutely required parameters to prevent early crashes during Fail-Fast
    if final_params.get('dem_path') is None:
        print("ERROR: Critical parameter 'dem_path' must be provided via config file, fallback, or CLI.")
        sys.exit(1)
    if final_params.get('origin_lat') is None or final_params.get('origin_lon') is None:
        print("ERROR: Critical parameters 'origin_lat' and 'origin_lon' must be provided via config file, fallback, or CLI.")
        sys.exit(1)

    # 5. Run Pre-Flight Validation (Fail-Fast Mechanism)
    validate_parameters(final_params)

    # Handle RFI Zone selection mapping (Checks Config-passed custom lists, or matches string presets)
    rfi_input = final_params.get('rfi_zones', 'none')
    selected_rfi = None
    
    if isinstance(rfi_input, str):
        if rfi_input.lower() == 'lima':
            selected_rfi = LIMA_RFI_ZONES
        elif rfi_input.lower() == 'arequipa':
            selected_rfi = AREQUIPA_RFI_ZONES
        elif rfi_input.lower() != 'none':
            # Attempt to parse as JSON if a raw string array was passed via CLI
            try:
                selected_rfi = json.loads(rfi_input)
            except Exception as e:
                print(f"WARNING: Could not parse custom rfi_zones string. Proceeding with 'none'. Error: {e}")
    elif isinstance(rfi_input, list):
        # Naturally supports custom RFI arrays loaded cleanly from the JSON config file
        selected_rfi = rfi_input

    # Output explanations if requested (either via flag or configured true)
    if final_params.get('print_info', True):
        print_tool_explanation()

    # Execute main search pipeline with our integrated parameters
    find_grand_regions_interactive(
        dem_path=final_params['dem_path'],
        target_antennas=final_params['target_antennas'], 
        rfi_zones=selected_rfi,
        min_width_km=final_params['min_width_km'],
        origin_lat=final_params['origin_lat'],
        origin_lon=final_params['origin_lon'],
        min_altitude=final_params['min_altitude'], 
        max_altitude=final_params['max_altitude'],
        antenna_spacing_km=final_params['antenna_spacing_km'],
        min_dist_km=final_params['min_dist_km'],
        max_dist_km=final_params['max_dist_km'],
        grid_type=final_params['grid_type'],       
        generate_kml=final_params['generate_kml'],     
        road_map_path=final_params['road_map_path'],    
        max_road_dist_km=final_params['max_road_dist_km'],
        search_mode=final_params['search_mode'],
        min_sub_array_size=final_params['min_sub_array_size'],
        min_aspect_deg=final_params['min_aspect_deg'], 
        max_aspect_deg=final_params['max_aspect_deg'],
        region_name=final_params['region_name'],
        fresnel_buffer=final_params['fresnel_buffer'],
        downsample_factor=final_params['downsample_factor'],
        output_image_path=final_params['output_image_path'],
        output_image_format=final_params['output_image_format']
    )