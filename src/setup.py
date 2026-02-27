import argparse
import sys
import os
import urllib.request
import urllib.error
import json
import subprocess
from tqdm import tqdm

# ==========================================
#          UI THEME & FORMATTING
# ==========================================
if sys.platform == 'win32':
    os.system('') 

def supports_color():
    supported_platform = sys.platform != 'win32' or 'ANSICON' in os.environ or 'WT_SESSION' in os.environ or os.environ.get('TERM') == 'xterm-256color'
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    return supported_platform and is_a_tty

def supports_emoji():
    if sys.stdout.encoding:
        return sys.stdout.encoding.lower() == 'utf-8'
    return False

USE_COLOR = supports_color()
USE_EMOJI = supports_emoji()

class C:
    HEADER = '\033[96m' if USE_COLOR else ''
    OK = '\033[92m' if USE_COLOR else ''
    WARN = '\033[93m' if USE_COLOR else ''
    FAIL = '\033[91m' if USE_COLOR else ''
    BOLD = '\033[1m' if USE_COLOR else ''
    MAGENTA = '\033[95m' if USE_COLOR else ''
    RESET = '\033[0m' if USE_COLOR else ''

class Icon:
    MAP = '🗺️  ' if USE_EMOJI else '[*] '
    GEAR = '⚙️  ' if USE_EMOJI else '[~] '
    DISK = '💾 ' if USE_EMOJI else '[S] '
    WARN = '⚠️  ' if USE_EMOJI else '[!] '
    CHECK = '✅ ' if USE_EMOJI else '[✓] '
    CROSS = '❌ ' if USE_EMOJI else '[x] '
    INFO = 'ℹ️  ' if USE_EMOJI else '[i] '

# ==========================================
#             TARGET REGIONS
# ==========================================
# Defined as (West, East, South, North)
REGIONS = {
    "lima": {
        "west": -78.07665824890137,
        "east": -75.39955615997313,
        "south": -13.252477566131276,
        "north": -10.228479499469358,
        "filename": "lima_AW3D30.tif",
        "preset": "lima",
        "demtype": "AW3D30"
    },
    "arequipa": {
        "west": -73.58612537384033,
        "east": -70.0852632522583,
        "south": -17.38995824658555,
        "north": -14.555380967667489,
        "filename": "arequipa_SRTMGL1.tif",
        "preset": "arequipa",
        "demtype": "SRTMGL1"
    }
}

# ==========================================
#               CORE LOGIC
# ==========================================
class TqdmUpTo(tqdm):
    """Callback class for urlretrieve to print a dynamic tqdm progress bar."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_dem(region_name, bounds, api_key, output_dir):
    """Constructs the API request and downloads the GTiff file."""
    url = (f"https://portal.opentopography.org/API/globaldem"
           f"?demtype={bounds['demtype']}"
           f"&south={bounds['south']}"
           f"&north={bounds['north']}"
           f"&west={bounds['west']}"
           f"&east={bounds['east']}"
           f"&outputFormat=GTiff"
           f"&API_Key={api_key}")
    
    filepath = os.path.join(output_dir, bounds['filename'])
    
    print(f"\n{C.BOLD}Processing Region: {region_name.title()}{C.RESET}")
    print(f"   {Icon.MAP}Requesting {bounds['demtype']} DEM from OpenTopography...")
    
    try:
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                      desc=f"      {Icon.INFO}Downloading", colour='magenta' if USE_COLOR else None) as t:
            urllib.request.urlretrieve(url, filepath, reporthook=t.update_to)
            
        print(f"      {Icon.CHECK}{C.OK}Saved to: {filepath}{C.RESET}")
        return filepath
    except urllib.error.HTTPError as e:
        print(f"      {Icon.CROSS}{C.FAIL}HTTP Error {e.code}: {e.reason}{C.RESET}")
        if e.code == 401:
            print(f"      {C.WARN}Make sure your OpenTopography API key is valid and active.{C.RESET}")
        return None
    except Exception as e:
        print(f"      {Icon.CROSS}{C.FAIL}Download failed: {e}{C.RESET}")
        return None

def generate_and_patch_config(region_name, preset, dem_filepath):
    """Calls site_searcher.py to generate a template, then updates the DEM path."""
    config_dir = os.path.join("..", "config")
    os.makedirs(config_dir, exist_ok=True)
    config_filename = os.path.join(config_dir, f"{region_name}_config.json")
    
    print(f"   {Icon.GEAR}Generating config file via site_searcher.py...")
    
    # 1. Generate the raw config using the main script
    try:
        subprocess.run([
            sys.executable, "site_searcher.py", 
            "--generate_config", config_filename, 
            "--config_preset", preset
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"      {Icon.CROSS}{C.FAIL}Failed to generate config. Ensure 'site_searcher.py' is in this directory.{C.RESET}")
        return
        
    # 2. Patch the JSON to point to the newly downloaded DEM
    try:
        with open(config_filename, 'r') as f:
            config_data = json.load(f)
            
        # Use a forward-slash relative path starting from the config's directory
        rel_dem_path = os.path.relpath(dem_filepath, start=config_dir).replace('\\', '/')
        config_data['dem_path'] = rel_dem_path
        
        with open(config_filename, 'w') as f:
            json.dump(config_data, f, indent=4)
            
        print(f"      {Icon.CHECK}{C.OK}Config configured and saved to: {config_filename}{C.RESET}")
    except Exception as e:
        print(f"      {Icon.CROSS}{C.FAIL}Error patching JSON config: {e}{C.RESET}")

# ==========================================
#                  MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRAND Site Search - Data Setup & Downloader")
    parser.add_argument("--open_topography_api_key", type=str, default=None, 
                        help="Your OpenTopography API key (Required to download DEMs).")
    
    args = parser.parse_args()

    print(f"\n{C.HEADER}===================================================={C.RESET}")
    print(f"{C.BOLD}   GRAND OPENTOPOGRAPHY DEM SETUP{C.RESET}")
    print(f"{C.HEADER}===================================================={C.RESET}")

    # Explicit enforcement of the API Key
    if not args.open_topography_api_key:
        print(f"\n{C.FAIL}{Icon.CROSS}ERROR: Missing Required Parameter.{C.RESET}")
        print(f"{C.WARN}The flag {C.BOLD}--open_topography_api_key{C.RESET}{C.WARN} must be provided.{C.RESET}")
        print(f"If you do not have an API key, you cannot use this setup script and must download the TIF files manually from OpenTopography.")
        print(f"Register for a free key at: {C.MAGENTA}https://portal.opentopography.org/myopentopo{C.RESET}\n")
        sys.exit(1)

    # Ensure target script exists
    if not os.path.exists("site_searcher.py"):
        print(f"\n{C.FAIL}{Icon.CROSS}ERROR: 'site_searcher.py' not found.{C.RESET}")
        print(f"{C.WARN}Please run this setup script from the same directory as your main code.{C.RESET}\n")
        sys.exit(1)

    # Dependency check using generate_env.py
    try:
        import generate_env
        print(f"\n{C.HEADER}===================================================={C.RESET}")
        print(f"{C.BOLD}   DEPENDENCY CHECK{C.RESET}")
        print(f"{C.HEADER}===================================================={C.RESET}")
        print(f"   {Icon.GEAR}Scanning 'site_searcher.py' for dependencies...")
        deps = generate_env.extract_dependencies("site_searcher.py")
        satisfied, missing = generate_env.check_installed_modules(deps)
        
        if missing:
            print(f"   {C.FAIL}{Icon.CROSS}Missing dependencies detected: {', '.join(missing)}{C.RESET}")
            print(f"   {C.WARN}It is highly recommended to run 'python generate_env.py' to setup your environment.{C.RESET}")
        else:
            print(f"   {C.OK}{Icon.CHECK}All script dependencies are satisfied!{C.RESET}")
    except ImportError:
        print(f"\n   {C.WARN}{Icon.WARN}Could not import 'generate_env.py' to verify dependencies. Skipping check.{C.RESET}")
    except Exception as e:
        print(f"\n   {C.FAIL}{Icon.CROSS}Error checking dependencies: {e}{C.RESET}")

    # Establish the ../input/dem/ directory structure
    print(f"\n{C.HEADER}===================================================={C.RESET}")
    print(f"{C.BOLD}   DOWNLOADING ASSETS{C.RESET}")
    print(f"{C.HEADER}===================================================={C.RESET}")
    output_dir = os.path.join("..", "input", "dem")
    os.makedirs(output_dir, exist_ok=True)
    print(f"   {Icon.DISK}Target directory verified: {C.MAGENTA}{os.path.abspath(output_dir)}{C.RESET}")

    # Process each region
    for region, bounds in REGIONS.items():
        downloaded_path = download_dem(region, bounds, args.open_topography_api_key, output_dir)
        
        if downloaded_path:
            generate_and_patch_config(region, bounds['preset'], downloaded_path)
            
    print(f"\n{C.HEADER}===================================================={C.RESET}")
    print(f"{C.OK}{Icon.CHECK}{C.BOLD}Setup Complete.{C.RESET}")
    print(f"You can now run the main script using your new configs:")
    print(f"  {C.MAGENTA}python site_searcher.py --config_path ../config/lima_config.json{C.RESET}")
    print(f"  {C.MAGENTA}python site_searcher.py --config_path ../config/arequipa_config.json{C.RESET}\n")