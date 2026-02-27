import ast
import sys
import os
import importlib.util

# ==========================================
#          UI THEME & FORMATTING
# ==========================================
if sys.platform == 'win32':
    os.system('') 

def supports_color():
    """Checks if the terminal supports ANSI colors."""
    supported_platform = sys.platform != 'win32' or 'ANSICON' in os.environ or 'WT_SESSION' in os.environ or os.environ.get('TERM') == 'xterm-256color'
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    return supported_platform and is_a_tty

def supports_emoji():
    """Checks if the terminal supports UTF-8 for emojis."""
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
    INFO = 'ℹ️  ' if USE_EMOJI else '[i] '
    CHECK = '✅ ' if USE_EMOJI else '[✓] '
    CROSS = '❌ ' if USE_EMOJI else '[x] '
    WARN = '⚠️  ' if USE_EMOJI else '[!] '

# ==========================================
#          STANDARD LIBRARY FILTER
# ==========================================
def get_stdlib_modules():
    """
    Returns a set of standard Python library module names.
    Uses Python 3.10+ native sys.stdlib_module_names if available, 
    otherwise relies on a robust fallback list covering standard modules.
    """
    if hasattr(sys, 'stdlib_module_names'):
        return set(sys.stdlib_module_names).union(set(sys.builtin_module_names))
    else:
        # Robust fallback for Python < 3.10
        return {
            'argparse', 'sys', 'os', 'shutil', 'tempfile', 'math', 'time', 'json',
            'xml', 'datetime', 're', 'multiprocessing', 'typing', 'pathlib',
            'collections', 'itertools', 'functools', 'logging', 'subprocess',
            'warnings', 'csv', 'urllib', 'requests', 'sqlite3', 'io', 'ast', 
            'importlib', 'socket', 'threading', 'queue', 'random', 'hashlib'
        }

# ==========================================
#               CORE LOGIC
# ==========================================
def extract_dependencies(filepath):
    """
    Parses a Python file using the Abstract Syntax Tree (AST) to identify
    all base module imports, ensuring no missed imports in try/except blocks.
    
    Parameters:
    - filepath (str): Path to the target Python script.
    
    Returns:
    - list: A sorted list of identified third-party dependency strings.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        tree = ast.parse(file.read(), filename=filepath)

    raw_imports = set()
    
    # Walk the syntax tree looking for 'import X' and 'from X import Y'
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                raw_imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                raw_imports.add(node.module.split('.')[0])

    stdlib_modules = get_stdlib_modules()
    
    # Filter out standard libraries and the script itself (if imported)
    third_party_deps = {
        mod for mod in raw_imports 
        if mod not in stdlib_modules and mod != os.path.splitext(os.path.basename(filepath))[0]
    }
    
    return sorted(list(third_party_deps))

def check_installed_modules(dependencies):
    """
    Checks which of the extracted module dependencies are currently installed
    and accessible in the active Python environment.
    """
    satisfied = []
    missing = []
    for dep in dependencies:
        try:
            # Using find_spec is the safest way to check if a module exists without actually executing it
            if importlib.util.find_spec(dep) is not None:
                satisfied.append(dep)
            else:
                missing.append(dep)
        except Exception:
            missing.append(dep)
    return satisfied, missing

def generate_conda_yaml(dependencies, output_file="environment.yml", env_name="site_search"):
    """
    Generates a formatted YAML file compatible with Conda.
    Prioritizes the conda-forge channel for scientific packages.
    
    Parameters:
    - dependencies (list): List of extracted third-party module names.
    - output_file (str): Desired output filepath.
    - env_name (str): Name of the generated conda environment.
    """
    yaml_lines = [
        f"name: {env_name}",
        "channels:",
        "  - conda-forge",
        "  - defaults",
        "dependencies:",
        "  - python>=3.8"  # Base Python requirement
    ]
    
    for dep in dependencies:
        yaml_lines.append(f"  - {dep}")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(yaml_lines) + "\n")

# ==========================================
#                  MAIN
# ==========================================
if __name__ == "__main__":
    print(f"\n{C.HEADER}===================================================={C.RESET}")
    print(f"{C.BOLD}   GRAND CONDA ENVIRONMENT GENERATOR{C.RESET}")
    print(f"{C.HEADER}===================================================={C.RESET}")
    
    target_script = "site_searcher.py"
    output_yml = "environment.yml"
    env_name = "grand_site_search"
    
    if not os.path.exists(target_script):
        print(f"   {C.FAIL}[!] Target script '{target_script}' not found in the current directory.{C.RESET}")
        print(f"   {C.WARN}Please run this script from the same directory as your main code.{C.RESET}\n")
        sys.exit(1)
        
    print(f"   -> Analyzing AST for: {C.MAGENTA}{target_script}{C.RESET}")
    
    try:
        deps = extract_dependencies(target_script)
        print(f"   -> External dependencies found: {C.MAGENTA}{len(deps)}{C.RESET}")
        for d in deps:
            print(f"      - {d}")
            
        # Check active Conda environment and satisfied dependencies
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env:
            print(f"\n   {Icon.INFO}Active Conda Environment Detected: {C.MAGENTA}{conda_env}{C.RESET}")
            satisfied, missing = check_installed_modules(deps)
            
            if satisfied:
                print(f"   -> Already satisfied ({len(satisfied)}/{len(deps)}): {C.OK}{', '.join(satisfied)}{C.RESET}")
            if missing:
                print(f"   -> Missing ({len(missing)}/{len(deps)}): {C.FAIL}{', '.join(missing)}{C.RESET}")
            elif len(satisfied) == len(deps):
                print(f"   {C.OK}{Icon.CHECK}All dependencies are currently satisfied in '{conda_env}'!{C.RESET}")
            
        generate_conda_yaml(deps, output_yml, env_name)
        
        print(f"\n{C.OK}{Icon.CHECK}Environment file successfully generated: {C.BOLD}{output_yml}{C.RESET}")
        
        print(f"\n{C.HEADER}===================================================={C.RESET}")
        print(f"   {C.BOLD}HOW TO DEPLOY YOUR ENVIRONMENT:{C.RESET}")
        print(f"{C.HEADER}===================================================={C.RESET}")
        print(f"1. To create the environment, copy and run this command:")
        print(f"   {C.MAGENTA}conda env create -f {output_yml}{C.RESET}\n")
        print(f"2. To activate the environment, run:")
        print(f"   {C.MAGENTA}conda activate {env_name}{C.RESET}\n")
        
    except Exception as e:
        print(f"\n   {C.FAIL}{Icon.CROSS}An error occurred during AST parsing/generation: {e}{C.RESET}")
        sys.exit(1)