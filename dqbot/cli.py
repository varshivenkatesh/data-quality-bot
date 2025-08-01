import argparse
import json
import sys
import os
from pathlib import Path
import pandas as pd
from io import BytesIO
try:
    import yaml
except ImportError:
    yaml = None

from dqbot.core import generate_report, get_default_config, load_config, visualize_report
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

# ============================================================================
# OPTIONAL AZURE INTEGRATION
# Uncomment the section below if you want to enable Azure Blob Storage support
# ============================================================================

# # Azure Container Connections - use the below setup!
# import os

# # Optional: load .env if it exists
# try:
#     from dotenv import load_dotenv

#     dotenv_path = os.path.join(os.getcwd(), ".env")
#     if os.path.exists(dotenv_path):
#         load_dotenv(dotenv_path)
# except ImportError:
#     # dotenv not installed — ignore silently
#     pass

# # Access environment variables (set in environment or optionally from .env)
# AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
# DEFAULT_BLOB_CONTAINER = os.getenv("INPUT_CONTAINER_NAME", "default-input-container")
# OUTPUT_BLOB_CONTAINER = os.getenv("OUTPUT_CONTAINER_NAME", "default-output-container")

# # Optional: raise helpful error if required env vars not set
# if not AZURE_STORAGE_CONNECTION_STRING:
#     raise EnvironmentError(
#         "Missing AZURE_STORAGE_CONNECTION_STRING. Set it as an environment variable."
#     )

# # Uncomment the import below if enabling Azure
# # from azure.storage.blob import BlobServiceClient

# ============================================================================
# LOCAL FILE PROCESSING (DEFAULT)
# ============================================================================

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.json'}

def generate_default_config(output_path="dqbot_config.yaml"):
    """Generate a default configuration file."""
    if yaml is None:
        print("❌ Error: PyYAML is required to generate YAML configs. Install with `pip install pyyaml`.")
        return False
    
    default_config = get_default_config()
    
    # Add helpful comments to the config
    config_with_comments = """# DQBot Configuration File
# Enable/disable specific data quality checks and configure their parameters

# Basic dataset statistics
basic_stats:
  enabled: true
  include: [num_rows, num_columns, column_types, columns]

# Null value analysis
null_analysis:
  enabled: true
  include_counts: true
  include_percentages: true

# Duplicate row analysis
duplicate_analysis:
  enabled: true
  include_count: true
  include_percentage: true

# Unique value counts for each column
unique_values:
  enabled: true

# Detect columns with constant values
constant_values:
  enabled: true

# Detect categorical columns with high cardinality
high_cardinality:
  enabled: true
  threshold: 0.5  # Fraction of total rows

# Detect potential primary key columns
primary_key_detection:
  enabled: true

# Detect columns with mixed data types
mixed_types:
  enabled: true

# Detect columns with whitespace issues
whitespace_issues:
  enabled: true

# Outlier detection for numeric columns
outlier_detection:
  enabled: true
  method: iqr  # Options: iqr, zscore
  iqr_multiplier: 1.5
  zscore_threshold: 3

# Value distribution analysis for categorical columns
value_distribution:
  enabled: true
  top_n: 5

# Date/datetime column analysis
date_analysis:
  enabled: true
  check_future_dates: true

# Legacy custom checks (original format)
custom_checks:
  numeric_ranges:
    # Example:
    # - column: quantity
    #   min: 1
    #   max: 1000
  date_validation:
    # Example:
    # - column: order_date
    #   allow_future_dates: false
  value_sets:
    # Example:
    # - column: status
    #   allowed_values: [pending, shipped, delivered]
"""
    
    try:
        with open(output_path, 'w') as f:
            f.write(config_with_comments)
        return True
    except Exception as e:
        print(f"❌ Error writing config file: {e}")
        return False


def validate_file_path(file_path):
    """Validate that the file exists and is readable."""
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return False
    
    if not os.path.isfile(file_path):
        print(f"❌ Error: Path is not a file: {file_path}")
        return False
    
    return True


def validate_folder_path(folder_path):
    """Validate that the folder exists and is readable."""
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder not found: {folder_path}")
        return False
    
    if not os.path.isdir(folder_path):
        print(f"❌ Error: Path is not a folder: {folder_path}")
        return False
    
    return True


def get_supported_files_in_folder(folder_path):
    """Get all supported files in the given folder."""
    supported_files = []
    
    try:
        for file_path in Path(folder_path).iterdir():
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                supported_files.append(file_path)
    except Exception as e:
        print(f"❌ Error reading folder contents: {e}")
        return []
    
    return sorted(supported_files)


def load_file_from_bytes(file_ext, byte_data):
    """Load file from bytes based on extension."""
    if file_ext == '.csv':
        return pd.read_csv(BytesIO(byte_data))
    elif file_ext == '.xlsx':
        return pd.read_excel(BytesIO(byte_data))
    elif file_ext == '.json':
        return pd.read_json(BytesIO(byte_data))
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def load_file_from_path(file_ext, path):
    """Load file from path based on extension."""
    if file_ext == '.csv':
        return pd.read_csv(path)
    elif file_ext == '.xlsx':
        return pd.read_excel(path)
    elif file_ext == '.json':
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def process_single_file(file_path, config, args, is_folder_mode=False):
    """Process a single file and generate report."""
    file_name = os.path.basename(file_path)
    file_ext = Path(file_path).suffix.lower()
    
    if is_folder_mode:
        print(f"\n{'='*60}")
        print(f"📊 Processing: {file_name}")
        print(f"{'='*60}")
    
    try:
        # Load file (local only - Azure support commented out)
        if args.azure:
            print("❌ Error: Azure Blob Storage support is currently disabled.")
            print("💡 To enable Azure support, uncomment the Azure integration section in the code.")
            return False
            
            # # AZURE CODE (COMMENTED OUT)
            # if not args.container:
            #     print("❌ Error: --container is required when using --azure")
            #     return False
            # print(f"☁️  Fetching '{file_name}' from Azure Blob Storage container '{args.container}'...")

            # blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
            # blob_client = blob_service_client.get_blob_client(container=args.container, blob=file_name)
            # blob_data = blob_client.download_blob().readall()
            # df = load_file_from_bytes(file_ext, blob_data)
            # print(f"✅ Loaded Azure blob '{file_name}' with {len(df)} rows, {len(df.columns)} columns")
        else:
            if not validate_file_path(file_path):
                return False
            print(f"📊 Loading file: {file_path}")
            df = load_file_from_path(file_ext, file_path)
            print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

    except Exception as e:
        print(f"❌ Error loading file {file_name}: {e}")
        return False

    try:
        # Generate report
        print("🔍 Running data quality analysis...")
        report = generate_report(df, config)
        
        # Add file name to report
        report['file_name'] = file_name

    except Exception as e:
        print(f"❌ Error generating report for {file_name}: {e}")
        return False
    
    try:
        # Determine output file name
        base_name = Path(file_name).stem
        if is_folder_mode:
            output_file = f"{base_name}_dq_report.json"
        else:
            output_file = os.path.basename(args.output)
        
        # Save report (local only)
        if args.azure:
            print("❌ Error: Azure output is currently disabled.")
            return False
            
            # # AZURE OUTPUT CODE (COMMENTED OUT)
            # output_json = json.dumps(report, indent=2)
            # blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
            # output_container_client = blob_service_client.get_blob_client(container="quality-reports", blob=output_file)
            # output_container_client.upload_blob(output_json, overwrite=True)
            # print(f"☁️  Report saved to Azure container 'quality-reports' as blob '{output_file}'")
            # output_path = None  # Can't generate vis for Azure files
        else:
            output_path = os.path.join("output", output_file)
            os.makedirs("output", exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"✅ Report saved to: {output_path}")

        # Print summary
        print(f"\n📋 Analysis Summary for {file_name}:")
        if 'num_rows' in report:
            print(f"   • Rows: {report['num_rows']:,}")
        if 'num_columns' in report:
            print(f"   • Columns: {report['num_columns']}")
        if 'duplicate_row_count' in report:
            print(f"   • Duplicates: {report['duplicate_row_count']}")
        if 'null_counts' in report:
            total_nulls = sum(report['null_counts'].values())
            print(f"   • Total null values: {total_nulls:,}")

        # Generate visualization if requested and not using Azure
        if args.vis and not args.azure and output_path:
            try:
                print(f"📊 Generating visualization for {file_name}...")
                with open(output_path, "r") as f:
                    report_data = json.load(f)
                visualize_report(report_data, output_path)
                print(f"✅ Visualization generated for {file_name}")
            except Exception as e:
                print(f"⚠️  Warning: Could not generate visualization for {file_name}: {e}")

    except Exception as e:
        print(f"❌ Error saving report for {file_name}: {e}")
        return False
    
    return True


def run_analysis(args):
    """Run the data quality analysis."""
    # Validate input arguments
    if args.file and args.folder:
        print("❌ Error: Cannot use both --file and --folder options together")
        return 1
    
    if not args.file and not args.folder:
        print("❌ Error: Must specify either --file or --folder")
        return 1
    
    # Load configuration
    try:
        if args.config:
            if not os.path.exists(args.config):
                print(f"❌ Error: Config file not found: {args.config}")
                return 1
            config = load_config(args.config)
            print(f"✅ Using config file: {args.config}")
        else:
            config = get_default_config()
            print("ℹ️  Using default configuration (all checks enabled)")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return 1
    
    # Process files
    if args.folder:
        # Folder mode
        if args.azure:
            print("❌ Error: Azure Blob Storage support is currently disabled for folder mode.")
            print("💡 To enable Azure support, uncomment the Azure integration section in the code.")
            return 1
        
        folder_path = os.path.join("data", args.folder)
        if not validate_folder_path(folder_path):
            return 1
        
        # Get all supported files in folder
        supported_files = get_supported_files_in_folder(folder_path)
        
        if not supported_files:
            print(f"⚠️  No supported files found in folder: {folder_path}")
            print(f"🔍 Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
            return 1
        
        print(f"📁 Found {len(supported_files)} supported files in folder: {folder_path}")
        print(f"🔍 Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        
        # Process each file
        success_count = 0
        for file_path in supported_files:
            if process_single_file(str(file_path), config, args, is_folder_mode=True):
                success_count += 1
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"🏁 FOLDER PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Successfully processed: {success_count}/{len(supported_files)} files")
        if success_count < len(supported_files):
            print(f"❌ Failed to process: {len(supported_files) - success_count} files")
        
        return 0 if success_count > 0 else 1
        
    else:
        # Single file mode
        if not args.azure:
            file_path = os.path.join("data", os.path.basename(args.file))
        else:
            file_path = args.file
        
        success = process_single_file(file_path, config, args, is_folder_mode=False)
        return 0 if success else 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='dqbot',
        description='Data Quality Bot - Configurable data quality analysis tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file analysis
  dqbot run --file data.csv                          # Use default config
  dqbot run --file data.csv --output report.json     # Custom output file
  dqbot run --file data.csv --vis                    # Generate visualization
  
  # Folder analysis (processes all supported files)
  dqbot run --folder my_data_folder                  # Process all files in folder
  dqbot run --folder my_data_folder --vis            # Process with visualizations
  
  # Azure Blob Storage (CURRENTLY DISABLED - see code comments to enable)
  # dqbot run --file data.csv --azure --container my-container
  
  # Configuration
  # Clone the project from github to add custom configs!
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run data quality analysis')
    
    # Mutually exclusive group for file vs folder
    input_group = run_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--file', '-f', 
        help='Path to data file to analyze (CSV, XLSX, JSON)'
    )
    input_group.add_argument(
        '--folder', '-d', 
        help='Path to folder containing data files to analyze (processes all supported files)'
    )
    
    run_parser.add_argument(
        '--config', '-c', 
        help='Path to JSON/YAML configuration file (optional, uses defaults if not provided)'
    )
    run_parser.add_argument(
        '--output', '-o', 
        default='output/dq_report.json', 
        help='Output report file path (default: output/dq_report.json). In folder mode, files are named automatically.'
    )
    run_parser.add_argument(
        '--azure', '-az',
        action='store_true',
        help='[CURRENTLY DISABLED] Fetch file from Azure Blob Storage instead of local file'
    )
    run_parser.add_argument(
        '--container', 
        default='default-input-container',
        help='[CURRENTLY DISABLED] Azure blob container name'
    )
    run_parser.add_argument(
        '--vis', 
        action='store_true', 
        help='Generate dashboard visualization from report(s)'
    )
    
    # Parse args
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == 'run':
        result = run_analysis(args)
        return result

    return 0


if __name__ == '__main__':
    sys.exit(main())