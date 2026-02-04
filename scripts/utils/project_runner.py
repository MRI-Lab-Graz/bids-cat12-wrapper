#!/usr/bin/env python3
"""
Project Runner - Execute CAT12 processing from project config JSON
Includes preflight checks, OpenNeuro download, and pipeline execution
"""

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

try:
    import openneuro
except ImportError:
    openneuro = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProjectRunner:
    """Run CAT12 project from configuration file."""
    
    def __init__(self, config_path: Path):
        """Initialize with project config file."""
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load environment variables from .env file
        self._load_env_file()
        
        with open(self.config_path) as f:
            self.config = json.load(f)
        
        self.project_folder = Path(self.config['study']['project_folder'])
        self.project_folder.mkdir(parents=True, exist_ok=True)
    
    def _load_env_file(self):
        """Load environment variables from .env file in project root."""
        # Look for .env file in the project root (3 levels up from this script)
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / '.env'
        
        if env_file.exists():
            logger.info(f"Loading environment from {env_file}")
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Handle export statements
                        if line.startswith('export '):
                            line = line[7:]
                        
                        # Parse KEY=VALUE
                        if '=' in line:
                            key, value = line.split('=', 1)
                            # Remove quotes if present
                            value = value.strip().strip('"').strip("'")
                            
                            # Expand environment variables in value
                            import re
                            # Handle $VAR and ${VAR} syntax
                            def expand_var(match):
                                var_name = match.group(1) or match.group(2)
                                return os.environ.get(var_name, match.group(0))
                            
                            value = re.sub(r'\$\{([^}]+)\}|\$(\w+)', expand_var, value)
                            
                            # Expand special shell variables
                            value = value.replace('$(pwd)', str(project_root))
                            value = value.replace('$HOME', str(Path.home()))
                            value = os.path.expanduser(value)
                            
                            os.environ[key.strip()] = value
                            logger.debug(f"  Set {key.strip()}={value}")
        else:
            logger.warning(f".env file not found at {env_file}")
        
    def run_preflight_checks(self) -> bool:
        """Run preflight checks before processing."""
        logger.info("=" * 70)
        logger.info("PREFLIGHT CHECKS")
        logger.info("=" * 70)
        
        checks = self.config['preprocessing'].get('preflight_checks', {})
        if not checks.get('check_environment', True):
            logger.info("Preflight checks disabled")
            return True
        
        all_passed = True
        
        # Check 1: Environment variables
        logger.info("\n✓ Checking environment variables...")
        required_vars = ['CAT12_ROOT', 'MCR_ROOT', 'USE_STANDALONE']
        for var in required_vars:
            value = os.environ.get(var)
            if value:
                logger.info(f"  ✓ {var}={value}")
            else:
                logger.error(f"  ✗ {var} not set!")
                all_passed = False
        
        # Check 2: CAT12 standalone executable
        logger.info("\n✓ Checking CAT12 standalone...")
        cat12_root = os.environ.get('CAT12_ROOT')
        if cat12_root:
            cat12_script = Path(cat12_root) / "standalone" / "cat_standalone.sh"
            if cat12_script.exists():
                logger.info(f"  ✓ Found: {cat12_script}")
            else:
                logger.error(f"  ✗ Not found: {cat12_script}")
                all_passed = False
        
        # Check 3: MCR installation
        logger.info("\n✓ Checking MATLAB Runtime...")
        mcr_root = os.environ.get('MCR_ROOT')
        if mcr_root:
            mcr_path = Path(mcr_root).expanduser()
            if mcr_path.exists():
                logger.info(f"  ✓ Found: {mcr_path}")
            else:
                logger.error(f"  ✗ Not found: {mcr_path}")
                all_passed = False
        
        # Check 4: Disk space
        if checks.get('check_disk_space', True):
            logger.info("\n✓ Checking disk space...")
            min_space_gb = checks.get('min_disk_space_gb', 50)
            stat = shutil.disk_usage(self.project_folder)
            free_gb = stat.free / (1024**3)
            if free_gb >= min_space_gb:
                logger.info(f"  ✓ Free space: {free_gb:.1f} GB (required: {min_space_gb} GB)")
            else:
                logger.warning(f"  ⚠ Low disk space: {free_gb:.1f} GB (recommended: {min_space_gb} GB)")
        
        # Check 5: Python dependencies
        if checks.get('check_dependencies', True):
            logger.info("\n✓ Checking Python dependencies...")
            try:
                import bids
                import nibabel
                logger.info("  ✓ Required packages installed")
            except ImportError as e:
                logger.error(f"  ✗ Missing package: {e}")
                all_passed = False
        
        logger.info("\n" + "=" * 70)
        if all_passed:
            logger.info("✓ All preflight checks PASSED")
        else:
            logger.error("✗ Some preflight checks FAILED")
        logger.info("=" * 70 + "\n")
        
        return all_passed
    
    def validate_subjects_sessions(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate that subjects and sessions exist in the OpenNeuro dataset.
        Returns (valid, actual_subjects, actual_sessions)
        """
        bids_config = self.config['preprocessing']['bids']
        
        if not bids_config.get('openneuro', False):
            return True, [], []
        
        logger.info("=" * 70)
        logger.info("VALIDATING SUBJECTS & SESSIONS")
        logger.info("=" * 70)
        
        dataset_id = bids_config['openneuro_dataset']
        requested_subjects = bids_config.get('participant_label', [])
        requested_sessions = bids_config.get('session_label', [])
        
        logger.info(f"Dataset: {dataset_id}")
        logger.info(f"Requested subjects: {requested_subjects}")
        logger.info(f"Requested sessions: {requested_sessions}")
        
        # Check if participants.tsv exists locally
        bids_dir = self.project_folder / bids_config['bids_dir']
        participants_file = bids_dir / 'participants.tsv'
        
        if not participants_file.exists():
            logger.info("\nparticipants.tsv not found locally, downloading metadata with AWS CLI...")
            bids_dir.mkdir(parents=True, exist_ok=True)
            
            # Use AWS CLI for faster metadata download
            s3_url = f"s3://openneuro.org/{dataset_id}"
            cmd = [
                'aws', 's3', 'cp', '--no-sign-request',
                f"{s3_url}/participants.tsv",
                str(participants_file)
            ]
            
            try:
                subprocess.run(cmd, check=True, timeout=60)
                logger.info("✓ Downloaded participants.tsv")
            except Exception as e:
                logger.error(f"Failed to download participants.tsv: {e}")
                return False, [], []
        
        # Read participants.tsv to get actual subjects
        import pandas as pd
        try:
            df = pd.read_csv(participants_file, sep='\t')
            actual_subjects = df['participant_id'].tolist()
            # Remove 'sub-' prefix for comparison
            actual_subjects_clean = [s.replace('sub-', '') for s in actual_subjects]
            
            logger.info(f"\n✓ Found {len(actual_subjects)} subjects in dataset")
            logger.info(f"  Sample subjects: {actual_subjects[:5]}")
            
            # Validate requested subjects
            invalid_subjects = []
            for sub in requested_subjects:
                if sub not in actual_subjects_clean:
                    invalid_subjects.append(sub)
                    logger.error(f"  ✗ Subject '{sub}' not found in dataset!")
            
            if invalid_subjects:
                logger.error(f"\nInvalid subjects: {invalid_subjects}")
                logger.info(f"Available subjects: {actual_subjects_clean[:10]}...")
                return False, actual_subjects_clean, []
            
            logger.info(f"✓ All requested subjects exist in dataset")
            
        except Exception as e:
            logger.error(f"Failed to read participants.tsv: {e}")
            return False, [], []
        
        # For sessions, we'll validate during download
        logger.info(f"✓ Session validation will occur during download")
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ Validation complete")
        logger.info("=" * 70 + "\n")
        
        return True, actual_subjects_clean, requested_sessions
    
    def download_openneuro(self) -> bool:
        """Download OpenNeuro dataset using openneuro-py."""
        bids_config = self.config['preprocessing']['bids']
        
        if not bids_config.get('openneuro', False):
            logger.info("OpenNeuro download not enabled")
            return True
        
        if not bids_config.get('openneuro_download', False):
            logger.info("OpenNeuro download disabled in config")
            return True
        
        if openneuro is None:
            logger.error("openneuro-py not installed. Install with: pip install openneuro-py")
            return False
        
        logger.info("=" * 70)
        logger.info("DOWNLOADING OPENNEURO DATASET")
        logger.info("=" * 70)
        
        dataset_id = bids_config['openneuro_dataset']
        bids_dir = self.project_folder / bids_config['bids_dir']
        bids_dir.mkdir(parents=True, exist_ok=True)
        
        participants = bids_config.get('participant_label', [])
        sessions = bids_config.get('session_label', [])
        only_anat = bids_config.get('openneuro_download_only_anat', True)
        
        logger.info(f"Dataset: {dataset_id}")
        logger.info(f"Target: {bids_dir}")
        logger.info(f"Subjects: {participants}")
        logger.info(f"Sessions: {sessions}")
        
        # Download using openneuro-py
        # Build include patterns for specific subjects and sessions
        include_patterns = []
        
        # Always include metadata (only exact filenames, no wildcards for metadata)
        include_patterns.extend([
            'dataset_description.json',
            'participants.tsv'
        ])
        
        # Add subject/session specific patterns
        for sub in participants:
            for ses in sessions:
                if only_anat:
                    # Only anatomical data - use glob pattern
                    include_patterns.append(f"sub-{sub}/ses-{ses}/anat/*")
                else:
                    # All data for this session
                    include_patterns.append(f"sub-{sub}/ses-{ses}/*")
        
        logger.info(f"\nDownloading with {len(include_patterns)} include patterns...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"\nAttempt {attempt + 1}/{max_retries}...")
                openneuro.download(
                    dataset=dataset_id,
                    target_dir=str(bids_dir),
                    include=include_patterns
                )
                
                logger.info("\n✓ Download complete!")
                
                # Verify downloads
                logger.info("\nVerifying downloaded files...")
                for sub in participants:
                    for ses in sessions:
                        anat_dir = bids_dir / f"sub-{sub}" / f"ses-{ses}" / "anat"
                        if anat_dir.exists():
                            nii_files = list(anat_dir.glob("*.nii.gz"))
                            if nii_files:
                                logger.info(f"  ✓ sub-{sub} ses-{ses}: {len(nii_files)} NIfTI file(s)")
                            else:
                                logger.warning(f"  ⚠ sub-{sub} ses-{ses}: No NIfTI files found!")
                        else:
                            logger.error(f"  ✗ sub-{sub} ses-{ses}: Directory not found!")
                
                logger.info("\n" + "=" * 70)
                logger.info("✓ OpenNeuro download complete")
                logger.info("=" * 70 + "\n")
                return True
                
            except Exception as e:
                logger.error(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info("Retrying...")
                else:
                    logger.error("All download attempts failed")
                    return False
        
        return False
    
    def run_preprocessing(self) -> bool:
        """Run CAT12 preprocessing."""
        logger.info("=" * 70)
        logger.info("RUNNING CAT12 PREPROCESSING")
        logger.info("=" * 70)
        
        bids_config = self.config['preprocessing']['bids']
        proc_config = self.config['preprocessing']['processing']
        exec_config = self.config['preprocessing']['execution']
        smooth_config = self.config['preprocessing']['smoothing']
        val_config = self.config['preprocessing']['validation']
        
        # Build bids_dir path
        bids_dir = self.project_folder / bids_config['bids_dir']
        output_dir = self.project_folder / exec_config['output_dir']
        
        logger.info(f"BIDS directory: {bids_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Build cat12_prepro command
        cat12_prepro = Path(__file__).parent.parent.parent / "cat12_prepro"
        
        cmd = [
            str(cat12_prepro),
            str(bids_dir),
            str(output_dir),
            'participant'
        ]
        
        # Add processing options
        if proc_config.get('preproc', True):
            cmd.append('--preproc')
        
        if proc_config.get('no_surface', False):
            cmd.append('--no-surface')
        
        if proc_config.get('qa', False):
            cmd.append('--qa')
        
        if proc_config.get('tiv', False):
            cmd.append('--tiv')
        
        if proc_config.get('roi', False):
            cmd.append('--roi')
        
        # Add participants
        for sub in bids_config.get('participant_label', []):
            cmd.extend(['--participant-label', sub])
        
        # Add sessions
        for ses in bids_config.get('session_label', []):
            cmd.extend(['--session-label', ses])
        
        # Add smoothing
        for smooth_vol in smooth_config.get('smooth_volume', []):
            cmd.extend(['--smooth-volume', str(smooth_vol)])
        
        for smooth_surf in smooth_config.get('smooth_surface', []):
            cmd.extend(['--smooth-surface', str(smooth_surf)])
        
        # Add validation options
        if val_config.get('no_validate', False):
            cmd.append('--no-validate')
        
        if val_config.get('dry_run', False):
            cmd.append('--dry-run')
        
        logger.info(f"\nCommand: {' '.join(cmd)}\n")
        
        # Run preprocessing
        try:
            subprocess.run(cmd, check=True)
            logger.info("\n" + "=" * 70)
            logger.info("✓ Preprocessing complete")
            logger.info("=" * 70)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"\n✗ Preprocessing failed: {e}")
            return False
    
    def run_statistics(self) -> bool:
        """Run statistical analysis if enabled."""
        stats_config = self.config.get('statistics', {})
        
        if not stats_config.get('enabled', False):
            logger.info("Statistics disabled in config - skipping")
            return True
        
        logger.info("=" * 70)
        logger.info("RUNNING STATISTICAL ANALYSIS")
        logger.info("=" * 70)
        
        # TODO: Implement statistics pipeline
        logger.info("Statistics pipeline not yet implemented")
        logger.info("Preprocessing outputs available for manual analysis")
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ Statistics phase complete")
        logger.info("=" * 70 + "\n")
        
        return True
    
    def run(self) -> bool:
        """Run full pipeline."""
        logger.info("\n" + "=" * 70)
        logger.info(f"PROJECT: {self.config['study']['name']}")
        logger.info(f"CONFIG: {self.config_path}")
        logger.info(f"FOLDER: {self.project_folder}")
        logger.info("=" * 70 + "\n")
        
        # Step 1: Preflight checks
        if not self.run_preflight_checks():
            logger.error("Preflight checks failed. Aborting.")
            return False
        
        # Step 2: Validate subjects and sessions
        valid, _, _ = self.validate_subjects_sessions()
        if not valid:
            logger.error("Subject/session validation failed. Aborting.")
            return False
        
        # Step 3: Download OpenNeuro data
        if not self.download_openneuro():
            logger.error("OpenNeuro download failed. Aborting.")
            return False
        
        # Step 4: Run preprocessing
        if not self.run_preprocessing():
            logger.error("Preprocessing failed.")
            return False
        
        # Step 5: Run statistics
        if not self.run_statistics():
            logger.error("Statistics failed.")
            return False
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ PROJECT COMPLETED SUCCESSFULLY")
        logger.info("=" * 70 + "\n")
        
        return True


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python project_runner.py <config.json>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    try:
        runner = ProjectRunner(config_path)
        success = runner.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
