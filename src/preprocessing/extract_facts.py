import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
UND_EXECUTABLE = REPO_ROOT / "Understand-7.0.1212-Linux-64bit" / "scitools" / "bin" / "linux64" / "und"

if __package__:
    from .constants import FACTS_ROOT, SUBJECT_SYSTEMS_ROOT, STOPWORDS_DIR_PATH, time_print
else:  # pragma: no cover - script execution fallback
    sys.path.append(str(REPO_ROOT))
    from src.preprocessing.constants import FACTS_ROOT, SUBJECT_SYSTEMS_ROOT, STOPWORDS_DIR_PATH, time_print

from src.utils.system_paths import resolve_system_subdir

def gen_doc_topics(project_name: str, artifacts_path: str):
  time_print("Generating DocTopics.")
  subprocess.run(["java", "-Xmx8g", "-cp", "ARCADE_Core.jar", "edu.usc.softarch.arcade.topics.DocTopics", "mode=generate", f"artifacts={artifacts_path}", f"project={project_name}", "filelevel=true", "overwrite=true"])

def run_understand(system_name: str, version: str, language: str):
  """
  Run SciTools Understand on a given system version to extract its dependencies.

  Parameters:
    system_name: The name of the system to run Understand on.
    version: The version of the system to run Understand on.
    language: The language of the system.
  """
  system_path = resolve_system_subdir(SUBJECT_SYSTEMS_ROOT, system_name)
  version_path = system_path / version
  und_path = version_path / f"{version}.und"
  deps_dir = system_path / "deps"
  deps_dir.mkdir(parents=True, exist_ok=True)
  deps_path = deps_dir / f"{version}_deps.csv"

  und_binary = UND_EXECUTABLE if UND_EXECUTABLE.exists() else Path("und")
  if und_binary == Path("und"):
    time_print("Warning: bundled Understand binary not found; falling back to system 'und'.")
  time_print(f"Creating UND project at {und_path}.")
  if language != "c":
    subprocess.run([str(und_binary), "-quiet", "create", "-languages", language, str(und_path)])
  else:
    subprocess.run([str(und_binary), "-quiet", "create", "-languages", "c++", str(und_path)])
  time_print(f"Adding all files from {version_path}.")
  subprocess.run([str(und_binary), "-quiet", "add", str(version_path), str(und_path)])
  time_print(f"Running UND analysis.")
  subprocess.run([str(und_binary), "-quiet", "analyze", str(und_path)])
  time_print(f"Exporting dependencies to {deps_path}.")
  subprocess.run(
    [
      str(und_binary),
      "-quiet",
      "export",
      "-dependencies",
      "-format",
      "long",
      "file",
      "csv",
      str(deps_path),
      str(und_path),
    ]
  )

def csv_to_rsf(input_path: str, output_path: str, project_root_name: str):
  """
  Convert a dependencies CSV file to an RSF file using the UnderstandCsvToRsf utility.

  Parameters:
    input_path: The path to the input CSV file.
    output_path: The path to the output RSF file.
    project_root_name: The root name of the project, used for filtering.
  """
  time_print("Parsing Understand CSV dependencies to RSF.")
  subprocess.run(["java", "-cp", "ARCADE_Core.jar", "edu.usc.softarch.arcade.facts.dependencies.UnderstandCsvToRsf", input_path, output_path, project_root_name])

def run_mallet(system_root: str, language: str, artifacts_output_path: str, selected_versions: Optional[List[str]] = None):
  """
  Run Mallet on a given system and generate artifacts.

  Parameters:
    system_root: The root directory of the system to run Mallet on.
    language: The language of the system.
    artifacts_output_path: The path to the output directory for the generated artifacts.
    selected_versions: Optional list of specific versions to run Mallet on. If None, runs on all versions.
  """
  if selected_versions:
    # Create temporary directory with selected versions only
    import tempfile
    import shutil

    with tempfile.TemporaryDirectory() as temp_dir:
      temp_system_root = os.path.join(temp_dir, os.path.basename(system_root))
      os.makedirs(temp_system_root, exist_ok=True)

      # Copy only selected versions to temp directory
      for version in selected_versions:
        src_path = f"{system_root}/{version}"
        if os.path.exists(src_path):
          dst_path = f"{temp_system_root}/{version}"
          time_print(f"Copying version {version} for Mallet processing.")
          shutil.copytree(src_path, dst_path)
        else:
          time_print(f"Warning: Version {version} not found in {system_root}")

      time_print(f"Running Mallet on selected versions: {', '.join(selected_versions)}.")
      subprocess.run(["java", "-cp", "ARCADE_Core.jar", "edu.usc.softarch.arcade.topics.MalletRunner", temp_system_root, language, artifacts_output_path, STOPWORDS_DIR_PATH])
  else:
    time_print(f"Running Mallet on all versions in {system_root}.")
    subprocess.run(["java", "-cp", "ARCADE_Core.jar", "edu.usc.softarch.arcade.topics.MalletRunner", system_root, language, artifacts_output_path, STOPWORDS_DIR_PATH])

def extract_facts(
    system_name: str,
    language: str,
    mallet_mode: str = "all",
    selected_versions: Optional[List[str]] = None,
) -> None:
  """
  Extract facts from a given system.

  Parameters:
    system_name: The name of the system to extract facts from.
    language: The language of the system.
    mallet_mode: Mode for running Mallet - "all" for all versions, "selected" for specific versions.
    selected_versions: List of specific versions to run Mallet on (used when mallet_mode="selected").
  """
  system_root = resolve_system_subdir(SUBJECT_SYSTEMS_ROOT, system_name)
  facts_dir = resolve_system_subdir(FACTS_ROOT, system_name, create=True)
  time_print(f"Using facts directory at {facts_dir}.")
  deps_dir = system_root / "deps"
  time_print(f"Ensuring Understand dependencies directory at {deps_dir}.")
  deps_dir.mkdir(parents=True, exist_ok=True)
  print()

  # Parse system Understand dependencies
  for entry in glob.glob(f"{system_root}/{system_name}-*"):
    prefix = f"{system_root}/"
    version = entry[len(prefix):]
    time_print(f"Extracting dependencies for {version}.")

    # Dependencies
    input_path = f"{system_root}/deps/{version}_deps.csv"
    output_path = f"{FACTS_ROOT}/{system_name}/{version}_deps.rsf"
    run_understand(system_name, version, language)
    project_root_name = version
    csv_to_rsf(input_path, output_path, project_root_name)
    time_print(f"Dependencies extracted for {version}.")
    print()

  # Mallet
  # artifacts_output_path = facts_dir / "artifacts"
  # os.makedirs(artifacts_output_path, exist_ok=True)
  #
  # if mallet_mode == "selected" and selected_versions:
  #   run_mallet(str(system_root), language, str(artifacts_output_path), selected_versions)
  # else:
  #   run_mallet(str(system_root), language, str(artifacts_output_path))
  #
  # gen_doc_topics(system_name, str(artifacts_output_path))
  # time_print("All facts extracted.")

def main():
  parser = argparse.ArgumentParser(description="Extract facts from a software system")
  parser.add_argument("system_name", help="The name of the system to extract facts from")
  parser.add_argument("language", help="The programming language of the system")
  parser.add_argument("--mallet-mode", choices=["all", "selected"], default="all",
                     help="Mode for running Mallet: 'all' for all versions, 'selected' for specific versions")
  parser.add_argument("--versions", nargs="+", help="Specific versions to run Mallet on (required when mallet-mode=selected)")

  args = parser.parse_args()

  if args.mallet_mode == "selected" and not args.versions:
    parser.error("--versions is required when --mallet-mode=selected")

  extract_facts(args.system_name, args.language, args.mallet_mode, args.versions)

if __name__ == "__main__":
  main()
