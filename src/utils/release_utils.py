#!/usr/bin/env python3
"""
Release Utilities - Support library for getting release dates and computing gold versions

This module provides reusable functionality for:
- Fetching release dates from various version control systems (GitLab, GitHub)
- Computing gold versions based on record dates and release dates
- Supporting multiple software projects (Inkscape, GraphViz, OpenCV, Wireshark, OpenRGB)
"""

import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests
from dateutil import parser as date_parser


def compute_gold_version(record_date_str, release_dates):
    """
    Compute gold version based on record date and release dates.
    Returns the next version released after the record date.

    Args:
        record_date_str (str): The date string of the record
        release_dates (dict): Dictionary with version as key and release date as value

    Returns:
        str: The gold version or 'Unknown' if cannot be determined
    """
    if not record_date_str or record_date_str == 'NAN':
        return 'Unknown'
    # Parse record date
    record_date = date_parser.parse(record_date_str)

    # Convert release dates to datetime objects and sort by date
    release_list = []
    for version, date_str in release_dates.items():
        release_date = date_parser.parse(date_str)
        release_list.append((version, release_date))


    # Sort by release date
    release_list.sort(key=lambda x: x[1])

    # Find the next version released after the record date
    for version, release_date in release_list:
        if release_date > record_date:
            return version

    # If no version found after record date, return the latest version
    if release_list:
        return "not yet"

    return 'no_version'


def normalize_version_label(system_name: str, version: str) -> str:
    """
    Normalize version labels so they align with smell evolution files.

    Args:
        system_name: Subject system name.
        version: Raw version string computed from release dates.

    Returns:
        Canonicalized version label.
    """
    if not version:
        return version

    system = system_name.lower()
    special_versions = {'Unknown', 'not yet', 'no_version'}

    if system == 'fdroid' and version not in special_versions:
        if version.startswith('fdroidclient-'):
            return version
        return f"fdroidclient-{version}"

    return version


def get_inkscape_minor_version_release_dates():
    """
    Get Inkscape minor/non-patch version release dates from GitLab API.
    Returns a dictionary with version as key and release date as value.
    """
    # Use repository tags endpoint for inkscape/inkscape project
    url = "https://gitlab.com/api/v4/projects/3472737/repository/tags"

    # Read GitLab access token from secrets file
    try:
        with open('secrets/gitlab_secret.txt', 'r') as f:
            token = f.read().strip()
    except FileNotFoundError:
        print("Warning: secrets/gitlab_secret.txt not found. Proceeding without authentication.")
        token = None

    # Set up headers with authorization if token is available
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    try:
        # Get all tags (may need pagination)
        all_tags = []
        page = 1
        per_page = 100

        while True:
            params = {'page': page, 'per_page': per_page}
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            tags = response.json()

            if not tags:
                break

            all_tags.extend(tags)
            page += 1

            # Safety break to avoid infinite loops - increased limit for more releases
            if page > 100:
                break

        version_dates = {}

        for tag in all_tags:
            tag_name = tag.get('name', '')
            commit_date = tag.get('commit', {}).get('created_at')

            # Handle Inkscape tag naming convention (INKSCAPE_X_Y or INKSCAPE_X_Y_Z)
            if tag_name.startswith('INKSCAPE_'):
                # Convert INKSCAPE_1_2 to 1.2, INKSCAPE_1_2_0 to 1.2.0
                version_parts = tag_name.replace('INKSCAPE_', '').split('_')
                if len(version_parts) >= 2:
                    try:
                        # Ensure parts are numeric
                        int(version_parts[0])
                        int(version_parts[1])

                        # Filter for minor/non-patch versions (X_Y format, not X_Y_Z unless Z is 0)
                        is_minor = (len(version_parts) == 2 or
                                   (len(version_parts) >= 3 and version_parts[2] == '0'))

                        if is_minor and commit_date:
                            version = '.'.join(version_parts[:2])  # Convert to X.Y format
                            version_dates[version] = commit_date

                    except ValueError:
                        # Skip non-numeric versions
                        continue
        return version_dates

    except requests.RequestException as e:
        print(f"Error fetching Inkscape tag data: {e}")
        return {}


def get_graphviz_minor_version_release_dates():
    """
    Get GraphViz minor/non-patch version release dates from GitLab API.
    Returns a dictionary with version as key and release date as value.
    """
    # Use GitLab repository tags endpoint for graphviz/graphviz project (ID: 4207231)
    url = "https://gitlab.com/api/v4/projects/4207231/repository/tags"

    # Read GitLab access token from secrets file
    try:
        with open('secrets/gitlab_secret.txt', 'r') as f:
            token = f.read().strip()
    except FileNotFoundError:
        print("Warning: secrets/gitlab_secret.txt not found. Proceeding without authentication.")
        token = None

    # Set up headers with authorization if token is available
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    try:
        # Get all tags (may need pagination)
        all_tags = []
        page = 1
        per_page = 100

        while True:
            params = {'page': page, 'per_page': per_page}
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            tags = response.json()

            if not tags:
                break

            all_tags.extend(tags)
            page += 1

            # Safety break to avoid infinite loops
            if page > 50:
                break

        version_dates = {}

        for tag in all_tags:
            tag_name = tag.get('name', '')
            commit_date = tag.get('commit', {}).get('created_at')

            # Handle GraphViz tag naming convention (X.Y.Z format)
            if tag_name and commit_date:
                # Remove any 'v' prefix and parse version
                clean_version = tag_name.lstrip('v')
                version_parts = clean_version.split('.')

                if len(version_parts) >= 2:
                    try:
                        # Ensure parts are numeric
                        int(version_parts[0])
                        int(version_parts[1])

                        # Filter for minor/non-patch versions (X.Y.0 or X.Y format)
                        is_minor = (len(version_parts) == 2 or
                                   (len(version_parts) >= 3 and version_parts[2] == '0'))

                        if is_minor:
                            version = '.'.join(version_parts[:2])  # Convert to X.Y format
                            version_dates[version] = commit_date

                    except (ValueError, IndexError):
                        # Skip non-numeric or malformed versions
                        continue

        return version_dates

    except requests.RequestException as e:
        print(f"Error fetching GraphViz tag data: {e}")
        return {}


def get_fdroid_version_release_dates():
    """
    Get F-Droid client version release dates from GitLab tags.
    Returns a dictionary with version as key and release date as value.
    """
    url = "https://gitlab.com/api/v4/projects/fdroid%2Ffdroidclient/repository/tags"

    try:
        with open('secrets/gitlab_secret.txt', 'r') as f:
            token = f.read().strip()
    except FileNotFoundError:
        print("Warning: secrets/gitlab_secret.txt not found. Proceeding without authentication.")
        token = None

    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    try:
        version_dates = {}
        page = 1
        per_page = 100

        while True:
            params = {'page': page, 'per_page': per_page}
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            tags = response.json()

            if not tags:
                break

            for tag in tags:
                tag_name = tag.get('name', '')
                commit_date = tag.get('commit', {}).get('created_at')
                if not tag_name or not commit_date:
                    continue

                # Normalize tag names like "v1.23.0" or "1_23_0"
                clean_name = tag_name.lstrip('v').replace('_', '.')

                if re.match(r'^\d+(\.\d+){1,2}$', clean_name):
                    version_dates[clean_name] = commit_date

            if len(tags) < per_page:
                break
            page += 1

        return version_dates

    except requests.RequestException as e:
        print(f"Error fetching F-Droid tag data: {e}")
        return {}


def get_opencv_minor_version_release_dates():
    """
    Get OpenCV minor/non-patch version release dates from GitHub API.
    Returns a dictionary with version as key and release date as value.
    """
    # Use GitHub repository releases endpoint for opencv/opencv
    url = "https://api.github.com/repos/opencv/opencv/releases"

    # Read GitHub access token from secrets file (optional)
    try:
        with open('secrets/github_secret.txt', 'r') as f:
            token = f.read().strip()
    except FileNotFoundError:
        print("Warning: secrets/github_secret.txt not found. Proceeding without authentication.")
        token = None

    # Set up headers with authorization if token is available
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if token:
        headers['Authorization'] = f'token {token}'

    try:
        # Get all releases
        all_releases = []
        page = 1
        per_page = 100

        while True:
            params = {'page': page, 'per_page': per_page}
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            releases = response.json()

            if not releases:
                break

            all_releases.extend(releases)
            page += 1

            if page > 50:
                break

        version_dates = {}

        for release in all_releases:
            tag_name = release.get('tag_name', '')
            published_at = release.get('published_at')

            if tag_name and published_at:
                # Remove any prefix and parse version
                clean_version = tag_name.lstrip('v')
                version_parts = clean_version.split('.')

                if len(version_parts) >= 2:
                    try:
                        int(version_parts[0])
                        int(version_parts[1])

                        # Filter for minor versions (X.Y.0 format)
                        is_minor = (len(version_parts) >= 3 and version_parts[2] == '0')

                        if is_minor:
                            version = '.'.join(version_parts[:2])
                            version_dates[version] = published_at

                    except (ValueError, IndexError):
                        continue

        return version_dates

    except requests.RequestException as e:
        print(f"Error fetching OpenCV release data: {e}")
        return {}


def get_wireshark_minor_version_release_dates():
    """
    Get Wireshark minor/non-patch version release dates from GitLab API.
    Returns a dictionary with version as key and release date as value.
    """
    # Use GitLab repository tags endpoint for wireshark/wireshark project (ID: 7898047)
    url = "https://gitlab.com/api/v4/projects/7898047/repository/tags"

    # Read GitLab access token from secrets file
    try:
        with open('secrets/gitlab_secret.txt', 'r') as f:
            token = f.read().strip()
    except FileNotFoundError:
        print("Warning: secrets/gitlab_secret.txt not found. Proceeding without authentication.")
        token = None

    # Set up headers with authorization if token is available
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    try:
        # Get all tags (may need pagination)
        all_tags = []
        page = 1
        per_page = 100

        while True:
            params = {'page': page, 'per_page': per_page}
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            tags = response.json()

            if not tags:
                break

            all_tags.extend(tags)
            page += 1

            # Safety break to avoid infinite loops
            if page > 50:
                break

        version_dates = {}

        for tag in all_tags:
            tag_name = tag.get('name', '')
            commit_date = tag.get('commit', {}).get('created_at')

            # Handle Wireshark tag naming convention (vX.Y.Z format)
            if tag_name and commit_date:
                clean_version = tag_name.lstrip('v')
                version_parts = clean_version.split('.')

                if len(version_parts) >= 2:
                    try:
                        int(version_parts[0])
                        int(version_parts[1])

                        # Filter for minor versions (X.Y.0 format)
                        is_minor = (len(version_parts) >= 3 and version_parts[2] == '0')

                        if is_minor:
                            version = '.'.join(version_parts[:2])
                            version_dates[version] = commit_date

                    except (ValueError, IndexError):
                        continue

        return version_dates

    except requests.RequestException as e:
        print(f"Error fetching Wireshark tag data: {e}")
        return {}


def get_openrgb_minor_version_release_dates():
    """
    Get OpenRGB minor/non-patch version release dates from GitLab API.
    Returns a dictionary with version as key and release date as value.
    """
    # Use GitLab repository tags endpoint for CalcProgrammer1/OpenRGB project (ID: 10582521)
    url = "https://gitlab.com/api/v4/projects/10582521/repository/tags"

    # Read GitLab access token from secrets file
    try:
        with open('secrets/gitlab_secret.txt', 'r') as f:
            token = f.read().strip()
    except FileNotFoundError:
        print("Warning: secrets/gitlab_secret.txt not found. Proceeding without authentication.")
        token = None

    # Set up headers with authorization if token is available
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    try:
        # Get all tags (may need pagination)
        all_tags = []
        page = 1
        per_page = 100

        while True:
            params = {'page': page, 'per_page': per_page}
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            tags = response.json()

            if not tags:
                break

            all_tags.extend(tags)
            page += 1

            # Safety break to avoid infinite loops
            if page > 50:
                break

        version_dates = {}

        for tag in all_tags:
            tag_name = tag.get('name', '')
            commit_date = tag.get('commit', {}).get('created_at')

            if tag_name and commit_date:
                # OpenRGB uses format like "release_0.9"
                if tag_name.startswith('release_'):
                    version_str = tag_name.replace('release_', '')
                    version_parts = version_str.split('.')

                    if len(version_parts) >= 2:
                        try:
                            int(version_parts[0])
                            int(version_parts[1])

                            # For OpenRGB, consider X.Y as minor versions
                            version = '.'.join(version_parts[:2])
                            version_dates[version] = commit_date

                        except (ValueError, IndexError):
                            continue

        return version_dates

    except requests.RequestException as e:
        print(f"Error fetching OpenRGB tag data: {e}")
        return {}


def get_stackgres_minor_version_release_dates():
    """
    Get StackGres minor/non-patch release dates from GitLab tags.

    The dataset uses major.minor(.0) labels (e.g., 1.10.0, 0.5), so we keep
    those shapes and ignore patch releases beyond .0.
    """
    url = "https://gitlab.com/api/v4/projects/ongresinc%2Fstackgres/repository/tags"

    try:
        with open('secrets/gitlab_secret.txt', 'r') as f:
            token = f.read().strip()
    except FileNotFoundError:
        print("Warning: secrets/gitlab_secret.txt not found. Proceeding without authentication.")
        token = None

    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    try:
        version_dates = {}
        page = 1
        per_page = 100

        while True:
            params = {'page': page, 'per_page': per_page}
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            tags = response.json()

            if not tags:
                break

            for tag in tags:
                tag_name = tag.get('name', '')
                commit_date = tag.get('commit', {}).get('created_at')
                if not tag_name or not commit_date:
                    continue

                clean_name = tag_name.lstrip('v').replace('_', '.')
                clean_name = re.sub(r'^stackgres[-_]*', '', clean_name, flags=re.IGNORECASE)
                clean_name = clean_name.split('-', 1)[0]

                match = re.match(r'^(\d+(?:\.\d+){1,2})$', clean_name)
                if not match:
                    continue

                version_parts = match.group(1).split('.')
                try:
                    # Require at least major.minor; keep patch when it is 0
                    int(version_parts[0])
                    int(version_parts[1])
                    if len(version_parts) == 2:
                        version = '.'.join(version_parts)
                    elif len(version_parts) >= 3 and version_parts[2] == '0':
                        version = '.'.join(version_parts[:3])
                    else:
                        continue
                except (ValueError, IndexError):
                    continue

                version_dates[version] = commit_date

            if len(tags) < per_page:
                break
            page += 1

        return version_dates

    except requests.RequestException as e:
        print(f"Error fetching StackGres tag data: {e}")
        return {}


def get_shepard_minor_version_release_dates():
    """
    Get Shepard minor/non-patch release dates from GitLab tags.

    Shepard mixes semantic versions (e.g., 5.1.0) and date-based tags
    (e.g., 2024.07.04); both forms are kept.
    """
    url = "https://gitlab.com/api/v4/projects/dlr-shepard%2Fshepard/repository/tags"

    try:
        with open('secrets/gitlab_secret.txt', 'r') as f:
            token = f.read().strip()
    except FileNotFoundError:
        print("Warning: secrets/gitlab_secret.txt not found. Proceeding without authentication.")
        token = None

    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    try:
        version_dates = {}
        page = 1
        per_page = 100

        while True:
            params = {'page': page, 'per_page': per_page}
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            tags = response.json()

            if not tags:
                break

            for tag in tags:
                tag_name = tag.get('name', '')
                commit_date = tag.get('commit', {}).get('created_at')
                if not tag_name or not commit_date:
                    continue

                clean_name = tag_name.lstrip('v').replace('_', '.')
                clean_name = re.sub(r'^shepard[-_]*', '', clean_name, flags=re.IGNORECASE)
                clean_name = clean_name.split('-', 1)[0]

                match = re.match(r'^(\d+(?:\.\d+){1,2})$', clean_name)
                if not match:
                    continue

                version_parts = match.group(1).split('.')
                try:
                    # Accept semantic minor releases and date-like tags (YYYY.MM.DD).
                    int(version_parts[0])
                    int(version_parts[1])
                    if len(version_parts) == 2:
                        version = '.'.join(version_parts)
                    elif len(version_parts) >= 3:
                        if len(version_parts[0]) == 4 or version_parts[2] == '0':
                            version = '.'.join(version_parts[:3])
                        else:
                            continue
                    else:
                        continue
                except (ValueError, IndexError):
                    continue

                version_dates[version] = commit_date

            if len(tags) < per_page:
                break
            page += 1

        return version_dates

    except requests.RequestException as e:
        print(f"Error fetching Shepard tag data: {e}")
        return {}


def get_release_dates_for_system(system_name):
    """
    Get release dates based on the system name.

    Args:
        system_name (str): Name of the system (inkscape, graphviz, opencv, wireshark, openrgb, fdroid)

    Returns:
        dict: Dictionary with version as key and release date as value
    """
    system_functions = {
        'inkscape': get_inkscape_minor_version_release_dates,
        'graphviz': get_graphviz_minor_version_release_dates,
        'opencv': get_opencv_minor_version_release_dates,
        'wireshark': get_wireshark_minor_version_release_dates,
        'openrgb': get_openrgb_minor_version_release_dates,
        'fdroid': get_fdroid_version_release_dates,
        'ongresinc-stackgres': get_stackgres_minor_version_release_dates,
        'stackgres': get_stackgres_minor_version_release_dates,
        'dlr-shepard-shepard': get_shepard_minor_version_release_dates,
        'shepard': get_shepard_minor_version_release_dates,
    }

    if system_name.lower() in system_functions:
        print(f"Getting release dates for {system_name}...")
        return system_functions[system_name.lower()]()
    else:
        print(f"Warning: No release date function for system '{system_name}'. Using Inkscape as fallback.")
        return get_inkscape_minor_version_release_dates()


def detect_subject_system(data_file_path, system_name=None):
    """
    Detect subject system from data file path or system_name parameter.

    Args:
        data_file_path (str): Path to the data file
        system_name (str, optional): Explicit system name

    Returns:
        str: Detected system name
    """
    if system_name:
        return system_name.lower()

    # Try to detect from file path
    data_path = Path(data_file_path)
    filename = data_path.name.lower()

    # Common system patterns
    system_patterns = {
        'inkscape': ['inkscape'],
        'graphviz': ['graphviz'],
        'opencv': ['opencv'],
        'wireshark': ['wireshark'],
        'openrgb': ['openrgb'],
        'fdroid': ['fdroid', 'fdroidclient']
    }

    for system, patterns in system_patterns.items():
        if any(pattern in filename for pattern in patterns):
            return system

    # Check parent directories
    for parent in data_path.parents:
        parent_name = parent.name.lower()
        for system, patterns in system_patterns.items():
            if any(pattern in parent_name for pattern in patterns):
                return system

    print("Warning: Could not detect subject system. Defaulting to 'inkscape'")
    return 'inkscape'
