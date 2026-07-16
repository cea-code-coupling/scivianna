#!/usr/bin/env bash

# Given a version number MAJOR.MINOR.PATCH, increment the:

# MAJOR version when you make incompatible API changes
# MINOR version when you add functionality in a backward compatible manner
# PATCH version when you make backward compatible bug fixes


# set -euo pipefail
unalias -a

readonly current_script_dir="$( cd "$( dirname "${0}" )" &> /dev/null && pwd )"
readonly project_root_dir="$(dirname ${current_script_dir})"
readonly version_file=${project_root_dir}/src/scivianna/VERSION

########################
# Main
########################

# Check current repo status
git_status="$(git status --porcelain)"
if [[ -n "${git_status}" ]]; then
    read -p "Current repository is not clean: '${git_status}'. Do you want to continue ? (yes/[no]) " answer
    if [[ -z "${answer}" ]]||[[ "${answer}" == "n"* ]]; then
        echo "Exiting after non clean repository, answer ${answer}"
        exit 0
    fi
fi

# Check version number
versions_tagged=$(git tag | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -1)
read -p  "Found version tagged ${versions_tagged}, new tag?" version_name

# Check input format
if ! (python3 -c "if not len('${version_name}'.split('.')) == 3: exit(1)"); then
    echo "ERROR: version format is not correct: '${version_name}', expected 'x.y.z'."
    exit 1
fi

for version_tagged in ${versions_tagged[@]}; do
    if ! (python3 -c "from packaging import version; exit( 1 if version.parse('${version_name}') <= version.parse('${version_tagged}') else 0)"); then
        echo "ERROR: version is lower or equal to existing one: '${version_name}' <= ${version_tagged}."
        exit 1
    fi
done


# Work in venv
if [ ! -d "${current_script_dir}/.venv_utils" ]; then
    python3 -m venv ${current_script_dir}/.venv_utils
fi
echo "Activating venv ${current_script_dir}/.venv_utils"
. ${current_script_dir}/.venv_utils/bin/activate
pip install --upgrade pip setuptools tox uv

# Tag the version
echo ${version_name} > ${version_file}
git add ${version_file}
git commit -m "Version ${version_name}"
git tag ${version_name}

# install current version in venv
python3 -m uv pip install --upgrade ${project_root_dir}

(cd ${project_root_dir} && tox)

# Push on origin
read -p "Do you want to push version '${version_name}' ? (yes/[no]) " answer
if [[ "${answer}" == "y"* ]]; then
    echo "Pushing tag..."
    git push origin
    git push origin ${version_name}
else
    echo "Reverting tag and aborting..."
    git reset --hard HEAD^
    git tag -d ${version_name}
    exit 0
fi

# Publish on pypi
read -p "Do you want to publish version '${version_name}' ? (yes/[no]) " answer
if [[ "${answer}" == "y"* ]]; then
    echo "Publishing..."
    if [ -d "dist" ]; then
        rm -rf dist
    fi
    python3 -m pip install --upgrade build twine
    python3 -m build
    python3 -m twine upload --repository pypi dist/*
else
    echo "You pushed a version tag but this version has not been published!"
    exit 1
fi