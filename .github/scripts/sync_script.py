# SAVEPOINT CHECKPOINT 1
# THIS IS THE FIRST WORKING SAFEST CODE
# Next addition to the code would be getting the commit history and then fetching files that are most recent

from github import Github
import git
import os
import datetime
from github import Repository
from typing import List

# Retrieve the GitHub token from environment variables
github_token = os.environ['YOUR_GITHUB_TOKEN']

# Initialize the GitHub client with the token
g = Github(github_token)

def get_latest_file(repo_name, path):
    repo = g.get_repo(repo_name)
    commits = repo.get_commits(path=path)
    
    for commit in commits:
        files = commit.files
        for file in files:
            if file.filename.startswith(path):
                return repo.get_contents(file.filename)
    
    return None


def update_repo2(latest_file_content, repo2_name, file_path_in_repo2):
    repo2 = g.get_repo(repo2_name)
    # Create a new branch name based on current timestamp
    new_branch = 'update-' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    # Get the default branch (usually 'main')
    source_branch = repo2.default_branch
    source = repo2.get_branch(source_branch)

    # Create the new branch from the default branch
    repo2.create_git_ref(ref='refs/heads/' + new_branch, sha=source.commit.sha)

    # Include the token in the clone URL for authentication
    token = os.environ['YOUR_GITHUB_TOKEN']
    repo2_clone_url = repo2.clone_url.replace('https://', f'https://{token}@')

    # Clone repo2 locally and checkout the new branch
    repo2_dir = '/tmp/repo2'
    repo = git.Repo.clone_from(repo2.clone_url, repo2_dir)
    repo.git.checkout(new_branch)

    # Set Git config
    repo.git.config('user.email', 'no-reply@github.com')
    repo.git.config('user.name', 'Actions service account')

    # Update the remote URL to include authentication token
    token = os.environ['YOUR_GITHUB_TOKEN']
    repo_url = repo2.clone_url.replace('https://', f'https://{token}@')
    repo.git.remote('set-url', 'origin', repo_url)

    # Update the file in repo2
    file_path = os.path.join(repo2_dir, file_path_in_repo2)
    with open(file_path, 'r+') as file:
        lines = file.readlines()
        file.seek(0)

        # Keep the first seven lines as is
        file.writelines(lines[:7])

        # Add four blank lines
        file.write('\n' * 4)

        # Add the <details> and <summary> tags with the current date
        current_date = datetime.datetime.now().strftime("%B %d, %Y")  # e.g., "January 12, 2024"
        file.write(f"<details>\n<summary>{current_date}</summary>\n\n")

        # Add the new content starting from the 12th line onwards
        file.write(latest_file_content + "\n")

        # Close the <details> tag
        file.write("</details>\n")

        # Append the rest of the existing content
        file.writelines(lines[7:])

    # Commit and push changes
    repo.git.add(file_path_in_repo2)
    repo.git.commit('-m', 'Update performance-reports.md')
    repo.git.push('origin', new_branch)

    # Create a pull request
    repo2.create_pull(title="Update Performance Reports", body="Automated PR", head=new_branch, base=source_branch)

# Usage
latest_file = get_latest_file("saifeemustafaq/AIYA_December_2023", "ImageClassifierDoc")
latest_file_content = latest_file.decoded_content.decode("utf-8")
update_repo2(latest_file_content, "saifeemustafaq/OnboardingHarness", "JanBlogs/performance-reports.md")



'''
+++Links that helped you for authentication+++
https://github.com/actions/checkout/issues/13
https://github.com/actions/checkout/pull/1184
https://github.com/actions/runner-images/blob/7fa12b880649ddd0a00fed00b79a5bd4aed462ab/.github/workflows/merge_pull_request.yml#L19-L20


In the python file finally we used 
    # Set Git config
    repo.git.config('user.email', 'no-reply@github.com')
    repo.git.config('user.name', 'Actions service account')

    # Update the remote URL to include authentication token
    token = os.environ['YOUR_GITHUB_TOKEN']
    repo_url = repo2.clone_url.replace('https://', f'https://{token}@')
    repo.git.remote('set-url', 'origin', repo_url)

while in the yaml file we used:

      - name: Set Git Config
        run: |
          git config user.name "Actions service account"
          git config user.email "no-reply@github.com"

+++Links to understand basic YAML structure for github automation+++
https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions

'''
