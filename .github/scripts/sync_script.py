from github import Github
import git
import os
import datetime

# Retrieve the GitHub token from environment variables
github_token = os.environ['YOUR_GITHUB_TOKEN']

# Initialize the GitHub client with the token
g = Github(github_token)

def get_latest_file(repo_name, path):
    repo = g.get_repo(repo_name)
    contents = repo.get_contents(path)
    latest_file = contents[-1]  # Assuming the latest file is the last in the list
    return latest_file

def update_repo2(latest_file_content, repo2_name, file_path_in_repo2):
    repo2 = g.get_repo(repo2_name)
    # Create a new branch name based on current timestamp
    new_branch = 'update-' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    # Get the default branch (usually 'main')
    source_branch = repo2.default_branch
    source = repo2.get_branch(source_branch)

    # Create the new branch from the default branch
    repo2.create_git_ref(ref='refs/heads/' + new_branch, sha=source.commit.sha)

    # Clone repo2 locally and checkout the new branch
    repo2_dir = '/tmp/repo2'
    repo = git.Repo.clone_from(repo2.clone_url, repo2_dir)
    repo.git.checkout(new_branch)

    # Set Git config
    #repo.git.config('user.email', 'github-actions@github.com')
    #repo.git.config('user.name', 'github-actions[bot]')

    # Update the file in repo2
    file_path = os.path.join(repo2_dir, file_path_in_repo2)
    with open(file_path, 'r+') as file:
        original_content = file.read()
        file.seek(0)
        file.write(latest_file_content + "\n" + original_content)

    # Commit and push changes
    repo.git.add(file_path_in_repo2)
    repo.git.commit('-m', 'Update performance-reports.md')
    repo.git.push('origin', new_branch)

    # Create a pull request
    repo2.create_pull(title="Update Performance Reports", body="Automated PR", head=new_branch, base=source_branch)

# Usage
latest_file = get_latest_file("saifeemustafaq/AIYA_December_2023", "ImageClassifierDoc")
latest_file_content = latest_file.decoded_content.decode("utf-8")
update_repo2(latest_file_content, "saifeemustafaq/OnboardingHarness", "JanBlogs/test.md")
