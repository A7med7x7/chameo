import click
import subprocess
import shutil
import sys

TEMPLATE = "https://github.com/A7med7x7/chameo.git"

@click.group()
def cli():
    """
    Chameo CLI wrapper
    """
    pass

@cli.command()
@click.argument("path", required=False, default=".")
@click.option("--template", "-t", default=TEMPLATE, help="Template repo (hidden default).", hidden=True)
@click.option("--vcs-ref", default="main", show_default=True, help="Template branch/tag/commit")

def create(path, template, vcs_ref):
    """
    Create a new project from the ReproGen template.
    """
    if shutil.which("copier") is None:
        click.echo("copier executable not found. Install dependencies 'pip install copier'")
        sys.exit(2)

    cmd = ["copier", "copy", "--vcs-ref", vcs_ref, template, path]
    click.echo("Running: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"copier failed with exit code {e.returncode}")
        sys.exit(e.returncode)