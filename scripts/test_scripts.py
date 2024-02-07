import pytest
from main import commands

@pytest.mark.parametrize("cmd", commands.values(), ids=list(commands.keys()))
def test_scripts_dry_run(cmd):
    """Just make sure all scripts work with dry run"""
    # Just add some dummy arguments
    args = {k:"" for k in cmd.script.__code__.co_varnames if k != "dry_run"}
    cmd.script(dry_run=True, **args)
