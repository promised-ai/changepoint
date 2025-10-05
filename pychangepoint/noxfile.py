import nox


@nox.session
def test(session: nox.Session):
    session.install(
        "maturin",
        "pytest",
        "pytest-benchmark",
        "numpy",
    )
    session.run("maturin", "develop")
    session.run("py.test", "tests")


@nox.session
def test_notebook(session: nox.Session):
    session.install(
        "maturin",
        "pytest",
        "numpy",
        "nbval",
        "matplotlib",
        "seaborn",
        "scipy",
        "pandas",
    )
    session.run("maturin", "develop")
    session.run("py.test", "--nbval-lax", "ChangePointExample.ipynb")


@nox.session
def lint(session):
    session.install("ruff")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")
