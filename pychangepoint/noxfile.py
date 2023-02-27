import nox


@nox.session
def test(session: nox.Session):
    session.install("maturin", "pytest", "pytest-benchmark")
    session.run("maturin", "develop")
    session.run("pytest", "tests")


@nox.session
def lint(session):
    session.install("flake8")
    session.run("flake8")
