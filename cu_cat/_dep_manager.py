import importlib


class DepManager:
    """Easily keep track of dependencies in a project.
    Parameters
    ----------
    package : str
        The name of the package to import.
    
    Attributes
    ----------
    pkgs : dict
        A dictionary of the imported packages.
        
    Examples
    --------
    >>> numpy = deps.numpy
    """
    def __init__(self):
        self.pkgs = {}

    def __getattr__(self, pkg: str):
        if pkg not in self.pkgs:
            self._add_deps(pkg)
        return self.pkgs.get(pkg)

    def _add_deps(self, pkg: str):
        try:
            pkg_val = importlib.import_module(pkg)
        except ModuleNotFoundError:
            # Optional dependency is absent. A broken wheel raises ImportError
            # (for example pyarrow refusing the installed NumPy) and must not
            # be swallowed: callers would otherwise see a later AttributeError
            # on None instead of the real install failure.
            self.pkgs[pkg] = None
            return
        self.pkgs[pkg] = pkg_val
        setattr(self, pkg, pkg_val)

    def import_from(self, pkg: str, name: str):
        try:
            module = __import__(pkg, fromlist=[name])
        except ModuleNotFoundError:
            self.pkgs[name] = None
            return None
        self.pkgs[name] = module
        return module


deps = DepManager()
