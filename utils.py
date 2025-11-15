import pickle
import logging
import os
import subprocess
import re


def githash():
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def write_to_local(path, obj, verbose=False):
    # write to local
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        logging.info(f"Write to local:/{path}")


def write_to(path, obj, verbose=False):
    write_to_local(path, obj, verbose=verbose)


def read_from_local(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def read_from(path):
    # read from local
    return read_from_local(path)


def camel_to_kebab(s: str) -> str:
    """Convert CamelCase or camelCase to kebab-case (lowercase with hyphens)."""
    if not s:
        return s
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1-\2', s)
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1-\2', s1)
    out = re.sub(r'[_\s]+', '-', s2)
    out = re.sub(r'-{2,}', '-', out)
    return out.strip('-').lower()

def kebab_to_camel(s: str) -> str:
    parts = [p for p in s.replace("_", "-").split("-") if p]
    return "".join(p.capitalize() for p in parts)

