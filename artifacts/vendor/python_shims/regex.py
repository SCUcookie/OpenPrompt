import re as _re

IGNORECASE = _re.IGNORECASE


def _translate(pattern):
    if isinstance(pattern, str):
        return pattern.replace(r"\p{L}", "A-Za-z").replace(r"\p{N}", "0-9")
    return pattern


def compile(pattern, flags=0):
    return _re.compile(_translate(pattern), flags)


def findall(pattern, string, flags=0):
    return _re.findall(_translate(pattern), string, flags)
