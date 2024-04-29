"""
Utility functions for Conductor
"""


def clean_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\n", "")
    return string
