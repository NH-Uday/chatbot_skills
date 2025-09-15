import re

def format_equations_for_mathjax(text: str) -> str:
    # Replace common math patterns with LaTeX
    replacements = {
        "∂": "\\partial",
        "Δ": "\\Delta",
        "*": " \\cdot ",
        "^2": "^{2}",
        "^3": "^{3}",
    }

    for ascii_symbol, latex_symbol in replacements.items():
        text = text.replace(ascii_symbol, latex_symbol)

    # Wrap full equations that are on their own lines with $$...$$
    text = re.sub(r"(?m)^([^\n=]*=[^\n]*)$", r"$$\1$$", text)

    # Optional: convert inline fractions like (∂T/∂x)
    text = re.sub(
        r"\(\s*\\?partial\s*T\s*/\s*\\?partial\s*x\s*\)",
        r"\\left(\\frac{\\partial T}{\\partial x}\\right)",
        text,
    )

    return text
